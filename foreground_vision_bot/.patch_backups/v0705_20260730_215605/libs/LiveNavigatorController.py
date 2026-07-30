from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from pathlib import Path
from time import monotonic, sleep
from typing import Any, Callable

import numpy as np

from mapper.rl.LayoutSources import FullRealMapGenerator, RealMapData
from mapper.rl.NavigatorCore import (
    NavigatorAction,
    NavigatorOutcome,
    NavigatorSimulatorConfig,
    NavigatorSimulatorCore,
    compute_distance_field,
)
from mapper.rl.NavigatorTraining import load_navigator_config
from project_paths import resolve_app_path
from worker_manager import CancellationToken

from .NativeMapContext import NativeMapContext
from .NavigatorActionExecutor import NavigatorActionExecutor


@dataclass(frozen=True, slots=True)
class LiveNavigatorConfig:
    model_path: str = "models/movement/navigator_ppo_final_offline.zip"
    training_config_path: str = "mapper/rl/navigator_training.json"
    decision_burst_seconds: float = 0.60
    goal_tolerance_cells: float = 2.0
    pose_retry_seconds: float = 0.05
    maximum_pose_wait_seconds: float = 1.5
    stuck_after_seconds: float = 1.5
    stuck_minimum_progress_cells: float = 0.35
    backward_recovery_seconds: float = 0.35

    def __post_init__(self) -> None:
        if self.decision_burst_seconds <= 0.0:
            raise ValueError("decision_burst_seconds must be positive")
        if self.goal_tolerance_cells < 0.0:
            raise ValueError("goal_tolerance_cells cannot be negative")
        if self.maximum_pose_wait_seconds <= 0.0:
            raise ValueError("maximum_pose_wait_seconds must be positive")
        if self.stuck_after_seconds <= 0.0:
            raise ValueError("stuck_after_seconds must be positive")
        if self.backward_recovery_seconds <= 0.0:
            raise ValueError("backward_recovery_seconds must be positive")


@dataclass(frozen=True, slots=True)
class NavigationBurstResult:
    goal_cell: tuple[int, int]
    actions: int
    elapsed_seconds: float
    initial_distance_cells: float
    final_distance_cells: float
    arrived: bool
    recovery_used: bool
    last_action: str


class _PolicyAdapter:
    def __init__(self, model: Any, *, supports_masks: bool) -> None:
        self.model = model
        self.supports_masks = bool(supports_masks)

    @classmethod
    def load(cls, path: Path) -> "_PolicyAdapter":
        if not path.is_file():
            alternative = path.with_suffix(".zip")
            if alternative.is_file():
                path = alternative
            else:
                raise FileNotFoundError(
                    f"Frozen movement navigator is missing: {path}"
                )
        try:
            from sb3_contrib import MaskablePPO

            return cls(MaskablePPO.load(str(path)), supports_masks=True)
        except Exception as mask_error:  # noqa: BLE001 - support older PPO exports.
            try:
                from stable_baselines3 import PPO

                return cls(PPO.load(str(path)), supports_masks=False)
            except Exception as ppo_error:  # noqa: BLE001 - include both causes.
                raise RuntimeError(
                    "Could not load the movement navigator as MaskablePPO or PPO. "
                    f"MaskablePPO: {mask_error}; PPO: {ppo_error}"
                ) from ppo_error

    def predict(
        self,
        observation: dict[str, np.ndarray],
        action_masks: np.ndarray,
    ) -> NavigatorAction:
        if self.supports_masks:
            action, _state = self.model.predict(
                observation,
                deterministic=True,
                action_masks=np.asarray(action_masks, dtype=np.bool_),
            )
        else:
            action, _state = self.model.predict(
                observation,
                deterministic=True,
            )
        value = int(action.item() if hasattr(action, "item") else action)
        return NavigatorAction(value)


class LiveNavigatorController:
    """Run the frozen goal-conditioned movement policy against the live map."""

    def __init__(
        self,
        bot,
        map_context: NativeMapContext,
        *,
        config: LiveNavigatorConfig | None = None,
        cancellation: CancellationToken | None = None,
        status_callback: Callable[[str], None] | None = None,
        policy: _PolicyAdapter | None = None,
    ) -> None:
        self.bot = bot
        self.map_context = map_context
        self.config = config or LiveNavigatorConfig()
        self.cancellation = cancellation or CancellationToken()
        self.status_callback = status_callback
        if bot.keyboard is None:
            raise RuntimeError("Keyboard input is unavailable")
        self.executor = NavigatorActionExecutor(bot.keyboard)

        model_path = resolve_app_path(self.config.model_path)
        self.policy = policy or _PolicyAdapter.load(model_path)
        simulator = self._load_simulator_config()
        # The live map already includes the final conservative occupancy. Keep
        # the same safety inflation and observation dimensions as training.
        data = RealMapData(
            occupancy=np.zeros_like(
                self.map_context.layout.traversable,
                dtype=np.uint8,
            ),
            traversable=self.map_context.layout.traversable.copy(),
            forbidden=self.map_context.layout.forbidden.copy(),
            map_directory=self.map_context.map_directory,
            map_name=self.map_context.map_name,
            source_bounds=self.map_context.source_bounds,
        )
        self.core = NavigatorSimulatorCore(
            config=simulator,
            generator=FullRealMapGenerator(data),
        )
        self.core.reset(seed=0)
        self.core.layout = self.map_context.layout
        self.core.safe_traversable = self.map_context.safe_traversable.copy()
        self._last_pose_cell: tuple[float, float] | None = None
        self._last_progress_at = monotonic()
        self._last_goal: tuple[int, int] | None = None

    def _load_simulator_config(self) -> NavigatorSimulatorConfig:
        configured = resolve_app_path(self.config.training_config_path)
        if configured.is_file():
            try:
                return load_navigator_config(configured).simulator
            except Exception as error:  # noqa: BLE001 - fallback remains compatible.
                self._status(
                    f"Could not load navigator config {configured}: {error}; "
                    "using built-in v0.6.4 defaults."
                )
        return NavigatorSimulatorConfig()

    def _status(self, message: str) -> None:
        if self.status_callback is not None:
            self.status_callback(message)

    def stop(self) -> None:
        self.executor.stop()

    def cast_eva(self) -> None:
        self.stop()
        if self.bot.keyboard is None:
            raise RuntimeError("Keyboard input is unavailable")
        self.bot.keyboard.press_key(0x70, press_time=0.03)

    def navigate_toward_native(
        self,
        native_x: float,
        native_z: float,
        *,
        duration_seconds: float | None = None,
    ) -> NavigationBurstResult:
        target_layout = self.map_context.native_to_layout_cells(native_x, native_z)
        goal = self.map_context.nearest_safe_cell(target_layout)
        if goal is None:
            raise RuntimeError(
                f"No safe destination near native target ({native_x:.2f}, {native_z:.2f})"
            )
        return self.navigate_toward_cell(
            goal,
            duration_seconds=duration_seconds,
        )

    def navigate_toward_cell(
        self,
        goal: tuple[int, int],
        *,
        duration_seconds: float | None = None,
    ) -> NavigationBurstResult:
        duration = (
            self.config.decision_burst_seconds
            if duration_seconds is None
            else max(0.05, float(duration_seconds))
        )
        if not self.map_context.inside_layout(goal):
            raise ValueError(f"Goal is outside the map: {goal}")
        if not self.map_context.safe_traversable[goal[1], goal[0]]:
            replacement = self.map_context.nearest_safe_cell(goal)
            if replacement is None:
                raise RuntimeError(f"Goal has no nearby safe cell: {goal}")
            goal = replacement

        self.core.goal = (int(goal[0]), int(goal[1]))
        self.core.distance_field = compute_distance_field(
            self.map_context.safe_traversable,
            self.core.goal,
        )
        self._last_goal = self.core.goal
        started = monotonic()
        actions = 0
        recovery_used = False
        last_action = "NONE"
        initial_distance = float("inf")
        final_distance = float("inf")

        try:
            while (
                monotonic() - started < duration
                and not self.cancellation.cancelled
                and self.bot.rl_enabled
            ):
                self._require_foreground()
                pose = self._wait_for_navigation_pose()
                layout_position = self.map_context.native_to_layout_cells(
                    pose.x,
                    pose.z,
                )
                position = self.map_context.nearest_safe_cell(
                    layout_position,
                    maximum_radius=5,
                )
                if position is None:
                    raise RuntimeError(
                        "Player position is not on or near a safe mapped cell"
                    )
                distance = float(self.core.distance_field[position[1], position[0]])
                if not isfinite(distance):
                    raise RuntimeError("Selected destination is unreachable on the map")
                if not isfinite(initial_distance):
                    initial_distance = distance
                final_distance = distance
                if distance <= self.config.goal_tolerance_cells:
                    break

                heading = pose.heading_degrees
                if heading is None:
                    raise RuntimeError(
                        "Navigator heading is unavailable; keep Bot Vision enabled "
                        "until the minimap arrow is reacquired"
                    )
                self._sync_core(position, layout_position, float(heading))
                masks = self.core.action_masks()
                action = self.policy.predict(self.core.observation(), masks)
                last_action = action.name
                before = layout_position
                before_distance = distance
                self.executor.execute(action)
                action_seconds = self._action_seconds(action, distance)
                self._wait(action_seconds)
                actions += 1

                after_pose = self._wait_for_navigation_pose()
                after = self.map_context.native_to_layout_cells(
                    after_pose.x,
                    after_pose.z,
                )
                after_cell = self.map_context.nearest_safe_cell(after, maximum_radius=5)
                after_distance = before_distance
                if after_cell is not None:
                    candidate = float(
                        self.core.distance_field[after_cell[1], after_cell[0]]
                    )
                    if isfinite(candidate):
                        after_distance = candidate
                        final_distance = candidate
                moved = hypot(after[0] - before[0], after[1] - before[1])
                progress = before_distance - after_distance
                self._record_outcome(
                    action,
                    moved,
                    progress,
                    action_seconds=action_seconds,
                )

                if progress >= self.config.stuck_minimum_progress_cells:
                    self._last_progress_at = monotonic()
                elif monotonic() - self._last_progress_at >= self.config.stuck_after_seconds:
                    self._backward_recovery()
                    recovery_used = True
                    self._last_progress_at = monotonic()
        except Exception:
            # Normal farming bursts deliberately retain their held movement
            # state. Only exceptional exits release input here.
            self.stop()
            raise

        if self.cancellation.cancelled or not self.bot.rl_enabled:
            self.stop()

        if not isfinite(initial_distance):
            pose = self._wait_for_navigation_pose()
            layout_position = self.map_context.native_to_layout_cells(pose.x, pose.z)
            position = self.map_context.nearest_safe_cell(layout_position)
            if position is not None:
                initial_distance = float(
                    self.core.distance_field[position[1], position[0]]
                )
                final_distance = initial_distance
        arrived = bool(
            isfinite(final_distance)
            and final_distance <= self.config.goal_tolerance_cells
        )
        return NavigationBurstResult(
            goal_cell=self.core.goal,
            actions=actions,
            elapsed_seconds=max(0.0, monotonic() - started),
            initial_distance_cells=float(initial_distance),
            final_distance_cells=float(final_distance),
            arrived=arrived,
            recovery_used=recovery_used,
            last_action=last_action,
        )

    def _sync_core(
        self,
        position: tuple[int, int],
        continuous: tuple[float, float],
        heading: float,
    ) -> None:
        self.core.position = position
        self.core.continuous_position = (
            float(continuous[0]),
            float(continuous[1]),
        )
        # Project/minimap convention is 0°=+Z/north and 90°=+X/east.
        # NavigatorCore uses 0°=+X/east and 90°=array-up/north.
        navigator_heading = (90.0 - float(heading)) % 360.0
        self.core.heading_deg = navigator_heading
        self.core.observed_heading_deg = navigator_heading
        self.core.heading_stale = False

    def _record_outcome(
        self,
        action: NavigatorAction,
        moved_cells: float,
        progress_cells: float,
        *,
        action_seconds: float,
    ) -> None:
        # Keep the recurrent state features of the live controller aligned with
        # the simulator used to train the frozen navigator. Position and heading
        # come from native memory; timing, steering history and motion outcome
        # are maintained here.
        self.core.step_count += 1
        self.core._record_steering(action)
        self.core.elapsed_seconds += max(0.0, float(action_seconds))
        self.core.forward_command_seconds += max(0.0, float(action_seconds))
        self.core.travel_distance += max(0.0, float(moved_cells))
        self.core.last_route_inefficiency = max(
            0.0,
            float(moved_cells) - max(0.0, float(progress_cells)),
        )
        self.core.last_action = action
        self.core.last_progress = float(progress_cells)
        if progress_cells > 0.05:
            self.core.steps_since_progress = 0
        else:
            self.core.steps_since_progress += 1
        if moved_cells <= 0.08:
            self.core.last_outcome = NavigatorOutcome.BLOCKED
        elif action is NavigatorAction.RUN_FORWARD_LEFT:
            self.core.last_outcome = NavigatorOutcome.STEERED_LEFT
        elif action is NavigatorAction.RUN_FORWARD_RIGHT:
            self.core.last_outcome = NavigatorOutcome.STEERED_RIGHT
        elif action is NavigatorAction.FORWARD_JUMP:
            self.core.last_outcome = NavigatorOutcome.JUMPED
            self.core.jump_count += 1
            self.core.jump_cooldown_remaining = self.core.config.jump_cooldown_steps
        else:
            self.core.last_outcome = NavigatorOutcome.MOVED
        if action is not NavigatorAction.FORWARD_JUMP and self.core.jump_cooldown_remaining:
            self.core.jump_cooldown_remaining -= 1

    def _action_seconds(self, action: NavigatorAction, distance: float) -> float:
        if action is NavigatorAction.FORWARD_JUMP:
            return float(self.core.config.jump_seconds)
        if distance <= self.core.config.final_approach_distance_cells:
            return float(self.core.config.final_approach_seconds)
        return float(self.core.config.forward_seconds)

    def _backward_recovery(self) -> None:
        self.stop()
        keyboard = self.bot.keyboard
        if keyboard is None:
            return
        # AZERTY and QWERTY both commonly use S for backward in FlyFF.
        keyboard.press_key(0x53, press_time=self.config.backward_recovery_seconds)

    def _require_foreground(self) -> None:
        keyboard = self.bot.keyboard
        if keyboard is None:
            raise RuntimeError("Keyboard input is unavailable")
        if not keyboard.is_target_foreground():
            self.stop()
            raise RuntimeError(
                "FlyFF must be focused for live movement. Return focus to the "
                "game and start the dry run again."
            )

    def _wait_for_navigation_pose(self):
        deadline = monotonic() + self.config.maximum_pose_wait_seconds
        while monotonic() < deadline and not self.cancellation.cancelled:
            pose = self.bot.get_navigation_pose(max_heading_age_seconds=1.0)
            if pose is not None and pose.heading_degrees is not None:
                return pose
            self._wait(self.config.pose_retry_seconds)
        raise RuntimeError("Timed out waiting for a native position and minimap heading")

    def _wait(self, seconds: float) -> None:
        deadline = monotonic() + max(0.0, float(seconds))
        while monotonic() < deadline:
            if self.cancellation.cancelled or not self.bot.rl_enabled:
                return
            sleep(min(0.02, max(0.0, deadline - monotonic())))
