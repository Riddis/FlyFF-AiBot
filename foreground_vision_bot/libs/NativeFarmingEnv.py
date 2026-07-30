from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from time import monotonic, sleep
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mapper.rl.NavigatorCore import compute_distance_field
from mapper.rl.TravelCost import TravelCostField

from .CameraDiscoverySweep import CameraDiscoverySweep
from .LiveNavigatorController import LiveNavigatorController, NavigationBurstResult
from .NativeFarmingObservation import (
    NativeFarmingObservation,
    NativeFarmingObservationBuilder,
)
from .NativeMapContext import NativeMapContext


@dataclass(frozen=True, slots=True)
class NativeFarmingEnvConfig:
    observation_delay_seconds: float = 0.05
    navigation_burst_seconds: float = 0.60
    eva_cooldown_seconds: float = 2.0
    eva_result_timeout_seconds: float = 0.40
    eva_result_poll_seconds: float = 0.05
    kill_counter_acquire_timeout_seconds: float = 2.0
    kill_counter_retry_seconds: float = 0.05
    episode_seconds: float = 300.0
    base_kill_reward: float = 1.0
    time_penalty_per_second: float = 0.01
    invalid_action_penalty: float = 0.10
    invalid_eva_penalty: float = 0.10
    eva_miss_penalty: float = 0.05
    density_delta_reward_scale: float = 0.01
    maximum_density_reward: float = 0.20
    max_kill_delta: int = 80
    camera_sweep_on_first_reset: bool = True
    camera_sweep_when_empty: bool = True

    def __post_init__(self) -> None:
        positive = (
            self.navigation_burst_seconds,
            self.eva_cooldown_seconds,
            self.eva_result_timeout_seconds,
            self.eva_result_poll_seconds,
            self.kill_counter_acquire_timeout_seconds,
            self.kill_counter_retry_seconds,
            self.episode_seconds,
            self.max_kill_delta,
        )
        if any(float(value) <= 0.0 for value in positive):
            raise ValueError("Native farming timing/count settings must be positive")
        if self.observation_delay_seconds < 0.0:
            raise ValueError("observation_delay_seconds cannot be negative")
        if self.maximum_density_reward < 0.0:
            raise ValueError("maximum_density_reward cannot be negative")


class NativeFarmingEnv(gym.Env):
    """Hierarchical live farming environment.

    PPO chooses one raw monster destination or CAST_EVA. The frozen movement
    navigator owns every low-level movement key decision.
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(
        self,
        *,
        bot,
        map_context: NativeMapContext,
        observation_builder: NativeFarmingObservationBuilder,
        navigator: LiveNavigatorController,
        camera_sweep: CameraDiscoverySweep,
        config: NativeFarmingEnvConfig | None = None,
    ) -> None:
        super().__init__()
        self.bot = bot
        self.map_context = map_context
        self.observation_builder = observation_builder
        self.navigator = navigator
        self.camera_sweep = camera_sweep
        self.config = config or NativeFarmingEnvConfig()

        self.cast_action = self.observation_builder.config.max_targets
        self.action_space = spaces.Discrete(self.cast_action + 1)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.observation_builder.observation_size,),
            dtype=np.float32,
        )

        self._episode_started_at = 0.0
        self._last_cast_at: float | None = None
        self._previous_kills: int | None = None
        self._last_kill_counter_read_ok = False
        self._consecutive_kill_counter_misses = 0
        self._snapshot: NativeFarmingObservation | None = None
        self._episode_reward = 0.0
        self._episode_kills = 0
        self._episode_steps = 0
        self._camera_warmed = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        del options
        super().reset(seed=seed)
        self.navigator.stop()
        self._episode_started_at = monotonic()
        self._last_cast_at = None
        self._episode_reward = 0.0
        self._episode_kills = 0
        self._episode_steps = 0

        initial_actors = self.bot.get_native_monsters()
        if self.config.camera_sweep_on_first_reset and not self._camera_warmed:
            self.camera_sweep.run(
                force=True,
                active_actor_count=len(initial_actors),
            )
            self._camera_warmed = True
        self._previous_kills = self._acquire_initial_kill_count()
        self._snapshot = self._read_snapshot()
        return self._snapshot.vector, self._info(
            action_name="RESET",
            kill_delta=0,
            reward_components=self._empty_components(),
            navigation=None,
            invalid_action=False,
            invalid_eva=False,
            eva_success=False,
            eva_miss=False,
        )

    def step(self, action: int):
        if self._snapshot is None:
            raise RuntimeError("reset() must be called before step()")
        started = monotonic()
        action_index = int(action)
        if not 0 <= action_index <= self.cast_action:
            raise ValueError(f"Invalid farming action: {action_index}")

        before = self._snapshot
        before_density = before.player_eva_count
        components = self._empty_components()
        navigation: NavigationBurstResult | None = None
        invalid_action = False
        invalid_eva = False
        eva_success = False
        eva_miss = False
        action_name = "CAST_EVA" if action_index == self.cast_action else "TARGET"

        if action_index == self.cast_action:
            if self._eva_cooldown_fraction() < 0.999:
                invalid_eva = True
                components["invalid_eva"] = -self.config.invalid_eva_penalty
            else:
                self.navigator.cast_eva()
                self._last_cast_at = monotonic()
        else:
            if action_index >= len(before.targets):
                invalid_action = True
                components["invalid_action"] = -self.config.invalid_action_penalty
            else:
                target = before.targets[action_index]
                navigation = self.navigator.navigate_toward_cell(
                    target.goal_cell,
                    duration_seconds=self.config.navigation_burst_seconds,
                )
                action_name = f"TARGET_{action_index}"

        if action_index == self.cast_action and not invalid_eva:
            current_kills = self._read_eva_result()
        else:
            self._wait(self.config.observation_delay_seconds)
            current_kills = self.bot.read_kill_count()

        self._record_kill_counter_read(current_kills)
        kill_delta = self._calculate_kill_delta(current_kills)
        components["kill"] = float(kill_delta) * self.config.base_kill_reward
        if action_index == self.cast_action and not invalid_eva:
            eva_success = kill_delta > 0
            eva_miss = kill_delta == 0 and current_kills is not None
            if eva_miss:
                components["eva_miss"] = -self.config.eva_miss_penalty

        after = self._read_snapshot()
        density_delta = after.player_eva_count - before_density
        if action_index != self.cast_action:
            components["density"] = float(
                np.clip(
                    density_delta * self.config.density_delta_reward_scale,
                    -self.config.maximum_density_reward,
                    self.config.maximum_density_reward,
                )
            )

        if (
            self.config.camera_sweep_when_empty
            and after.visible_count == 0
            and self.camera_sweep.should_run(active_actor_count=0)
        ):
            # A discovery sweep intentionally turns in place, so it is one
            # of the few normal paths that must release persistent forward input.
            self.navigator.stop()
            self.camera_sweep.run(active_actor_count=0)
            after = self._read_snapshot()

        elapsed = max(0.0, monotonic() - started)
        components["time"] = -elapsed * self.config.time_penalty_per_second
        reward = float(sum(components.values()))

        self._snapshot = after
        self._episode_steps += 1
        self._episode_kills += int(kill_delta)
        self._episode_reward += reward
        truncated = monotonic() - self._episode_started_at >= self.config.episode_seconds
        info = self._info(
            action_name=action_name,
            kill_delta=kill_delta,
            reward_components=components,
            navigation=navigation,
            invalid_action=invalid_action,
            invalid_eva=invalid_eva,
            eva_success=eva_success,
            eva_miss=eva_miss,
        )
        if truncated:
            info["episode_summary"] = {
                "reward": float(self._episode_reward),
                "kills": int(self._episode_kills),
                "steps": int(self._episode_steps),
                "elapsed_seconds": max(
                    0.0,
                    monotonic() - self._episode_started_at,
                ),
            }
        return after.vector, reward, False, truncated, info

    def heuristic_action(self, *, minimum_cast_targets: int = 4) -> int:
        """Policy-free action used by the required no-learning dry run."""
        snapshot = self._snapshot
        if snapshot is None:
            raise RuntimeError("reset() must be called before heuristic_action()")
        if (
            self._eva_cooldown_fraction() >= 0.999
            and snapshot.player_eva_count >= int(minimum_cast_targets)
        ):
            return self.cast_action
        if not snapshot.targets:
            return self.cast_action
        best_index = max(
            range(len(snapshot.targets)),
            key=lambda index: (
                snapshot.targets[index].utility,
                -snapshot.targets[index].geodesic_cells,
            ),
        )
        return int(best_index)

    def close(self) -> None:
        self.navigator.stop()

    @property
    def kill_counter_status(self) -> str:
        value = self._previous_kills
        if self._last_kill_counter_read_ok and value is not None:
            return f"OK:{value}"
        if value is not None:
            return (
                f"STALE:{value}/misses={self._consecutive_kill_counter_misses}"
            )
        return f"MISSING:misses={self._consecutive_kill_counter_misses}"

    def _read_snapshot(self) -> NativeFarmingObservation:
        pose = self.bot.get_navigation_pose(max_heading_age_seconds=1.5)
        if pose is None:
            raise RuntimeError("Native player position is unavailable")
        player_layout = self.map_context.native_to_layout_cells(pose.x, pose.z)
        origin = self.map_context.nearest_safe_cell(player_layout, maximum_radius=6)
        if origin is None:
            raise RuntimeError("Player is not on or near the selected map")
        distance = compute_distance_field(
            self.map_context.safe_traversable,
            origin,
        )
        reachable = np.isfinite(distance)
        eta = np.full(distance.shape, np.inf, dtype=np.float32)
        eta[reachable] = distance[reachable] * 0.10
        travel = TravelCostField(
            distance_cells=np.ascontiguousarray(distance),
            eta_seconds=np.ascontiguousarray(eta),
            reachable=np.ascontiguousarray(reachable),
            safe_traversable=self.map_context.safe_traversable,
            origin=origin,
        )
        actors = self.bot.get_native_monsters(
            vision_radius_native=(
                self.observation_builder.config.vision_radius_cells
                * self.map_context.native_units_per_cell
            )
        )
        return self.observation_builder.build(
            player_pose=pose,
            actors=actors,
            travel_cost=travel,
            eva_cooldown_fraction=self._eva_cooldown_fraction(),
        )

    def _acquire_initial_kill_count(self) -> int | None:
        deadline = monotonic() + self.config.kill_counter_acquire_timeout_seconds
        attempts = 0
        while (
            monotonic() < deadline
            and not bool(
                getattr(
                    getattr(self.navigator, "cancellation", None),
                    "cancelled",
                    False,
                )
            )
            and self.bot.rl_enabled
        ):
            attempts += 1
            sample = self.bot.read_kill_count()
            if sample is not None:
                self._last_kill_counter_read_ok = True
                self._consecutive_kill_counter_misses = 0
                return max(0, int(sample))
            self._last_kill_counter_read_ok = False
            self._consecutive_kill_counter_misses = attempts
            self._wait(self.config.kill_counter_retry_seconds)
        return None

    def _record_kill_counter_read(self, current: int | None) -> None:
        self._last_kill_counter_read_ok = current is not None
        if current is None:
            self._consecutive_kill_counter_misses += 1
        else:
            self._consecutive_kill_counter_misses = 0

    def _read_eva_result(self) -> int | None:
        deadline = monotonic() + self.config.eva_result_timeout_seconds
        latest: int | None = None
        while monotonic() < deadline:
            self._wait(self.config.eva_result_poll_seconds)
            sample = self.bot.read_kill_count()
            if sample is None:
                continue
            latest = int(sample)
            if self._previous_kills is not None and latest != self._previous_kills:
                return latest
        return latest

    def _calculate_kill_delta(self, current: int | None) -> int:
        if current is None:
            return 0
        current = int(current)
        previous = self._previous_kills
        self._previous_kills = current
        if previous is None or current < previous:
            return 0
        return int(np.clip(current - previous, 0, self.config.max_kill_delta))

    def _eva_cooldown_fraction(self) -> float:
        if self._last_cast_at is None:
            return 1.0
        return float(
            np.clip(
                (monotonic() - self._last_cast_at)
                / self.config.eva_cooldown_seconds,
                0.0,
                1.0,
            )
        )

    def _info(
        self,
        *,
        action_name: str,
        kill_delta: int,
        reward_components: dict[str, float],
        navigation: NavigationBurstResult | None,
        invalid_action: bool,
        invalid_eva: bool,
        eva_success: bool,
        eva_miss: bool,
    ) -> dict[str, object]:
        snapshot = self._snapshot
        navigation_payload = None
        if navigation is not None:
            navigation_payload = {
                "goal_cell": list(navigation.goal_cell),
                "actions": navigation.actions,
                "elapsed_seconds": navigation.elapsed_seconds,
                "initial_distance_cells": navigation.initial_distance_cells,
                "final_distance_cells": navigation.final_distance_cells,
                "arrived": navigation.arrived,
                "recovery_used": navigation.recovery_used,
                "last_action": navigation.last_action,
            }
        return {
            "action_name": str(action_name),
            "kills": self._previous_kills,
            "kill_delta": int(kill_delta),
            "visible_mobs": 0 if snapshot is None else snapshot.visible_count,
            "nearby_mobs": 0 if snapshot is None else snapshot.player_eva_count,
            "best_target_pack": (
                0 if snapshot is None else snapshot.best_target_nearby_count
            ),
            "eva_cooldown_fraction": self._eva_cooldown_fraction(),
            "invalid_action": bool(invalid_action),
            "invalid_eva": bool(invalid_eva),
            "eva_success": bool(eva_success),
            "eva_miss": bool(eva_miss),
            "reward_components": dict(reward_components),
            "navigation": navigation_payload,
        }

    @staticmethod
    def _empty_components() -> dict[str, float]:
        return {
            "kill": 0.0,
            "density": 0.0,
            "invalid_action": 0.0,
            "invalid_eva": 0.0,
            "eva_miss": 0.0,
            "time": 0.0,
        }

    @staticmethod
    def _wait(seconds: float) -> None:
        sleep(max(0.0, float(seconds)))
