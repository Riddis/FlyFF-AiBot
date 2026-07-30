from __future__ import annotations

import heapq
import math
from dataclasses import asdict, dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from .ProceduralDungeon import DungeonLayout


class NavigatorAction(IntEnum):
    """Forward-running actions for the goal-conditioned navigator.

    Every ordinary action keeps the forward key held.  Left/right actions are
    steering arcs, not in-place rotations.  Backpedal is intentionally absent;
    it belongs to the deterministic stuck-recovery layer used by the live bot.
    """

    RUN_FORWARD = 0
    RUN_FORWARD_LEFT = 1
    RUN_FORWARD_RIGHT = 2
    FORWARD_JUMP = 3


class NavigatorOutcome(IntEnum):
    NONE = 0
    MOVED = 1
    STEERED_LEFT = 2
    STEERED_RIGHT = 3
    BLOCKED = 4
    SLID = 5
    GOAL_REACHED = 6
    FORBIDDEN_CONTACT = 7
    SAFETY_BUFFER_CONTACT = 8
    JUMPED = 9
    JUMP_REJECTED = 10
    RECOVERY_STEER = 11


@dataclass(frozen=True)
class NavigatorSimulatorConfig:
    """Physics, safety and reward settings for destination navigation.

    The navigator is rewarded only for reducing shortest-path distance to a
    supplied goal and reaching it quickly and safely.  It receives no map
    coverage, frontier, wall-discovery, mob, pack-size, or kill reward.

    The movement action space is deliberately forward-only.  Steering changes
    heading continuously while moving along an arc, and all actions have the
    same normal control cadence.  This avoids the stop/rotate/go behaviour that
    a discrete in-place-turn policy can learn.

    Obstacles and teleport cells are inflated before task sampling, path-cost
    calculation, observations, and action masking.  A radius of two therefore
    gives the learned policy two cells of map-error tolerance.
    """

    max_steps: int = 900
    minimum_goal_distance_cells: float = 12.0
    maximum_goal_distance_cells: float = 120.0
    goal_tolerance_cells: float = 2.0

    # A fixed portion of episodes starts close to the destination.  These
    # tasks specifically teach smooth final approaches, recovery from an
    # overshoot, and goals that begin behind the current heading.
    near_goal_task_probability: float = 0.35
    near_goal_minimum_distance_cells: float = 4.0
    near_goal_maximum_distance_cells: float = 15.0
    near_goal_overshoot_heading_probability: float = 0.35
    near_goal_overshoot_heading_jitter_degrees: float = 20.0

    forward_seconds: float = 0.10
    jump_seconds: float = 0.10
    movement_cells_per_action: float = 1.0
    steering_degrees_per_action: float = 15.0
    motion_collision_substeps: int = 8

    # The policy keeps W held but receives twice-finer steering and movement
    # increments near the destination.  Speed and turn rate stay unchanged;
    # only the decision cadence increases.
    final_approach_distance_cells: float = 10.0
    final_approach_seconds: float = 0.05
    final_approach_movement_scale: float = 0.50
    final_approach_steering_scale: float = 0.50

    progress_reward_per_cell: float = 1.0
    arrival_reward: float = 12.0
    time_penalty_per_second: float = 0.10
    collision_penalty: float = 0.40
    slide_penalty: float = 0.15
    safety_buffer_contact_penalty: float = 0.50
    forbidden_penalty: float = 20.0
    timeout_penalty: float = 3.0
    stagnation_penalty: float = 0.03
    jump_rejected_penalty: float = 0.05
    jump_flair_reward: float = 0.001
    steering_reversal_penalty: float = 0.025
    recovery_steer_penalty: float = 0.08
    route_inefficiency_penalty_per_cell: float = 0.03
    near_goal_regression_penalty_per_cell: float = 0.20
    stagnation_grace_steps: int = 22
    stagnation_truncation_steps: int = 240
    steering_reversal_window_steps: int = 3
    prevent_direct_steering_reversal: bool = True

    wall_slide_probability: float = 0.20
    heading_staleness_probability_after_steer: float = 0.015
    heading_recovery_probability_after_action: float = 0.50

    obstacle_buffer_radius_cells: int = 2
    teleport_buffer_radius_cells: int = 2

    jump_enabled: bool = True
    jump_min_remaining_distance_cells: float = 24.0
    jump_lookahead_cells: int = 4
    jump_cooldown_steps: int = 40
    jump_route_tolerance_cells: float = 1e-4

    local_radius_cells: int = 10

    def __post_init__(self) -> None:
        if self.max_steps < 20:
            raise ValueError("max_steps must be at least 20")
        if self.minimum_goal_distance_cells < 1.0:
            raise ValueError("minimum goal distance must be positive")
        if self.maximum_goal_distance_cells < self.minimum_goal_distance_cells:
            raise ValueError("maximum goal distance must be >= minimum")
        if self.goal_tolerance_cells < 0.0:
            raise ValueError("goal tolerance cannot be negative")
        if not 0.0 <= self.near_goal_task_probability <= 1.0:
            raise ValueError("near-goal task probability must be between zero and one")
        if self.near_goal_minimum_distance_cells < self.goal_tolerance_cells:
            raise ValueError("near-goal minimum distance must exceed goal tolerance")
        if self.near_goal_maximum_distance_cells < self.near_goal_minimum_distance_cells:
            raise ValueError("near-goal maximum distance must be >= its minimum")
        if not 0.0 <= self.near_goal_overshoot_heading_probability <= 1.0:
            raise ValueError("overshoot heading probability must be between zero and one")
        if not 0.0 <= self.near_goal_overshoot_heading_jitter_degrees <= 90.0:
            raise ValueError("overshoot heading jitter must be between zero and 90 degrees")
        if min(self.forward_seconds, self.jump_seconds, self.final_approach_seconds) <= 0.0:
            raise ValueError("action durations must be positive")
        if self.jump_seconds > self.forward_seconds + 1e-9:
            raise ValueError(
                "jump_seconds may not exceed forward_seconds; flair must not slow travel"
            )
        if self.movement_cells_per_action <= 0.0:
            raise ValueError("movement_cells_per_action must be positive")
        if self.final_approach_distance_cells <= self.goal_tolerance_cells:
            raise ValueError("final-approach distance must exceed goal tolerance")
        if not 0.0 < self.final_approach_movement_scale <= 1.0:
            raise ValueError("final-approach movement scale must be in (0, 1]")
        if not 0.0 < self.final_approach_steering_scale <= 1.0:
            raise ValueError("final-approach steering scale must be in (0, 1]")
        expected_final_seconds = self.forward_seconds * self.final_approach_movement_scale
        if abs(self.final_approach_seconds - expected_final_seconds) > 1e-9:
            raise ValueError(
                "final_approach_seconds must preserve run speed: "
                "forward_seconds * final_approach_movement_scale"
            )
        if not 0.0 < self.steering_degrees_per_action <= 45.0:
            raise ValueError("steering_degrees_per_action must be in (0, 45]")
        if self.motion_collision_substeps < 2:
            raise ValueError("motion_collision_substeps must be at least two")
        if self.local_radius_cells < 3:
            raise ValueError("local radius must be at least three cells")
        if self.stagnation_grace_steps < 0:
            raise ValueError("stagnation grace cannot be negative")
        if self.stagnation_truncation_steps <= self.stagnation_grace_steps:
            raise ValueError("stagnation truncation must exceed its grace")
        if self.steering_reversal_window_steps < 1:
            raise ValueError("steering reversal window must be positive")
        if self.obstacle_buffer_radius_cells < 0:
            raise ValueError("obstacle buffer radius cannot be negative")
        if self.teleport_buffer_radius_cells < 0:
            raise ValueError("teleport buffer radius cannot be negative")
        if self.jump_min_remaining_distance_cells < 0.0:
            raise ValueError("jump minimum remaining distance cannot be negative")
        if self.jump_lookahead_cells < 1:
            raise ValueError("jump lookahead must be at least one cell")
        if self.jump_cooldown_steps < 1:
            raise ValueError("jump cooldown must be at least one step")
        if self.jump_route_tolerance_cells < 0.0:
            raise ValueError("jump route tolerance cannot be negative")
        if not 0.0 <= self.jump_flair_reward <= 0.01:
            raise ValueError("jump flair reward must remain tiny (between 0 and 0.01)")
        probabilities = (
            self.wall_slide_probability,
            self.heading_staleness_probability_after_steer,
            self.heading_recovery_probability_after_action,
        )
        if not all(0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("navigator probabilities must be between zero and one")
        non_negative = (
            self.progress_reward_per_cell,
            self.arrival_reward,
            self.time_penalty_per_second,
            self.collision_penalty,
            self.slide_penalty,
            self.safety_buffer_contact_penalty,
            self.forbidden_penalty,
            self.timeout_penalty,
            self.stagnation_penalty,
            self.jump_rejected_penalty,
            self.steering_reversal_penalty,
            self.recovery_steer_penalty,
            self.route_inefficiency_penalty_per_cell,
            self.near_goal_regression_penalty_per_cell,
        )
        if any(value < 0.0 for value in non_negative):
            raise ValueError("navigator rewards and penalties cannot be negative")


@dataclass(frozen=True)
class NavigatorStep:
    observation: dict[str, NDArray[np.float32]]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]


@dataclass(frozen=True)
class InflatedNavigationMasks:
    safe_traversable: NDArray[np.bool_]
    safety_buffer: NDArray[np.bool_]
    obstacle_buffer: NDArray[np.bool_]
    teleport_buffer: NDArray[np.bool_]


@dataclass(frozen=True)
class NavigatorMotionPreview:
    action: NavigatorAction
    outcome: NavigatorOutcome
    final_heading_deg: float
    target_continuous: tuple[float, float]
    target_cell: tuple[int, int]
    projected_geodesic_distance: float
    route_delta_cells: float
    movement_distance_cells: float
    action_seconds: float
    final_approach: bool
    reaches_goal_region: bool

    @property
    def is_safe(self) -> bool:
        return self.outcome in (
            NavigatorOutcome.MOVED,
            NavigatorOutcome.STEERED_LEFT,
            NavigatorOutcome.STEERED_RIGHT,
        )


# Index 0 points east; positive indices rotate left/counter-clockwise in world
# coordinates where grid Y grows downward.
DIRECTIONS_8: tuple[tuple[int, int], ...] = (
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
)
MOVE_COSTS: tuple[float, ...] = tuple(
    math.sqrt(2.0) if dx and dy else 1.0 for dx, dy in DIRECTIONS_8
)

# safe free, actual wall, actual teleport, outside, goal, inflated buffer
NAV_LOCAL_CHANNELS = 6
_STATE_PREFIX_SIZE = 11
_STATE_TAIL_SIZE = 6
NAV_STATE_SIZE = (
    _STATE_PREFIX_SIZE
    + len(NavigatorAction)
    + len(NavigatorOutcome)
    + _STATE_TAIL_SIZE
)


class NavigatorSimulatorCore:
    """Game-free goal-conditioned movement simulator for open FlyFF maps."""

    def __init__(self, *, config: NavigatorSimulatorConfig, generator) -> None:
        self.config = config
        self.generator = generator
        self.rng = np.random.default_rng()
        self.layout: DungeonLayout | None = None
        self.safe_traversable = np.empty((0, 0), dtype=np.bool_)
        self.safety_buffer = np.empty((0, 0), dtype=np.bool_)
        self.obstacle_buffer = np.empty((0, 0), dtype=np.bool_)
        self.teleport_buffer = np.empty((0, 0), dtype=np.bool_)
        self._position = (0, 0)
        self.continuous_position = (0.0, 0.0)
        self.goal = (0, 0)
        self.heading_deg = 0.0
        self.observed_heading_deg = 0.0
        self.heading_stale = False
        self.last_action = NavigatorAction.RUN_FORWARD
        self.last_outcome = NavigatorOutcome.NONE
        self.step_count = 0
        self.elapsed_seconds = 0.0
        self.forward_command_seconds = 0.0
        self.travel_distance = 0.0
        self.collision_count = 0
        self.slide_count = 0
        self.forbidden_contacts = 0
        self.safety_buffer_contacts = 0
        self.jump_count = 0
        self.jump_rejected_count = 0
        self.jump_cooldown_remaining = 0
        self.last_jump_eligible = False
        self.steps_since_progress = 0
        self.last_progress = 0.0
        self.last_route_inefficiency = 0.0
        self.initial_distance = 0.0
        self.near_goal_task = False
        self.goal_reached_latched = False
        self.distance_field = np.empty((0, 0), dtype=np.float32)
        self.path: list[tuple[int, int]] = []
        self.steering_reversal_count = 0
        self.last_nonzero_steer_sign = 0
        self.last_nonzero_steer_step = -10_000
        self.last_steer_sign = 0
        self.recovery_steer_sign = 0

    @property
    def position(self) -> tuple[int, int]:
        return self._position

    @position.setter
    def position(self, value: tuple[int, int]) -> None:
        x, y = int(value[0]), int(value[1])
        self._position = (x, y)
        self.continuous_position = (float(x), float(y))

    @property
    def heading_index(self) -> int:
        return int(round(self.heading_deg / 45.0)) % 8

    @heading_index.setter
    def heading_index(self, value: int) -> None:
        self.heading_deg = _wrap_degrees(int(value) * 45.0)

    @property
    def observed_heading_index(self) -> int:
        return int(round(self.observed_heading_deg / 45.0)) % 8

    @observed_heading_index.setter
    def observed_heading_index(self, value: int) -> None:
        self.observed_heading_deg = _wrap_degrees(int(value) * 45.0)

    @property
    def forward_duty_cycle(self) -> float:
        if self.elapsed_seconds <= 1e-9:
            return 1.0
        return float(np.clip(self.forward_command_seconds / self.elapsed_seconds, 0.0, 1.0))

    @property
    def steering_reversals_per_minute(self) -> float:
        if self.elapsed_seconds <= 1e-9:
            return 0.0
        return 60.0 * self.steering_reversal_count / self.elapsed_seconds

    @property
    def final_approach_active(self) -> bool:
        distance = self.current_geodesic_distance
        return bool(
            math.isfinite(distance)
            and distance <= self.config.final_approach_distance_cells
        )

    def _sample_initial_heading(self) -> float:
        if (
            self.near_goal_task
            and self.rng.random()
            < self.config.near_goal_overshoot_heading_probability
        ):
            px, py = self.position
            gx, gy = self.goal
            toward_goal = math.degrees(math.atan2(-(gy - py), gx - px)) % 360.0
            jitter = self.rng.uniform(
                -self.config.near_goal_overshoot_heading_jitter_degrees,
                self.config.near_goal_overshoot_heading_jitter_degrees,
            )
            return _wrap_degrees(toward_goal + 180.0 + float(jitter))
        return float(self.rng.uniform(0.0, 360.0))

    def reset(self, seed: int | None = None) -> dict[str, NDArray[np.float32]]:
        self.rng = np.random.default_rng(seed)
        last_error: Exception | None = None
        for _ in range(48):
            layout = self.generator.generate(self.rng)
            masks = inflate_navigation_masks(
                layout,
                obstacle_radius_cells=self.config.obstacle_buffer_radius_cells,
                teleport_radius_cells=self.config.teleport_buffer_radius_cells,
            )
            try:
                position, goal, distance_field, near_goal_task = self._sample_task(
                    layout,
                    masks.safe_traversable,
                )
            except ValueError as error:
                last_error = error
                continue
            self.layout = layout
            self.safe_traversable = masks.safe_traversable
            self.safety_buffer = masks.safety_buffer
            self.obstacle_buffer = masks.obstacle_buffer
            self.teleport_buffer = masks.teleport_buffer
            self.position = position
            self.goal = goal
            self.distance_field = distance_field
            self.near_goal_task = near_goal_task
            break
        else:
            raise ValueError(
                "could not sample a navigation task after applying safety buffers"
            ) from last_error

        self.heading_deg = self._sample_initial_heading()
        self.observed_heading_deg = self.heading_deg
        self.heading_stale = False
        self.last_action = NavigatorAction.RUN_FORWARD
        self.last_outcome = NavigatorOutcome.NONE
        self.step_count = 0
        self.elapsed_seconds = 0.0
        self.forward_command_seconds = 0.0
        self.travel_distance = 0.0
        self.collision_count = 0
        self.slide_count = 0
        self.forbidden_contacts = 0
        self.safety_buffer_contacts = 0
        self.jump_count = 0
        self.jump_rejected_count = 0
        self.jump_cooldown_remaining = 0
        self.last_jump_eligible = False
        self.steps_since_progress = 0
        self.last_progress = 0.0
        self.last_route_inefficiency = 0.0
        self.initial_distance = self.current_geodesic_distance
        self.goal_reached_latched = False
        self.path = [self.position]
        self.steering_reversal_count = 0
        self.last_nonzero_steer_sign = 0
        self.last_nonzero_steer_step = -10_000
        self.last_steer_sign = 0
        self.recovery_steer_sign = 0
        return self.observation()

    @property
    def current_geodesic_distance(self) -> float:
        x, y = self.position
        return float(self.distance_field[y, x])

    @property
    def goal_reached(self) -> bool:
        return bool(
            self.goal_reached_latched
            or self.current_geodesic_distance <= self.config.goal_tolerance_cells
        )

    def action_masks(self) -> NDArray[np.bool_]:
        masks = np.zeros(len(NavigatorAction), dtype=np.bool_)
        if self.layout is None:
            masks[int(NavigatorAction.RUN_FORWARD_LEFT)] = True
            masks[int(NavigatorAction.RUN_FORWARD_RIGHT)] = True
            return masks

        normal_actions = (
            NavigatorAction.RUN_FORWARD,
            NavigatorAction.RUN_FORWARD_LEFT,
            NavigatorAction.RUN_FORWARD_RIGHT,
        )
        for action in normal_actions:
            masks[int(action)] = self.preview_action(action).is_safe

        if self.config.prevent_direct_steering_reversal:
            if self.last_action is NavigatorAction.RUN_FORWARD_LEFT:
                opposite = NavigatorAction.RUN_FORWARD_RIGHT
                if masks[int(opposite)] and (
                    masks[int(NavigatorAction.RUN_FORWARD)]
                    or masks[int(NavigatorAction.RUN_FORWARD_LEFT)]
                ):
                    masks[int(opposite)] = False
            elif self.last_action is NavigatorAction.RUN_FORWARD_RIGHT:
                opposite = NavigatorAction.RUN_FORWARD_LEFT
                if masks[int(opposite)] and (
                    masks[int(NavigatorAction.RUN_FORWARD)]
                    or masks[int(NavigatorAction.RUN_FORWARD_RIGHT)]
                ):
                    masks[int(opposite)] = False

        masks[int(NavigatorAction.FORWARD_JUMP)] = self._jump_is_eligible()

        # A forward-only controller still needs a legal recovery choice in a
        # tight dead end.  Pick one steering direction and hold it until a safe
        # moving arc becomes available; exposing both directions would permit
        # left/right oscillation while translation is clipped by the buffer.
        if masks[:3].any():
            self.recovery_steer_sign = 0
        else:
            if self.recovery_steer_sign == 0:
                px, py = self.position
                gx, gy = self.goal
                desired = math.degrees(math.atan2(-(gy - py), gx - px)) % 360.0
                error = ((desired - self.heading_deg + 180.0) % 360.0) - 180.0
                self.recovery_steer_sign = 1 if error >= 0.0 else -1
            action = (
                NavigatorAction.RUN_FORWARD_LEFT
                if self.recovery_steer_sign > 0
                else NavigatorAction.RUN_FORWARD_RIGHT
            )
            masks[int(action)] = True
        return masks

    def preview_action(self, action: NavigatorAction | int) -> NavigatorMotionPreview:
        if self.layout is None:
            raise RuntimeError("reset must be called before preview_action")
        action = NavigatorAction(int(action))
        final_approach = self.final_approach_active
        movement_scale = (
            self.config.final_approach_movement_scale if final_approach else 1.0
        )
        steering_scale = (
            self.config.final_approach_steering_scale if final_approach else 1.0
        )
        steering_sign = self._steering_sign(action)
        steering_delta = (
            steering_sign
            * self.config.steering_degrees_per_action
            * steering_scale
        )
        final_heading = _wrap_degrees(self.heading_deg + steering_delta)
        motion_heading = _wrap_degrees(self.heading_deg + steering_delta * 0.5)
        vx, vy = _heading_vector(motion_heading)
        px, py = self.continuous_position
        distance = self.config.movement_cells_per_action * movement_scale
        target_continuous = (px + vx * distance, py + vy * distance)
        target_cell = _continuous_to_cell(target_continuous)
        outcome = self._classify_motion_path(target_continuous)
        if outcome is NavigatorOutcome.MOVED:
            if steering_sign > 0:
                outcome = NavigatorOutcome.STEERED_LEFT
            elif steering_sign < 0:
                outcome = NavigatorOutcome.STEERED_RIGHT
        projected = self.current_geodesic_distance
        tx, ty = target_cell
        if (
            0 <= tx < self.distance_field.shape[1]
            and 0 <= ty < self.distance_field.shape[0]
        ):
            projected = float(self.distance_field[ty, tx])
        route_delta = projected + distance - self.current_geodesic_distance
        return NavigatorMotionPreview(
            action=action,
            outcome=outcome,
            final_heading_deg=final_heading,
            target_continuous=target_continuous,
            target_cell=target_cell,
            projected_geodesic_distance=projected,
            route_delta_cells=float(route_delta),
            movement_distance_cells=float(distance),
            action_seconds=float(
                self.config.final_approach_seconds
                if final_approach
                else self.config.forward_seconds
            ),
            final_approach=bool(final_approach),
            reaches_goal_region=bool(
                outcome
                in (
                    NavigatorOutcome.MOVED,
                    NavigatorOutcome.STEERED_LEFT,
                    NavigatorOutcome.STEERED_RIGHT,
                )
                and self._segment_reaches_goal_region(
                    self.continuous_position,
                    target_continuous,
                )
            ),
        )

    def step(self, action: NavigatorAction | int) -> NavigatorStep:
        if self.layout is None:
            raise RuntimeError("reset must be called before step")
        action = NavigatorAction(int(action))
        self.step_count += 1
        jump_eligible_before = self._jump_is_eligible()
        self.last_jump_eligible = jump_eligible_before
        distance_before = self.current_geodesic_distance
        action_seconds = self._action_seconds(action)
        travel_before = self.travel_distance
        forbidden_contact = False
        jump_performed = False
        goal_crossed = False
        reversal = self._record_steering(action)

        if action is NavigatorAction.FORWARD_JUMP and not jump_eligible_before:
            outcome = NavigatorOutcome.JUMP_REJECTED
            self.jump_rejected_count += 1
        else:
            preview = self.preview_action(
                NavigatorAction.RUN_FORWARD
                if action is NavigatorAction.FORWARD_JUMP
                else action
            )
            all_unsafe_before = self._all_normal_previews_unsafe()
            self.heading_deg = preview.final_heading_deg
            outcome = preview.outcome
            recovery_steer = (
                action in (
                    NavigatorAction.RUN_FORWARD_LEFT,
                    NavigatorAction.RUN_FORWARD_RIGHT,
                )
                and not preview.is_safe
                and all_unsafe_before
            )
            if preview.is_safe:
                goal_crossed = preview.reaches_goal_region
                self._commit_motion(preview.target_continuous, preview.target_cell)
                if goal_crossed:
                    self.goal_reached_latched = True
                if action is NavigatorAction.FORWARD_JUMP:
                    outcome = NavigatorOutcome.JUMPED
                    jump_performed = True
                    self.jump_count += 1
            elif recovery_steer:
                # Exceptional dead-end recovery: forward remains commanded, but
                # the inflated safety layer clips translation while steering
                # changes heading.  This cannot become ordinary in-place turn
                # behaviour because it is available only when every arc is unsafe.
                outcome = NavigatorOutcome.RECOVERY_STEER
                self.observed_heading_deg = self.heading_deg
                self.heading_stale = False
            elif outcome is NavigatorOutcome.FORBIDDEN_CONTACT:
                forbidden_contact = True
            elif (
                outcome in (
                    NavigatorOutcome.BLOCKED,
                    NavigatorOutcome.SAFETY_BUFFER_CONTACT,
                )
                and self.rng.random() < self.config.wall_slide_probability
            ):
                slide = self._slide_candidate(self.heading_deg)
                if slide is not None:
                    slide_target = (float(slide[0]), float(slide[1]))
                    goal_crossed = self._segment_reaches_goal_region(
                        self.continuous_position,
                        slide_target,
                    )
                    self._commit_motion(slide_target, slide)
                    if goal_crossed:
                        self.goal_reached_latched = True
                    outcome = NavigatorOutcome.SLID

            self._update_observed_heading_after_motion(action, outcome)

        self.elapsed_seconds += action_seconds
        if outcome is not NavigatorOutcome.JUMP_REJECTED:
            # Every legal navigator action includes the forward key.
            self.forward_command_seconds += action_seconds
        self.last_action = action
        self.last_outcome = outcome
        self._advance_jump_cooldown(jump_performed)
        distance_after = self.current_geodesic_distance
        progress = 0.0
        if math.isfinite(distance_before) and math.isfinite(distance_after):
            progress = distance_before - distance_after
        self.last_progress = float(progress)
        actual_movement = max(0.0, self.travel_distance - travel_before)
        route_inefficiency = max(0.0, actual_movement - max(0.0, progress))
        self.last_route_inefficiency = float(route_inefficiency)

        reward = self.config.progress_reward_per_cell * progress
        reward -= self.config.time_penalty_per_second * action_seconds
        reward -= (
            self.config.route_inefficiency_penalty_per_cell
            * route_inefficiency
        )
        if (
            math.isfinite(distance_before)
            and distance_before <= self.config.final_approach_distance_cells
            and progress < 0.0
        ):
            reward -= (
                self.config.near_goal_regression_penalty_per_cell
                * (-progress)
            )
        if reversal:
            reward -= self.config.steering_reversal_penalty
        if outcome is NavigatorOutcome.RECOVERY_STEER:
            reward -= self.config.recovery_steer_penalty
        elif outcome is NavigatorOutcome.BLOCKED:
            reward -= self.config.collision_penalty
            self.collision_count += 1
        elif outcome is NavigatorOutcome.SLID:
            reward -= self.config.slide_penalty
            self.collision_count += 1
            self.slide_count += 1
        elif outcome is NavigatorOutcome.SAFETY_BUFFER_CONTACT:
            reward -= self.config.safety_buffer_contact_penalty
            self.safety_buffer_contacts += 1
        elif outcome is NavigatorOutcome.FORBIDDEN_CONTACT:
            reward -= self.config.forbidden_penalty
            self.collision_count += 1
            self.forbidden_contacts += 1
        elif outcome is NavigatorOutcome.JUMP_REJECTED:
            reward -= self.config.jump_rejected_penalty
        elif outcome is NavigatorOutcome.JUMPED:
            reward += self.config.jump_flair_reward

        if progress > 1e-6:
            self.steps_since_progress = 0
        else:
            self.steps_since_progress += 1
            if self.steps_since_progress > self.config.stagnation_grace_steps:
                reward -= self.config.stagnation_penalty

        terminated = False
        truncated = False
        if forbidden_contact:
            terminated = True
        elif self.goal_reached:
            terminated = True
            outcome = NavigatorOutcome.GOAL_REACHED
            self.last_outcome = outcome
            reward += self.config.arrival_reward
        elif self.step_count >= self.config.max_steps:
            truncated = True
            reward -= self.config.timeout_penalty
        elif self.steps_since_progress >= self.config.stagnation_truncation_steps:
            truncated = True
            reward -= self.config.timeout_penalty

        if (
            self.heading_stale
            and self.rng.random()
            < self.config.heading_recovery_probability_after_action
        ):
            self.observed_heading_deg = self.heading_deg
            self.heading_stale = False

        info = self.info()
        info.update(
            {
                "progress_cells": float(progress),
                "route_inefficiency_cells": float(route_inefficiency),
                "action_seconds": float(action_seconds),
                "final_approach_active": bool(
                    distance_before <= self.config.final_approach_distance_cells
                ),
                "goal_region_crossed": bool(goal_crossed),
                "forbidden_contact": bool(forbidden_contact),
                "jump_eligible_before_action": bool(jump_eligible_before),
                "jump_performed": bool(jump_performed),
                "steering_reversal": bool(reversal),
                "terminated": terminated,
                "truncated": truncated,
            }
        )
        return NavigatorStep(
            observation=self.observation(),
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def observation(self) -> dict[str, NDArray[np.float32]]:
        return {
            "local_map": self._local_map(),
            "state": self._state_vector(),
        }

    def info(self) -> dict[str, object]:
        source = self.layout.source_name if self.layout is not None else "unknown"
        optimal = max(1e-6, float(self.initial_distance))
        return {
            "layout_source": source,
            "position": list(self.position),
            "continuous_position": [float(v) for v in self.continuous_position],
            "goal": list(self.goal),
            "heading_deg": float(self.heading_deg),
            "observed_heading_deg": float(self.observed_heading_deg),
            "heading_index": int(self.heading_index),
            "observed_heading_index": int(self.observed_heading_index),
            "heading_stale": bool(self.heading_stale),
            "last_action": self.last_action.name,
            "motion_outcome": self.last_outcome.name,
            "step_count": int(self.step_count),
            "elapsed_seconds": float(self.elapsed_seconds),
            "forward_command_seconds": float(self.forward_command_seconds),
            "forward_duty_cycle": float(self.forward_duty_cycle),
            "steering_reversal_count": int(self.steering_reversal_count),
            "recovery_steer_sign": int(self.recovery_steer_sign),
            "steering_reversals_per_minute": float(
                self.steering_reversals_per_minute
            ),
            "geodesic_distance": float(self.current_geodesic_distance),
            "initial_geodesic_distance": float(self.initial_distance),
            "goal_tolerance_cells": float(self.config.goal_tolerance_cells),
            "near_goal_task": bool(self.near_goal_task),
            "final_approach_active": bool(self.final_approach_active),
            "last_route_inefficiency_cells": float(self.last_route_inefficiency),
            "route_detour_ratio": float(self.route_detour_ratio()),
            "travel_distance": float(self.travel_distance),
            "collision_count": int(self.collision_count),
            "slide_count": int(self.slide_count),
            "forbidden_contacts": int(self.forbidden_contacts),
            "safety_buffer_contacts": int(self.safety_buffer_contacts),
            "safety_buffer_cells": int(np.count_nonzero(self.safety_buffer)),
            "safe_traversable_cells": int(np.count_nonzero(self.safe_traversable)),
            "obstacle_buffer_radius_cells": int(
                self.config.obstacle_buffer_radius_cells
            ),
            "teleport_buffer_radius_cells": int(
                self.config.teleport_buffer_radius_cells
            ),
            "jump_count": int(self.jump_count),
            "jump_rejected_count": int(self.jump_rejected_count),
            "jump_cooldown_remaining": int(self.jump_cooldown_remaining),
            "jump_eligible": bool(self._jump_is_eligible()),
            "goal_reached": bool(self.goal_reached),
            "path_efficiency": float(optimal / max(optimal, self.travel_distance)),
            "reward_scheme": asdict(self.config),
        }

    def route_detour_ratio(self) -> float:
        px, py = self.position
        gx, gy = self.goal
        direct = math.hypot(gx - px, gy - py)
        if direct <= 1e-6:
            return 1.0
        geodesic = self.current_geodesic_distance
        if not math.isfinite(geodesic):
            return float("inf")
        return geodesic / direct

    def _sample_task(
        self,
        layout: DungeonLayout,
        safe_traversable: NDArray[np.bool_],
    ) -> tuple[
        tuple[int, int],
        tuple[int, int],
        NDArray[np.float32],
        bool,
    ]:
        del layout
        free_points = np.argwhere(safe_traversable)
        if len(free_points) < 2:
            raise ValueError("navigator layout requires at least two safe cells")

        near_goal_task = bool(
            self.rng.random() < self.config.near_goal_task_probability
        )
        if near_goal_task:
            minimum_distance = self.config.near_goal_minimum_distance_cells
            maximum_distance = self.config.near_goal_maximum_distance_cells
        else:
            minimum_distance = self.config.minimum_goal_distance_cells
            maximum_distance = self.config.maximum_goal_distance_cells

        for _ in range(40):
            sy, sx = free_points[int(self.rng.integers(0, len(free_points)))]
            start = (int(sx), int(sy))
            from_start = compute_distance_field(safe_traversable, start)
            candidates = np.argwhere(
                np.isfinite(from_start)
                & (from_start >= minimum_distance)
                & (from_start <= maximum_distance)
            )
            if len(candidates):
                gy, gx = candidates[int(self.rng.integers(0, len(candidates)))]
                goal = (int(gx), int(gy))
                return (
                    start,
                    goal,
                    compute_distance_field(safe_traversable, goal),
                    near_goal_task,
                )

        # If a small crop cannot provide the requested near-goal band, fall
        # back to the ordinary curriculum rather than rejecting the layout.
        if near_goal_task:
            for _ in range(40):
                sy, sx = free_points[int(self.rng.integers(0, len(free_points)))]
                start = (int(sx), int(sy))
                from_start = compute_distance_field(safe_traversable, start)
                candidates = np.argwhere(
                    np.isfinite(from_start)
                    & (from_start >= self.config.minimum_goal_distance_cells)
                    & (from_start <= self.config.maximum_goal_distance_cells)
                )
                if len(candidates):
                    gy, gx = candidates[int(self.rng.integers(0, len(candidates)))]
                    goal = (int(gx), int(gy))
                    return (
                        start,
                        goal,
                        compute_distance_field(safe_traversable, goal),
                        False,
                    )

        sy, sx = free_points[int(self.rng.integers(0, len(free_points)))]
        start = (int(sx), int(sy))
        from_start = compute_distance_field(safe_traversable, start)
        finite = np.argwhere(np.isfinite(from_start))
        if len(finite) < 2:
            raise ValueError("inflated safety map has no distinct reachable goal")
        distances = from_start[finite[:, 0], finite[:, 1]]
        index = int(np.argmax(distances))
        gy, gx = finite[index]
        goal = (int(gx), int(gy))
        if goal == start:
            raise ValueError("could not sample a distinct navigation goal")
        return (
            start,
            goal,
            compute_distance_field(safe_traversable, goal),
            False,
        )

    def _classify_motion_path(
        self,
        target_continuous: tuple[float, float],
    ) -> NavigatorOutcome:
        assert self.layout is not None
        px, py = self.continuous_position
        tx, ty = target_continuous
        steps = max(
            self.config.motion_collision_substeps,
            int(math.ceil(math.hypot(tx - px, ty - py) * 8.0)),
        )
        previous_cell = self.position
        for index in range(1, steps + 1):
            alpha = index / steps
            cell = _continuous_to_cell(
                (px + (tx - px) * alpha, py + (ty - py) * alpha)
            )
            cx, cy = cell
            if not (0 <= cx < self.layout.width and 0 <= cy < self.layout.height):
                return NavigatorOutcome.BLOCKED
            if self.layout.forbidden[cy, cx]:
                return NavigatorOutcome.FORBIDDEN_CONTACT
            if self.safety_buffer[cy, cx]:
                return NavigatorOutcome.SAFETY_BUFFER_CONTACT
            if not self.safe_traversable[cy, cx]:
                return NavigatorOutcome.BLOCKED
            if cell != previous_cell and not self._can_step(previous_cell, cell):
                return NavigatorOutcome.BLOCKED
            previous_cell = cell
        return NavigatorOutcome.MOVED

    def _segment_reaches_goal_region(
        self,
        start_continuous: tuple[float, float],
        target_continuous: tuple[float, float],
    ) -> bool:
        """Return True when a safe movement segment enters the goal region.

        The region uses the same geodesic distance field as navigation rather
        than raw Euclidean distance.  A target separated by a wall therefore
        cannot be accepted merely because it is visually close.  Sampling the
        full segment also catches a forward step that passes through the goal
        radius and ends just beyond it.
        """

        sx, sy = start_continuous
        tx, ty = target_continuous
        steps = max(
            self.config.motion_collision_substeps,
            int(math.ceil(math.hypot(tx - sx, ty - sy) * 12.0)),
        )
        for index in range(steps + 1):
            alpha = index / max(1, steps)
            cx, cy = _continuous_to_cell(
                (sx + (tx - sx) * alpha, sy + (ty - sy) * alpha)
            )
            if not (
                0 <= cx < self.distance_field.shape[1]
                and 0 <= cy < self.distance_field.shape[0]
                and self.safe_traversable[cy, cx]
            ):
                continue
            if float(self.distance_field[cy, cx]) <= self.config.goal_tolerance_cells:
                return True
        return False

    def _commit_motion(
        self,
        target_continuous: tuple[float, float],
        target_cell: tuple[int, int],
    ) -> None:
        px, py = self.continuous_position
        tx, ty = target_continuous
        self.continuous_position = (float(tx), float(ty))
        self._position = (int(target_cell[0]), int(target_cell[1]))
        self.travel_distance += math.hypot(tx - px, ty - py)
        if not self.path or self.path[-1] != self.position:
            self.path.append(self.position)

    def _slide_candidate(self, heading_deg: float) -> tuple[int, int] | None:
        candidates: list[tuple[int, int]] = []
        x, y = self.position
        for offset in (90.0, -90.0):
            vx, vy = _heading_vector(heading_deg + offset)
            target = (int(round(x + vx)), int(round(y + vy)))
            if target != self.position and self._can_step(self.position, target):
                candidates.append(target)
        if not candidates:
            return None
        return candidates[int(self.rng.integers(0, len(candidates)))]

    def _can_step(self, source: tuple[int, int], target: tuple[int, int]) -> bool:
        tx, ty = target
        if not (
            0 <= tx < self.safe_traversable.shape[1]
            and 0 <= ty < self.safe_traversable.shape[0]
            and self.safe_traversable[ty, tx]
        ):
            return False
        sx, sy = source
        dx, dy = tx - sx, ty - sy
        if abs(dx) > 1 or abs(dy) > 1:
            return False
        if dx and dy:
            if not self.safe_traversable[sy, sx + dx]:
                return False
            if not self.safe_traversable[sy + dy, sx]:
                return False
        return True

    def _all_normal_previews_unsafe(self) -> bool:
        return not any(
            self.preview_action(action).is_safe
            for action in (
                NavigatorAction.RUN_FORWARD,
                NavigatorAction.RUN_FORWARD_LEFT,
                NavigatorAction.RUN_FORWARD_RIGHT,
            )
        )

    def _jump_is_eligible(self) -> bool:
        if not self.config.jump_enabled or self.layout is None:
            return False
        if self.jump_cooldown_remaining > 0:
            return False
        if self.current_geodesic_distance < self.config.jump_min_remaining_distance_cells:
            return False
        if self.last_action not in (
            NavigatorAction.RUN_FORWARD,
            NavigatorAction.FORWARD_JUMP,
        ):
            return False
        if self.last_outcome not in (
            NavigatorOutcome.MOVED,
            NavigatorOutcome.JUMPED,
            NavigatorOutcome.GOAL_REACHED,
        ):
            return False

        preview = self.preview_action(NavigatorAction.RUN_FORWARD)
        if not preview.is_safe:
            return False
        if preview.route_delta_cells > self.config.jump_route_tolerance_cells:
            return False

        px, py = self.continuous_position
        vx, vy = _heading_vector(self.heading_deg)
        source = self.position
        for distance in range(1, self.config.jump_lookahead_cells + 1):
            candidate = _continuous_to_cell((px + vx * distance, py + vy * distance))
            if candidate == source:
                continue
            if not self._can_step(source, candidate):
                return False
            source = candidate
        return True

    def _advance_jump_cooldown(self, jump_performed: bool) -> None:
        if jump_performed:
            self.jump_cooldown_remaining = self.config.jump_cooldown_steps
        elif self.jump_cooldown_remaining > 0:
            self.jump_cooldown_remaining -= 1

    def _action_seconds(self, action: NavigatorAction) -> float:
        if action is NavigatorAction.FORWARD_JUMP:
            return self.config.jump_seconds
        if self.final_approach_active:
            return self.config.final_approach_seconds
        return self.config.forward_seconds

    def _steering_sign(self, action: NavigatorAction) -> int:
        if action is NavigatorAction.RUN_FORWARD_LEFT:
            return 1
        if action is NavigatorAction.RUN_FORWARD_RIGHT:
            return -1
        return 0

    def _record_steering(self, action: NavigatorAction) -> bool:
        sign = self._steering_sign(action)
        reversal = False
        if sign:
            if (
                self.last_nonzero_steer_sign == -sign
                and self.step_count - self.last_nonzero_steer_step
                <= self.config.steering_reversal_window_steps
            ):
                reversal = True
                self.steering_reversal_count += 1
            self.last_nonzero_steer_sign = sign
            self.last_nonzero_steer_step = self.step_count
        self.last_steer_sign = sign
        return reversal

    def _update_observed_heading_after_motion(
        self,
        action: NavigatorAction,
        outcome: NavigatorOutcome,
    ) -> None:
        moved = outcome in (
            NavigatorOutcome.MOVED,
            NavigatorOutcome.STEERED_LEFT,
            NavigatorOutcome.STEERED_RIGHT,
            NavigatorOutcome.SLID,
            NavigatorOutcome.JUMPED,
        )
        steering = self._steering_sign(action) != 0
        if (
            moved
            and steering
            and self.rng.random()
            < self.config.heading_staleness_probability_after_steer
        ):
            self.heading_stale = True
            return
        if moved:
            self.observed_heading_deg = self.heading_deg
            self.heading_stale = False

    def _local_map(self) -> NDArray[np.float32]:
        assert self.layout is not None
        radius = self.config.local_radius_cells
        size = radius * 2 + 1
        output = np.zeros((NAV_LOCAL_CHANNELS, size, size), dtype=np.float32)
        px, py = self.position
        forward_x, forward_y = _heading_vector(self.observed_heading_deg)
        right_x, right_y = -forward_y, forward_x

        for local_y in range(size):
            for local_x in range(size):
                local_right = local_x - radius
                local_forward = radius - local_y
                world_dx = int(round(local_right * right_x + local_forward * forward_x))
                world_dy = int(round(local_right * right_y + local_forward * forward_y))
                wx, wy = px + world_dx, py + world_dy
                if not (0 <= wx < self.layout.width and 0 <= wy < self.layout.height):
                    output[3, local_y, local_x] = 1.0
                    continue
                if self.layout.forbidden[wy, wx]:
                    output[2, local_y, local_x] = 1.0
                elif self.safety_buffer[wy, wx]:
                    output[5, local_y, local_x] = 1.0
                elif self.safe_traversable[wy, wx]:
                    output[0, local_y, local_x] = 1.0
                else:
                    output[1, local_y, local_x] = 1.0
                if (wx, wy) == self.goal:
                    output[4, local_y, local_x] = 1.0
        return output

    def _state_vector(self) -> NDArray[np.float32]:
        px, py = self.position
        gx, gy = self.goal
        dx = float(gx - px)
        dy = float(gy - py)
        max_distance = max(1.0, self.config.maximum_goal_distance_cells)
        forward_x, forward_y = _heading_vector(self.observed_heading_deg)
        right_x, right_y = -forward_y, forward_x
        goal_forward = dx * forward_x + dy * forward_y
        goal_right = dx * right_x + dy * right_y
        euclidean = math.hypot(dx, dy)
        bearing = math.atan2(-dy, dx)
        heading_angle = math.radians(self.observed_heading_deg)
        heading_error = _wrap_angle(bearing - heading_angle)

        state = np.zeros(NAV_STATE_SIZE, dtype=np.float32)
        state[0] = np.clip(goal_forward / max_distance, -1.0, 1.0)
        state[1] = np.clip(goal_right / max_distance, -1.0, 1.0)
        state[2] = np.clip(euclidean / max_distance, 0.0, 1.0)
        geodesic = self.current_geodesic_distance
        state[3] = 1.0 if not math.isfinite(geodesic) else np.clip(
            geodesic / max_distance,
            0.0,
            1.0,
        )
        state[4] = math.sin(heading_error)
        state[5] = math.cos(heading_error)
        state[6] = math.sin(heading_angle)
        state[7] = math.cos(heading_angle)
        state[8] = 0.0 if self.heading_stale else 1.0
        detour = self.route_detour_ratio()
        state[9] = 1.0 if not math.isfinite(detour) else np.clip(
            (detour - 1.0) / 3.0,
            0.0,
            1.0,
        )
        state[10] = self.forward_duty_cycle

        action_offset = _STATE_PREFIX_SIZE
        state[action_offset + int(self.last_action)] = 1.0
        outcome_offset = action_offset + len(NavigatorAction)
        state[outcome_offset + int(self.last_outcome)] = 1.0
        tail = outcome_offset + len(NavigatorOutcome)
        state[tail] = np.clip(
            self.steps_since_progress
            / max(1, self.config.stagnation_truncation_steps),
            0.0,
            1.0,
        )
        state[tail + 1] = np.clip(self.last_progress / 2.0, -1.0, 1.0)
        state[tail + 2] = 1.0 if self._jump_is_eligible() else 0.0
        state[tail + 3] = np.clip(
            self.jump_cooldown_remaining / max(1, self.config.jump_cooldown_steps),
            0.0,
            1.0,
        )
        state[tail + 4] = float(self.last_steer_sign)
        state[tail + 5] = np.clip(
            self.steering_reversals_per_minute / 120.0,
            0.0,
            1.0,
        )
        return state


def inflate_navigation_masks(
    layout: DungeonLayout,
    *,
    obstacle_radius_cells: int,
    teleport_radius_cells: int,
) -> InflatedNavigationMasks:
    """Create conservative navigation masks around walls and teleport cells.

    Chebyshev/square inflation is intentional: a configured radius of two means
    the navigator keeps at least two cells of clearance horizontally,
    vertically and diagonally. The outside of the array is treated as blocked.
    """

    traversable = np.asarray(layout.traversable, dtype=np.bool_)
    forbidden = np.asarray(layout.forbidden, dtype=np.bool_)
    actual_obstacles = ~traversable & ~forbidden
    obstacle_inflated = dilate_mask(
        actual_obstacles,
        int(obstacle_radius_cells),
        outside_value=True,
    )
    teleport_inflated = dilate_mask(
        forbidden,
        int(teleport_radius_cells),
        outside_value=False,
    )
    safety_buffer = (
        (obstacle_inflated | teleport_inflated)
        & traversable
        & ~forbidden
    )
    safe = traversable & ~obstacle_inflated & ~teleport_inflated & ~forbidden
    return InflatedNavigationMasks(
        safe_traversable=np.ascontiguousarray(safe),
        safety_buffer=np.ascontiguousarray(safety_buffer),
        obstacle_buffer=np.ascontiguousarray(
            obstacle_inflated & traversable & ~forbidden
        ),
        teleport_buffer=np.ascontiguousarray(
            teleport_inflated & traversable & ~forbidden
        ),
    )


def dilate_mask(
    mask: NDArray[np.bool_],
    radius_cells: int,
    *,
    outside_value: bool = False,
) -> NDArray[np.bool_]:
    source = np.asarray(mask, dtype=np.bool_)
    radius = int(radius_cells)
    if radius < 0:
        raise ValueError("dilation radius cannot be negative")
    if radius == 0:
        return source.copy()
    padded = np.pad(
        source,
        radius,
        mode="constant",
        constant_values=bool(outside_value),
    )
    result = np.zeros_like(source)
    height, width = source.shape
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            result |= padded[dy : dy + height, dx : dx + width]
    return result


def compute_distance_field(
    traversable: NDArray[np.bool_],
    goal: tuple[int, int],
) -> NDArray[np.float32]:
    """Eight-connected Dijkstra field with diagonal corner-cut prevention."""

    mask = np.asarray(traversable, dtype=np.bool_)
    gx, gy = goal
    if not (0 <= gx < mask.shape[1] and 0 <= gy < mask.shape[0] and mask[gy, gx]):
        raise ValueError("goal must be traversable")
    distances = np.full(mask.shape, np.inf, dtype=np.float64)
    distances[gy, gx] = 0.0
    queue: list[tuple[float, int, int]] = [(0.0, gx, gy)]
    while queue:
        distance, x, y = heapq.heappop(queue)
        if distance > float(distances[y, x]) + 1e-6:
            continue
        for index, (dx, dy) in enumerate(DIRECTIONS_8):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < mask.shape[1] and 0 <= ny < mask.shape[0]):
                continue
            if not mask[ny, nx]:
                continue
            if dx and dy and (not mask[y, nx] or not mask[ny, x]):
                continue
            candidate = distance + MOVE_COSTS[index]
            if candidate + 1e-6 >= float(distances[ny, nx]):
                continue
            distances[ny, nx] = candidate
            heapq.heappush(queue, (candidate, nx, ny))
    return distances.astype(np.float32)


def _continuous_to_cell(position: tuple[float, float]) -> tuple[int, int]:
    return int(round(position[0])), int(round(position[1]))


def _heading_vector(heading_deg: float) -> tuple[float, float]:
    angle = math.radians(_wrap_degrees(heading_deg))
    return math.cos(angle), -math.sin(angle)


def _wrap_degrees(value: float) -> float:
    return float(value % 360.0)


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
