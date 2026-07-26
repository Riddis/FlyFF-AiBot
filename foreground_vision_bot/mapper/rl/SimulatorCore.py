from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mapper.OccupancyGrid import BLOCKED, FREE, UNKNOWN

from .ActionMask import fallback_action, valid_action_names
from .Observation import ObservationEncoder, PolicyContext
from .PolicyTypes import MapperAction, MotionOutcome, ObservationQuality
from .ProceduralDungeon import DungeonLayout, ProceduralDungeonGenerator


@dataclass(frozen=True)
class MapperSimulatorConfig:
    max_steps: int = 1200
    base_camera_obstruction_probability: float = 0.018
    contact_camera_obstruction_probability: float = 0.16
    maximum_camera_obstruction_steps: int = 4
    heading_dropout_probability: float = 0.025
    turn_heading_dropout_probability: float = 0.08
    wall_slide_probability: float = 0.30
    lateral_slide_cell_probability: float = 0.12
    completion_coverage: float = 0.60

    new_free_reward: float = 1.25
    new_wall_reward: float = 0.25
    completion_reward: float = 20.0
    successful_move_bonus: float = 0.05
    step_penalty: float = 0.010
    turn_penalty: float = 0.030
    consecutive_turn_penalty: float = 0.050
    wait_penalty: float = 0.080
    reacquire_penalty: float = 0.040
    consecutive_wait_penalty: float = 0.120
    unproductive_wait_penalty: float = 0.250
    failed_recovery_penalty: float = 0.080
    maximum_wait_streak: int = 2
    repeated_contact_penalty: float = 0.15
    maximum_contact_penalty_streak: int = 4
    invalid_observation_penalty: float = 0.35
    invalid_action_penalty: float = 0.25
    successful_recovery_reward: float = 0.35
    frontier_progress_reward: float = 0.08
    frontier_regression_penalty: float = 0.03
    stagnation_grace_steps: int = 12
    stagnation_penalty: float = 0.030
    stagnation_truncation_steps: int = 180
    stagnation_truncation_penalty: float = 8.0

    def __post_init__(self) -> None:
        if self.max_steps < 10:
            raise ValueError("max_steps must be at least 10")
        probabilities = (
            self.base_camera_obstruction_probability,
            self.contact_camera_obstruction_probability,
            self.heading_dropout_probability,
            self.turn_heading_dropout_probability,
            self.wall_slide_probability,
            self.lateral_slide_cell_probability,
            self.completion_coverage,
        )
        if not all(0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("simulator probabilities must be between zero and one")
        if self.maximum_camera_obstruction_steps < 1:
            raise ValueError("maximum obstruction duration must be positive")
        if self.maximum_contact_penalty_streak < 1:
            raise ValueError("maximum contact penalty streak must be positive")
        if self.maximum_wait_streak < 1:
            raise ValueError("maximum wait streak must be positive")
        if self.stagnation_grace_steps < 0:
            raise ValueError("stagnation grace steps cannot be negative")
        if self.stagnation_truncation_steps <= self.stagnation_grace_steps:
            raise ValueError(
                "stagnation truncation must be greater than the stagnation grace"
            )
        non_negative = (
            self.successful_move_bonus,
            self.step_penalty,
            self.turn_penalty,
            self.consecutive_turn_penalty,
            self.wait_penalty,
            self.reacquire_penalty,
            self.consecutive_wait_penalty,
            self.unproductive_wait_penalty,
            self.failed_recovery_penalty,
            self.repeated_contact_penalty,
            self.invalid_observation_penalty,
            self.invalid_action_penalty,
            self.frontier_progress_reward,
            self.frontier_regression_penalty,
            self.stagnation_penalty,
            self.stagnation_truncation_penalty,
        )
        if any(value < 0.0 for value in non_negative):
            raise ValueError("simulator penalties and bonuses cannot be negative")


@dataclass(frozen=True)
class SimulatorStep:
    observation: dict[str, NDArray[np.float32]]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]


class MapperSimulatorCore:
    """Game-free exploration simulator with Flyff-like observation failures."""

    DIRECTIONS: tuple[tuple[int, int], ...] = (
        (1, 0),
        (0, -1),
        (-1, 0),
        (0, 1),
    )

    def __init__(
        self,
        config: MapperSimulatorConfig | None = None,
        generator: ProceduralDungeonGenerator | None = None,
    ) -> None:
        self.config = config or MapperSimulatorConfig()
        self.generator = generator or ProceduralDungeonGenerator()
        self.rng = np.random.default_rng()
        self.layout: DungeonLayout | None = None
        self.known = np.empty((0, 0), dtype=np.uint8)
        self.visits = np.empty((0, 0), dtype=np.uint16)
        self.position = (0, 0)
        self.heading_index = 1
        self.camera_obscured_remaining = 0
        self.heading_available = True
        self.pose_known = True
        self.last_action = MapperAction.WAIT
        self.last_outcome = MotionOutcome.NONE
        self.quality = ObservationQuality.VALID
        self.contact_streak = 0
        self.turn_streak = 0
        self.wait_streak = 0
        self.maximum_wait_streak_seen = 0
        self.step_count = 0
        self.path: list[tuple[int, int]] = []
        self.discovered_free = 0
        self.discovered_walls = 0
        self.steps_since_discovery = 0
        self._cached_policy_context: PolicyContext | None = None
        self._cached_policy_key: tuple[object, ...] | None = None
        self._map_revision = 0

    def reset(self, seed: int | None = None) -> dict[str, NDArray[np.float32]]:
        self.rng = np.random.default_rng(seed)
        self.layout = self.generator.generate(self.rng)
        self.known = np.full(self.layout.traversable.shape, UNKNOWN, dtype=np.uint8)
        self.visits = np.zeros(self.layout.traversable.shape, dtype=np.uint16)
        self.position = self.layout.spawn
        self.heading_index = int(self.rng.integers(0, 4))
        self.camera_obscured_remaining = 0
        self.heading_available = True
        self.pose_known = True
        self.last_action = MapperAction.WAIT
        self.last_outcome = MotionOutcome.NONE
        self.quality = ObservationQuality.VALID
        self.contact_streak = 0
        self.turn_streak = 0
        self.wait_streak = 0
        self.maximum_wait_streak_seen = 0
        self.step_count = 0
        self.path = []
        self.discovered_free = 0
        self.discovered_walls = 0
        self.steps_since_discovery = 0
        self._cached_policy_context = None
        self._cached_policy_key = None
        self._map_revision = 0
        self._mark_free(self.position)
        return self.observation()

    def step(self, action: MapperAction | int) -> SimulatorStep:
        if self.layout is None:
            raise RuntimeError("reset must be called before step")

        requested_action = MapperAction(int(action))
        context_before = self.policy_context()
        mask_before = context_before.action_mask()
        action_was_masked = not bool(mask_before[int(requested_action)])
        executed_action = (
            fallback_action(mask_before) if action_was_masked else requested_action
        )
        frontier_distance_before = context_before.frontier_distance
        previous_turn_streak = self.turn_streak
        previous_wait_streak = self.wait_streak
        recovery_needed_before = self._recovery_needed()

        self.step_count += 1
        self.last_action = executed_action
        reward = -self.config.step_penalty
        if action_was_masked:
            reward -= self.config.invalid_action_penalty

        newly_free = 0
        newly_blocked = 0

        if executed_action is MapperAction.FORWARD:
            self.turn_streak = 0
            self.wait_streak = 0
            outcome, newly_free, newly_blocked = self._forward()
            if outcome is MotionOutcome.MOVED:
                reward += self.config.successful_move_bonus
            elif outcome is MotionOutcome.INVALID_OBSERVATION:
                reward -= self.config.invalid_observation_penalty
            elif outcome in (MotionOutcome.BLOCKED, MotionOutcome.CONTACT_SLIDE):
                reward -= self.config.repeated_contact_penalty * min(
                    self.config.maximum_contact_penalty_streak,
                    max(1, self.contact_streak),
                )
        elif executed_action is MapperAction.TURN_LEFT:
            self.turn_streak = previous_turn_streak + 1
            self.wait_streak = 0
            self.heading_index = (self.heading_index + 1) % 4
            self.last_outcome = MotionOutcome.TURNED
            self.contact_streak = 0
            reward -= self.config.turn_penalty
            if previous_turn_streak > 0:
                reward -= self.config.consecutive_turn_penalty * previous_turn_streak
            self._maybe_drop_heading(after_turn=True)
            self._turn_may_clear_camera()
        elif executed_action is MapperAction.TURN_RIGHT:
            self.turn_streak = previous_turn_streak + 1
            self.wait_streak = 0
            self.heading_index = (self.heading_index - 1) % 4
            self.last_outcome = MotionOutcome.TURNED
            self.contact_streak = 0
            reward -= self.config.turn_penalty
            if previous_turn_streak > 0:
                reward -= self.config.consecutive_turn_penalty * previous_turn_streak
            self._maybe_drop_heading(after_turn=True)
            self._turn_may_clear_camera()
        elif executed_action is MapperAction.WAIT:
            self.turn_streak = 0
            self.wait_streak = previous_wait_streak + 1
            self.maximum_wait_streak_seen = max(
                self.maximum_wait_streak_seen,
                self.wait_streak,
            )
            reward -= self.config.wait_penalty
            if previous_wait_streak > 0:
                reward -= self.config.consecutive_wait_penalty * previous_wait_streak
            self.camera_obscured_remaining = max(
                0,
                self.camera_obscured_remaining - 1,
            )
            self.last_outcome = MotionOutcome.NONE
        elif executed_action is MapperAction.REACQUIRE_HEADING:
            self.turn_streak = 0
            self.wait_streak = 0
            reward -= self.config.reacquire_penalty
            self._reacquire()
            self.last_outcome = MotionOutcome.INVALID_OBSERVATION
        elif executed_action is MapperAction.BACKTRACK:
            self.turn_streak = 0
            self.wait_streak = 0
            outcome, newly_free = self._backtrack()
            self.last_outcome = outcome
            if outcome is MotionOutcome.INVALID_OBSERVATION:
                reward -= self.config.invalid_observation_penalty

        reward += newly_free * self.config.new_free_reward
        reward += newly_blocked * self.config.new_wall_reward

        if newly_free > 0 or newly_blocked > 0:
            self.steps_since_discovery = 0
        else:
            self.steps_since_discovery += 1
            overdue = self.steps_since_discovery - self.config.stagnation_grace_steps
            if overdue > 0:
                reward -= self.config.stagnation_penalty * min(
                    5.0,
                    1.0 + overdue / 20.0,
                )

        self._advance_camera_state(executed_action=executed_action)
        self._maybe_drop_heading(after_turn=False)
        self._update_quality()
        recovery_succeeded = recovery_needed_before and (
            self.camera_obscured_remaining == 0
            and self.heading_available
            and self.pose_known
        )
        if recovery_succeeded:
            self.last_outcome = MotionOutcome.RECOVERED
            self._update_quality()
            reward += self.config.successful_recovery_reward
        elif executed_action is MapperAction.WAIT:
            reward -= self.config.unproductive_wait_penalty
        elif executed_action is MapperAction.REACQUIRE_HEADING:
            reward -= self.config.failed_recovery_penalty

        self._invalidate_policy_cache()
        context_after = self.policy_context()
        frontier_distance_after = context_after.frontier_distance
        if (
            frontier_distance_before > 0
            and frontier_distance_after > 0
            and newly_free == 0
            and newly_blocked == 0
        ):
            if frontier_distance_after < frontier_distance_before:
                reward += self.config.frontier_progress_reward
            elif frontier_distance_after > frontier_distance_before:
                reward -= self.config.frontier_regression_penalty

        completed = self.coverage >= self.config.completion_coverage
        if completed:
            reward += self.config.completion_reward
        stagnation_truncated = (
            not completed
            and self.steps_since_discovery >= self.config.stagnation_truncation_steps
        )
        if stagnation_truncated:
            reward -= self.config.stagnation_truncation_penalty
        terminated = completed
        truncated = (
            not completed
            and (
                self.step_count >= self.config.max_steps
                or stagnation_truncated
            )
        )
        observation = self.observation(context=context_after)
        mask_after = context_after.action_mask()
        return SimulatorStep(
            observation=observation,
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info={
                "coverage": self.coverage,
                "completed": completed,
                "known_cells": int(np.count_nonzero(self.known != UNKNOWN)),
                "new_free": newly_free,
                "new_blocked": newly_blocked,
                "quality": self.quality.name,
                "motion_outcome": self.last_outcome.name,
                "camera_obscured": self.camera_obscured_remaining > 0,
                "heading_available": self.heading_available,
                "contact_streak": self.contact_streak,
                "turn_streak": self.turn_streak,
                "wait_streak": self.wait_streak,
                "maximum_wait_streak_seen": self.maximum_wait_streak_seen,
                "steps_since_discovery": self.steps_since_discovery,
                "requested_action": requested_action.name,
                "executed_action": executed_action.name,
                "action_was_masked": action_was_masked,
                "valid_actions": valid_action_names(mask_after),
                "frontier_distance": frontier_distance_after,
                "recovery_needed_before": recovery_needed_before,
                "recovery_succeeded": recovery_succeeded,
                "stagnation_truncated": stagnation_truncated,
            },
        )

    @property
    def coverage(self) -> float:
        if self.layout is None or self.layout.free_cell_count <= 0:
            return 0.0
        known_free = int(np.count_nonzero(self.known == FREE))
        return min(1.0, known_free / self.layout.free_cell_count)

    def policy_context(self) -> PolicyContext:
        cache_key = self._policy_cache_key()
        if (
            self._cached_policy_context is not None
            and self._cached_policy_key == cache_key
        ):
            return self._cached_policy_context
        frontiers = self.frontier_cells()
        path = self._nearest_frontier_path(frontiers)
        direction, distance = self._frontier_guidance(frontiers, path)
        self._cached_policy_context = PolicyContext(
            heading_index=self.heading_index,
            quality=self.quality,
            last_outcome=self.last_outcome,
            last_action=self.last_action,
            pose_known=self.pose_known,
            heading_available=self.heading_available,
            camera_obscured=self.camera_obscured_remaining > 0,
            contact_streak=self.contact_streak,
            frontier_count=len(frontiers),
            coverage=self.coverage,
            progress_fraction=self.step_count / self.config.max_steps,
            backtrack_available=bool(self.path),
            turn_streak=self.turn_streak,
            wait_streak=self.wait_streak,
            maximum_wait_streak=self.config.maximum_wait_streak,
            steps_since_discovery=self.steps_since_discovery,
            frontier_relative_direction=direction,
            frontier_distance=distance,
        )
        self._cached_policy_key = cache_key
        return self._cached_policy_context

    def _policy_cache_key(self) -> tuple[object, ...]:
        return (
            self.position,
            self.heading_index,
            self.camera_obscured_remaining,
            self.heading_available,
            self.pose_known,
            self.last_action,
            self.last_outcome,
            self.quality,
            self.contact_streak,
            self.turn_streak,
            self.wait_streak,
            self.step_count,
            len(self.path),
            self.steps_since_discovery,
            self._map_revision,
        )

    def observation(
        self,
        *,
        context: PolicyContext | None = None,
    ) -> dict[str, NDArray[np.float32]]:
        return ObservationEncoder.encode(
            self.known,
            self.visits,
            centre_x=self.position[0],
            centre_y=self.position[1],
            context=context or self.policy_context(),
        )

    def action_masks(self) -> NDArray[np.bool_]:
        """Method name required by sb3-contrib MaskablePPO."""

        return self.policy_context().action_mask()

    def _invalidate_policy_cache(self) -> None:
        self._cached_policy_context = None
        self._cached_policy_key = None

    def frontier_cells(self) -> list[tuple[int, int]]:
        frontiers: list[tuple[int, int]] = []
        for y, x in np.argwhere(self.known == FREE):
            point = (int(x), int(y))
            if any(self._known_value(point[0] + dx, point[1] + dy) == UNKNOWN for dx, dy in self.DIRECTIONS):
                frontiers.append(point)
        return frontiers

    def frontier_count(self) -> int:
        return len(self.frontier_cells())

    def nearest_frontier_path(self) -> list[tuple[int, int]]:
        return self._nearest_frontier_path(self.frontier_cells())

    def _nearest_frontier_path(
        self,
        frontiers: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        start = self.position
        frontier_set = set(frontiers)
        if start in frontier_set:
            return []
        queue: deque[tuple[int, int]] = deque([start])
        parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        target: tuple[int, int] | None = None
        while queue:
            current = queue.popleft()
            if current in frontier_set:
                target = current
                break
            for dx, dy in self.DIRECTIONS:
                nxt = (current[0] + dx, current[1] + dy)
                if nxt in parents or self._known_value(*nxt) != FREE:
                    continue
                parents[nxt] = current
                queue.append(nxt)
        if target is None:
            return []
        path: list[tuple[int, int]] = []
        cursor = target
        while cursor != start:
            path.append(cursor)
            parent = parents[cursor]
            if parent is None:
                break
            cursor = parent
        path.reverse()
        return path

    def frontier_guidance(self) -> tuple[int | None, int]:
        frontiers = self.frontier_cells()
        return self._frontier_guidance(
            frontiers,
            self._nearest_frontier_path(frontiers),
        )

    def _frontier_guidance(
        self,
        frontiers: list[tuple[int, int]],
        path: list[tuple[int, int]],
    ) -> tuple[int | None, int]:
        target: tuple[int, int] | None = path[0] if path else None
        distance = len(path)
        if target is None and self.position in set(frontiers):
            for absolute_direction, (dx, dy) in enumerate(self.DIRECTIONS):
                candidate = (self.position[0] + dx, self.position[1] + dy)
                if self._known_value(*candidate) == UNKNOWN:
                    target = candidate
                    distance = 1
                    break
        if target is None:
            return None, 0
        dx = target[0] - self.position[0]
        dy = target[1] - self.position[1]
        try:
            absolute_direction = self.DIRECTIONS.index(
                (int(np.sign(dx)), int(np.sign(dy)))
            )
        except ValueError:
            return None, distance
        relative = (absolute_direction - self.heading_index) % 4
        return relative, distance

    def _forward(self) -> tuple[MotionOutcome, int, int]:
        if self.camera_obscured_remaining > 0 or not self.heading_available:
            self.pose_known = False
            self.last_outcome = MotionOutcome.INVALID_OBSERVATION
            return self.last_outcome, 0, 0

        dx, dy = self.DIRECTIONS[self.heading_index]
        target = (self.position[0] + dx, self.position[1] + dy)
        if self._is_free(target):
            self.path.append(self.position)
            self.position = target
            newly_free = self._mark_free(target)
            self.pose_known = True
            self.contact_streak = 0
            self.last_outcome = MotionOutcome.MOVED
            return self.last_outcome, newly_free, 0

        newly_blocked = self._mark_blocked(target)
        self.contact_streak += 1
        self.pose_known = True
        if self.rng.random() < self.config.wall_slide_probability:
            self.last_outcome = MotionOutcome.CONTACT_SLIDE
            if self.rng.random() < self.config.lateral_slide_cell_probability:
                self._attempt_lateral_slide()
        else:
            self.last_outcome = MotionOutcome.BLOCKED
        if self.rng.random() < self.config.contact_camera_obstruction_probability:
            self._start_camera_obstruction()
        return self.last_outcome, 0, newly_blocked

    def _backtrack(self) -> tuple[MotionOutcome, int]:
        if self.camera_obscured_remaining > 0 or not self.heading_available:
            self.pose_known = False
            return MotionOutcome.INVALID_OBSERVATION, 0
        if not self.path:
            return MotionOutcome.BLOCKED, 0
        target = self.path.pop()
        self.position = target
        self.contact_streak = 0
        self.pose_known = True
        return MotionOutcome.MOVED, self._mark_free(target)

    def _attempt_lateral_slide(self) -> None:
        directions = (
            (self.heading_index + 1) % 4,
            (self.heading_index - 1) % 4,
        )
        order = self.rng.permutation(2)
        for index in order:
            dx, dy = self.DIRECTIONS[directions[int(index)]]
            target = (self.position[0] + dx, self.position[1] + dy)
            if self._is_free(target):
                self.path.append(self.position)
                self.position = target
                self._mark_free(target)
                return

    def _reacquire(self) -> bool:
        recovered = False
        if self.camera_obscured_remaining > 0:
            reduction = int(self.rng.integers(1, 4))
            self.camera_obscured_remaining = max(
                0,
                self.camera_obscured_remaining - reduction,
            )
            recovered |= self.camera_obscured_remaining == 0
        if not self.heading_available and self.rng.random() < 0.90:
            self.heading_available = True
            recovered = True
        if self.camera_obscured_remaining == 0:
            self.pose_known = True
        return recovered

    def _turn_may_clear_camera(self) -> None:
        if self.camera_obscured_remaining <= 0:
            return
        if self.rng.random() < 0.65:
            self.camera_obscured_remaining = max(
                0,
                self.camera_obscured_remaining - 1,
            )

    def _advance_camera_state(self, *, executed_action: MapperAction) -> None:
        if self.camera_obscured_remaining > 0:
            # WAIT and REACQUIRE already applied their explicit recovery effect.
            # Other actions still consume time and may let transient clipping clear.
            if executed_action not in (
                MapperAction.WAIT,
                MapperAction.REACQUIRE_HEADING,
            ):
                self.camera_obscured_remaining = max(
                    0,
                    self.camera_obscured_remaining - 1,
                )
            return
        if self.rng.random() < self.config.base_camera_obstruction_probability:
            self._start_camera_obstruction()

    def _start_camera_obstruction(self) -> None:
        self.camera_obscured_remaining = int(
            self.rng.integers(
                1,
                self.config.maximum_camera_obstruction_steps + 1,
            )
        )

    def _maybe_drop_heading(self, *, after_turn: bool) -> None:
        if not self.heading_available:
            return
        probability = (
            self.config.turn_heading_dropout_probability
            if after_turn
            else self.config.heading_dropout_probability
        )
        if self.rng.random() < probability:
            self.heading_available = False

    def _recovery_needed(self) -> bool:
        return (
            self.camera_obscured_remaining > 0
            or not self.heading_available
            or not self.pose_known
            or self.quality
            in (
                ObservationQuality.CAMERA_OBSCURED,
                ObservationQuality.HEADING_UNAVAILABLE,
                ObservationQuality.UNRESOLVED,
            )
        )

    def _update_quality(self) -> None:
        if self.camera_obscured_remaining > 0:
            self.quality = ObservationQuality.CAMERA_OBSCURED
        elif not self.heading_available:
            self.quality = ObservationQuality.HEADING_UNAVAILABLE
        elif self.last_outcome in (
            MotionOutcome.BLOCKED,
            MotionOutcome.CONTACT_SLIDE,
        ):
            self.quality = ObservationQuality.CONTACT
        elif self.last_outcome is MotionOutcome.INVALID_OBSERVATION:
            self.quality = ObservationQuality.UNRESOLVED
        else:
            self.quality = ObservationQuality.VALID

    def _known_value(self, x: int, y: int) -> int:
        if not (0 <= x < self.known.shape[1] and 0 <= y < self.known.shape[0]):
            return BLOCKED
        return int(self.known[y, x])

    def _is_free(self, position: tuple[int, int]) -> bool:
        if self.layout is None:
            return False
        x, y = position
        return bool(
            0 <= x < self.layout.width
            and 0 <= y < self.layout.height
            and self.layout.traversable[y, x]
        )

    def _mark_free(self, position: tuple[int, int]) -> int:
        x, y = position
        new = int(self.known[y, x] != FREE)
        self.known[y, x] = FREE
        self.visits[y, x] = min(
            np.iinfo(np.uint16).max,
            int(self.visits[y, x]) + 1,
        )
        self.discovered_free += new
        self._map_revision += 1
        return new

    def _mark_blocked(self, position: tuple[int, int]) -> int:
        x, y = position
        if not (0 <= x < self.known.shape[1] and 0 <= y < self.known.shape[0]):
            return 0
        new = int(self.known[y, x] == UNKNOWN)
        if self.known[y, x] != FREE:
            self.known[y, x] = BLOCKED
        self.discovered_walls += new
        if new:
            self._map_revision += 1
        return new
