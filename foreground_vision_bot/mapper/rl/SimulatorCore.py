from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mapper.OccupancyGrid import BLOCKED, FREE, UNKNOWN

from .Observation import ObservationEncoder, PolicyContext
from .PolicyTypes import MapperAction, MotionOutcome, ObservationQuality
from .ProceduralDungeon import DungeonLayout, ProceduralDungeonGenerator


@dataclass(frozen=True)
class MapperSimulatorConfig:
    max_steps: int = 900
    base_camera_obstruction_probability: float = 0.018
    contact_camera_obstruction_probability: float = 0.16
    maximum_camera_obstruction_steps: int = 4
    heading_dropout_probability: float = 0.025
    turn_heading_dropout_probability: float = 0.08
    wall_slide_probability: float = 0.30
    lateral_slide_cell_probability: float = 0.12
    completion_coverage: float = 0.97

    new_free_reward: float = 1.0
    new_wall_reward: float = 0.18
    completion_reward: float = 8.0
    step_penalty: float = 0.015
    turn_penalty: float = 0.025
    wait_penalty: float = 0.025
    repeated_contact_penalty: float = 0.12
    invalid_observation_penalty: float = 0.45
    successful_recovery_reward: float = 0.30

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
        self.step_count = 0
        self.path: list[tuple[int, int]] = []
        self.discovered_free = 0
        self.discovered_walls = 0

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
        self.step_count = 0
        self.path = []
        self.discovered_free = 0
        self.discovered_walls = 0
        self._mark_free(self.position)
        return self.observation()

    def step(self, action: MapperAction | int) -> SimulatorStep:
        if self.layout is None:
            raise RuntimeError("reset must be called before step")
        action = MapperAction(int(action))
        self.step_count += 1
        self.last_action = action
        reward = -self.config.step_penalty
        newly_free = 0
        newly_blocked = 0
        recovered = False

        if action is MapperAction.FORWARD:
            outcome, newly_free, newly_blocked = self._forward()
            if outcome is MotionOutcome.INVALID_OBSERVATION:
                reward -= self.config.invalid_observation_penalty
            elif outcome in (MotionOutcome.BLOCKED, MotionOutcome.CONTACT_SLIDE):
                reward -= self.config.repeated_contact_penalty * max(
                    1,
                    self.contact_streak,
                )
        elif action is MapperAction.TURN_LEFT:
            self.heading_index = (self.heading_index + 1) % 4
            self.last_outcome = MotionOutcome.TURNED
            self.contact_streak = 0
            reward -= self.config.turn_penalty
            self._maybe_drop_heading(after_turn=True)
            self._turn_may_clear_camera()
        elif action is MapperAction.TURN_RIGHT:
            self.heading_index = (self.heading_index - 1) % 4
            self.last_outcome = MotionOutcome.TURNED
            self.contact_streak = 0
            reward -= self.config.turn_penalty
            self._maybe_drop_heading(after_turn=True)
            self._turn_may_clear_camera()
        elif action is MapperAction.WAIT:
            reward -= self.config.wait_penalty
            before = self.camera_obscured_remaining
            self.camera_obscured_remaining = max(
                0,
                self.camera_obscured_remaining - 1,
            )
            if before > 0 and self.camera_obscured_remaining == 0:
                recovered = True
            self.last_outcome = (
                MotionOutcome.RECOVERED if recovered else MotionOutcome.NONE
            )
        elif action is MapperAction.REACQUIRE_HEADING:
            reward -= self.config.wait_penalty
            recovered = self._reacquire()
            self.last_outcome = (
                MotionOutcome.RECOVERED
                if recovered
                else MotionOutcome.INVALID_OBSERVATION
            )
        elif action is MapperAction.BACKTRACK:
            outcome, newly_free = self._backtrack()
            self.last_outcome = outcome
            if outcome is MotionOutcome.INVALID_OBSERVATION:
                reward -= self.config.invalid_observation_penalty

        reward += newly_free * self.config.new_free_reward
        reward += newly_blocked * self.config.new_wall_reward
        if recovered:
            reward += self.config.successful_recovery_reward

        self._advance_camera_state()
        self._maybe_drop_heading(after_turn=False)
        self._update_quality()
        terminated = self.coverage >= self.config.completion_coverage
        if terminated:
            reward += self.config.completion_reward
        truncated = self.step_count >= self.config.max_steps
        observation = self.observation()
        return SimulatorStep(
            observation=observation,
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info={
                "coverage": self.coverage,
                "known_cells": int(np.count_nonzero(self.known != UNKNOWN)),
                "new_free": newly_free,
                "new_blocked": newly_blocked,
                "quality": self.quality.name,
                "motion_outcome": self.last_outcome.name,
                "camera_obscured": self.camera_obscured_remaining > 0,
                "heading_available": self.heading_available,
                "contact_streak": self.contact_streak,
            },
        )

    @property
    def coverage(self) -> float:
        if self.layout is None or self.layout.free_cell_count <= 0:
            return 0.0
        known_free = int(np.count_nonzero(self.known == FREE))
        return min(1.0, known_free / self.layout.free_cell_count)

    def observation(self) -> dict[str, NDArray[np.float32]]:
        return ObservationEncoder.encode(
            self.known,
            self.visits,
            centre_x=self.position[0],
            centre_y=self.position[1],
            context=PolicyContext(
                heading_index=self.heading_index,
                quality=self.quality,
                last_outcome=self.last_outcome,
                last_action=self.last_action,
                pose_known=self.pose_known,
                heading_available=self.heading_available,
                camera_obscured=self.camera_obscured_remaining > 0,
                contact_streak=self.contact_streak,
                frontier_count=self.frontier_count(),
                coverage=self.coverage,
                progress_fraction=self.step_count / self.config.max_steps,
                backtrack_available=bool(self.path),
            ),
        )

    def frontier_count(self) -> int:
        free = np.argwhere(self.known == FREE)
        count = 0
        for y, x in free:
            for dx, dy in self.DIRECTIONS:
                nx, ny = int(x + dx), int(y + dy)
                if 0 <= nx < self.known.shape[1] and 0 <= ny < self.known.shape[0]:
                    if self.known[ny, nx] == UNKNOWN:
                        count += 1
                        break
        return count

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

    def _advance_camera_state(self) -> None:
        if self.camera_obscured_remaining > 0:
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
        return new

    def _mark_blocked(self, position: tuple[int, int]) -> int:
        x, y = position
        if not (0 <= x < self.known.shape[1] and 0 <= y < self.known.shape[0]):
            return 0
        new = int(self.known[y, x] == UNKNOWN)
        if self.known[y, x] != FREE:
            self.known[y, x] = BLOCKED
        self.discovered_walls += new
        return new
