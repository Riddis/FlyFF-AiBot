from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mapper.AdaptiveMotionModel import AdaptiveForwardOutcome
from mapper.AdaptiveMotionTracker import MotionEstimate
from mapper.OccupancyGrid import OccupancyGrid, UNKNOWN

from .Observation import ObservationEncoder, PolicyContext
from .PolicyTypes import MapperAction, MotionOutcome, ObservationQuality


@dataclass(frozen=True)
class LivePolicyInput:
    observation: dict[str, object]
    action_mask: NDArray[np.bool_]
    context: PolicyContext


@dataclass
class LivePolicyMemory:
    last_action: MapperAction = MapperAction.WAIT
    last_outcome: MotionOutcome = MotionOutcome.NONE
    quality: ObservationQuality = ObservationQuality.VALID
    contact_streak: int = 0
    turn_streak: int = 0
    wait_streak: int = 0
    steps_since_discovery: int = 0
    last_known_cells: int = 0
    step_count: int = 0

    def update_after_step(
        self,
        *,
        actual_action: str,
        forward_outcome: AdaptiveForwardOutcome | None,
        motion: MotionEstimate | None,
        pose_known: bool,
        heading_known: bool,
        known_cell_count: int | None = None,
    ) -> None:
        self.step_count += 1
        self.last_action = _action_or_wait(actual_action)
        if self.last_action in (MapperAction.TURN_LEFT, MapperAction.TURN_RIGHT):
            self.turn_streak += 1
        else:
            self.turn_streak = 0
        if self.last_action is MapperAction.WAIT:
            self.wait_streak += 1
        else:
            self.wait_streak = 0

        if forward_outcome is AdaptiveForwardOutcome.MOVED:
            self.last_outcome = MotionOutcome.MOVED
            self.contact_streak = 0
        elif forward_outcome is AdaptiveForwardOutcome.BLOCKED:
            self.last_outcome = MotionOutcome.BLOCKED
            self.contact_streak += 1
        elif forward_outcome is AdaptiveForwardOutcome.UNCERTAIN:
            self.last_outcome = MotionOutcome.INVALID_OBSERVATION
        elif actual_action.startswith("TURN"):
            self.last_outcome = MotionOutcome.TURNED
            self.contact_streak = 0
        else:
            self.last_outcome = MotionOutcome.NONE

        if known_cell_count is not None:
            known = max(0, int(known_cell_count))
            if known > self.last_known_cells:
                self.steps_since_discovery = 0
            else:
                self.steps_since_discovery += 1
            self.last_known_cells = max(self.last_known_cells, known)

        if not heading_known:
            self.quality = ObservationQuality.HEADING_UNAVAILABLE
        elif not pose_known:
            self.quality = _uncertain_quality(motion)
        elif forward_outcome is AdaptiveForwardOutcome.BLOCKED:
            self.quality = ObservationQuality.CONTACT
        else:
            self.quality = ObservationQuality.VALID


def build_live_policy_input(
    grid: OccupancyGrid,
    memory: LivePolicyMemory,
    *,
    max_steps_hint: int = 1200,
) -> LivePolicyInput:
    context = _build_context(grid, memory, max_steps_hint=max_steps_hint)
    observation = ObservationEncoder.encode(
        grid.cells,
        grid.visits,
        centre_x=grid.pose.x,
        centre_y=grid.pose.y,
        context=context,
        coordinates_are_world=True,
        world_origin=grid.origin,
    )
    return LivePolicyInput(
        observation=observation,
        action_mask=context.action_mask(),
        context=context,
    )


def build_live_observation(
    grid: OccupancyGrid,
    memory: LivePolicyMemory,
    *,
    max_steps_hint: int = 1200,
) -> dict[str, object]:
    return build_live_policy_input(
        grid,
        memory,
        max_steps_hint=max_steps_hint,
    ).observation


def _build_context(
    grid: OccupancyGrid,
    memory: LivePolicyMemory,
    *,
    max_steps_hint: int,
) -> PolicyContext:
    known = max(1, grid.known_cell_count())
    visited = int((grid.visits > 0).sum())
    coverage_proxy = min(1.0, visited / known)
    path = grid.nearest_frontier_path()
    relative_direction, frontier_distance = _frontier_guidance(grid, path)
    return PolicyContext(
        heading_index=grid.pose.heading_index,
        quality=memory.quality,
        last_outcome=memory.last_outcome,
        last_action=memory.last_action,
        pose_known=grid.metadata.position_known,
        heading_available=grid.metadata.heading_known,
        camera_obscured=(memory.quality is ObservationQuality.CAMERA_OBSCURED),
        contact_streak=memory.contact_streak,
        frontier_count=len(grid.frontier_cells()),
        coverage=coverage_proxy,
        progress_fraction=min(1.0, memory.step_count / max_steps_hint),
        backtrack_available=bool(path),
        turn_streak=memory.turn_streak,
        wait_streak=memory.wait_streak,
        steps_since_discovery=memory.steps_since_discovery,
        frontier_relative_direction=relative_direction,
        frontier_distance=frontier_distance,
    )


def _frontier_guidance(
    grid: OccupancyGrid,
    path: list[tuple[int, int]],
) -> tuple[int | None, int]:
    position = (grid.pose.x, grid.pose.y)
    target = path[0] if path else None
    distance = len(path)
    if target is None:
        for absolute_direction, (dx, dy) in enumerate(grid.DIRECTIONS):
            candidate = (position[0] + dx, position[1] + dy)
            if grid.value(*candidate) == UNKNOWN:
                target = candidate
                distance = 1
                break
    if target is None:
        return None, 0
    delta = (target[0] - position[0], target[1] - position[1])
    normalized = (int(np.sign(delta[0])), int(np.sign(delta[1])))
    try:
        absolute_direction = grid.DIRECTIONS.index(normalized)
    except ValueError:
        return None, distance
    return (absolute_direction - grid.pose.heading_index) % 4, distance


def _action_or_wait(name: str) -> MapperAction:
    try:
        return MapperAction[str(name).strip().upper()]
    except KeyError:
        return MapperAction.WAIT


def _uncertain_quality(motion: MotionEstimate | None) -> ObservationQuality:
    if motion is None:
        return ObservationQuality.UNRESOLVED
    flow = motion.directional_flow
    if motion.teleport_likely:
        return ObservationQuality.CAMERA_OBSCURED
    if flow.detected_points >= 40 and flow.valid_tracks < 5:
        return ObservationQuality.CAMERA_OBSCURED
    if motion.change_score >= 0.16 and flow.valid_tracks < 8:
        return ObservationQuality.CAMERA_OBSCURED
    return ObservationQuality.UNRESOLVED
