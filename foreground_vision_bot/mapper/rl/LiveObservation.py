from __future__ import annotations

from dataclasses import dataclass

from mapper.AdaptiveMotionModel import AdaptiveForwardOutcome
from mapper.AdaptiveMotionTracker import MotionEstimate
from mapper.OccupancyGrid import OccupancyGrid

from .Observation import ObservationEncoder, PolicyContext
from .PolicyTypes import MapperAction, MotionOutcome, ObservationQuality


@dataclass
class LivePolicyMemory:
    last_action: MapperAction = MapperAction.WAIT
    last_outcome: MotionOutcome = MotionOutcome.NONE
    quality: ObservationQuality = ObservationQuality.VALID
    contact_streak: int = 0
    step_count: int = 0

    def update_after_step(
        self,
        *,
        actual_action: str,
        forward_outcome: AdaptiveForwardOutcome | None,
        motion: MotionEstimate | None,
        pose_known: bool,
        heading_known: bool,
    ) -> None:
        self.step_count += 1
        self.last_action = _action_or_wait(actual_action)
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

        if not heading_known:
            self.quality = ObservationQuality.HEADING_UNAVAILABLE
        elif not pose_known:
            self.quality = _uncertain_quality(motion)
        elif forward_outcome is AdaptiveForwardOutcome.BLOCKED:
            self.quality = ObservationQuality.CONTACT
        else:
            self.quality = ObservationQuality.VALID


def build_live_observation(
    grid: OccupancyGrid,
    memory: LivePolicyMemory,
    *,
    max_steps_hint: int = 900,
) -> dict[str, object]:
    known = max(1, grid.known_cell_count())
    visited = int((grid.visits > 0).sum())
    coverage_proxy = min(1.0, visited / known)
    context = PolicyContext(
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
        backtrack_available=bool(grid.nearest_frontier_path()),
    )
    return ObservationEncoder.encode(
        grid.cells,
        grid.visits,
        centre_x=grid.pose.x,
        centre_y=grid.pose.y,
        context=context,
        coordinates_are_world=True,
        world_origin=grid.origin,
    )


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
