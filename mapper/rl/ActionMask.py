from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .PolicyTypes import MapperAction, MotionOutcome, ObservationQuality


MAXIMUM_WAIT_STREAK = 2
DEFAULT_FRONTIER_ESCAPE_STEPS = 30


@dataclass(frozen=True)
class ActionMaskContext:
    """Minimal state needed to decide which high-level actions are safe/useful."""

    quality: ObservationQuality
    last_outcome: MotionOutcome
    last_action: MapperAction
    pose_known: bool
    heading_available: bool
    camera_obscured: bool
    backtrack_available: bool
    turn_streak: int = 0
    wait_streak: int = 0
    maximum_wait_streak: int = MAXIMUM_WAIT_STREAK
    steps_since_discovery: int = 0
    frontier_relative_direction: int | None = None
    frontier_distance: int = 0
    frontier_escape_steps: int = DEFAULT_FRONTIER_ESCAPE_STEPS

    @property
    def frontier_escape_active(self) -> bool:
        return bool(
            self.frontier_relative_direction is not None
            and self.frontier_distance > 0
            and self.steps_since_discovery >= max(1, self.frontier_escape_steps)
        )


def build_action_mask(
    context: ActionMaskContext,
) -> NDArray[np.bool_]:
    """Return a boolean mask where True means the action is currently valid.

    Recovery remains safety-first. During ordinary navigation the policy keeps
    the v1.8 freedom to move or turn. Once it has failed to discover anything
    for a bounded interval, v1.9 enters *frontier escape*: only actions that
    rotate toward or advance along the shortest known frontier route remain
    valid. This prevents a learned local loop from consuming the whole episode.
    """

    mask = np.zeros(len(MapperAction), dtype=np.bool_)

    camera_recovery_needed = (
        context.camera_obscured
        or context.quality is ObservationQuality.CAMERA_OBSCURED
    )
    active_recovery_needed = (
        not context.pose_known
        or not context.heading_available
        or context.quality
        in (
            ObservationQuality.HEADING_UNAVAILABLE,
            ObservationQuality.UNRESOLVED,
        )
    )
    recovery_needed = camera_recovery_needed or active_recovery_needed

    if recovery_needed:
        mask[MapperAction.REACQUIRE_HEADING] = True

        passive_wait_allowed = (
            camera_recovery_needed
            and context.heading_available
            and context.pose_known
            and context.wait_streak < max(1, context.maximum_wait_streak)
        )
        if passive_wait_allowed:
            mask[MapperAction.WAIT] = True

        # One position-preserving turn may clear Flyff camera clipping. Repeated
        # free rotation remains forbidden until a fresh observation arrives.
        if (
            camera_recovery_needed
            and context.turn_streak == 0
            and context.wait_streak == 0
            and context.pose_known
        ):
            mask[MapperAction.TURN_LEFT] = True
            mask[MapperAction.TURN_RIGHT] = True
        return _ensure_nonempty(mask)

    contact = context.quality is ObservationQuality.CONTACT or context.last_outcome in (
        MotionOutcome.BLOCKED,
        MotionOutcome.CONTACT_SLIDE,
    )

    if context.frontier_escape_active:
        return _frontier_escape_mask(context, contact=contact)

    if not contact:
        mask[MapperAction.FORWARD] = True
    if context.backtrack_available:
        mask[MapperAction.BACKTRACK] = True

    # Two same-direction turns are enough for a U-turn. Beyond that the agent
    # must commit to movement/backtracking instead of spinning indefinitely.
    if context.turn_streak < 2:
        mask[MapperAction.TURN_LEFT] = True
        mask[MapperAction.TURN_RIGHT] = True

        # Do not immediately undo a turn when no movement/observation happened
        # between the two decisions. Two turns in the same direction remain valid.
        if context.turn_streak > 0:
            if context.last_action is MapperAction.TURN_LEFT:
                mask[MapperAction.TURN_RIGHT] = False
            elif context.last_action is MapperAction.TURN_RIGHT:
                mask[MapperAction.TURN_LEFT] = False

    # At a confirmed contact there must always be an escape action even when
    # there is no backtrack path and the turn streak was already saturated.
    if contact and not mask.any():
        mask[MapperAction.TURN_LEFT] = True
        mask[MapperAction.TURN_RIGHT] = True

    return _ensure_nonempty(mask)


def _frontier_escape_mask(
    context: ActionMaskContext,
    *,
    contact: bool,
) -> NDArray[np.bool_]:
    """Constrain a stalled policy to the next shortest-path frontier action."""

    mask = np.zeros(len(MapperAction), dtype=np.bool_)
    relative = int(context.frontier_relative_direction or 0) % 4

    if contact:
        # The known route may just have been invalidated by a newly observed
        # wall. Keep the directional turn when one exists, but retain backtrack
        # as a guaranteed way out until the next context recomputation.
        if context.backtrack_available:
            mask[MapperAction.BACKTRACK] = True
        if relative == 1:
            mask[MapperAction.TURN_LEFT] = True
        elif relative == 3:
            mask[MapperAction.TURN_RIGHT] = True
        else:
            _enable_u_turn(mask, context)
        return _ensure_nonempty(mask)

    if relative == 0:
        mask[MapperAction.FORWARD] = True
    elif relative == 1:
        mask[MapperAction.TURN_LEFT] = True
    elif relative == 3:
        mask[MapperAction.TURN_RIGHT] = True
    else:
        _enable_u_turn(mask, context)

    return _ensure_nonempty(mask)


def _enable_u_turn(
    mask: NDArray[np.bool_],
    context: ActionMaskContext,
) -> None:
    """Choose a stable U-turn direction without permitting left/right ping-pong."""

    if context.last_action is MapperAction.TURN_LEFT and context.turn_streak > 0:
        mask[MapperAction.TURN_LEFT] = True
    elif context.last_action is MapperAction.TURN_RIGHT and context.turn_streak > 0:
        mask[MapperAction.TURN_RIGHT] = True
    else:
        mask[MapperAction.TURN_LEFT] = True
        mask[MapperAction.TURN_RIGHT] = True


def valid_action_names(mask: NDArray[np.bool_]) -> tuple[str, ...]:
    values = np.asarray(mask, dtype=np.bool_).reshape(-1)
    if values.size != len(MapperAction):
        raise ValueError("action mask has the wrong length")
    return tuple(action.name for action in MapperAction if values[int(action)])


def fallback_action(mask: NDArray[np.bool_]) -> MapperAction:
    """Choose a deterministic safe action for random env checks/old callers."""

    values = _ensure_nonempty(np.asarray(mask, dtype=np.bool_).reshape(-1))
    preference = (
        MapperAction.REACQUIRE_HEADING,
        MapperAction.WAIT,
        MapperAction.FORWARD,
        MapperAction.BACKTRACK,
        MapperAction.TURN_LEFT,
        MapperAction.TURN_RIGHT,
    )
    for action in preference:
        if values[int(action)]:
            return action
    raise RuntimeError("no valid mapper action is available")


def _ensure_nonempty(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    if mask.size != len(MapperAction):
        raise ValueError("action mask has the wrong length")
    if not mask.any():
        mask[MapperAction.REACQUIRE_HEADING] = True
    return mask
