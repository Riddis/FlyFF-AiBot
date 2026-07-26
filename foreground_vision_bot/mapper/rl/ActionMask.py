from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .PolicyTypes import MapperAction, MotionOutcome, ObservationQuality


MAXIMUM_WAIT_STREAK = 2


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


def build_action_mask(
    context: ActionMaskContext,
) -> NDArray[np.bool_]:
    """Return a boolean mask where True means the action is currently valid.

    v1.8 separates passive camera waiting from active heading/pose recovery. WAIT
    is available only for a genuine camera obstruction and only for a bounded
    number of consecutive decisions. Heading loss, unknown pose and unresolved
    observations must use REACQUIRE_HEADING instead of waiting forever.
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
        # REACQUIRE_HEADING is the escape hatch for every invalid state and is
        # mandatory once the passive wait budget has been exhausted.
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
