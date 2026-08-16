from __future__ import annotations

from mapper.rl.ActionMask import ActionMaskContext, build_action_mask
from mapper.rl.PolicyTypes import MapperAction, MotionOutcome, ObservationQuality


def context(**overrides) -> ActionMaskContext:
    values = {
        "quality": ObservationQuality.VALID,
        "last_outcome": MotionOutcome.NONE,
        "last_action": MapperAction.WAIT,
        "pose_known": True,
        "heading_available": True,
        "camera_obscured": False,
        "backtrack_available": False,
        "turn_streak": 0,
        "wait_streak": 0,
        "maximum_wait_streak": 2,
    }
    values.update(overrides)
    return ActionMaskContext(**values)


def enabled(mask, action: MapperAction) -> bool:
    return bool(mask[int(action)])


def test_normal_navigation_masks_recovery_actions() -> None:
    mask = build_action_mask(context())

    assert enabled(mask, MapperAction.FORWARD)
    assert enabled(mask, MapperAction.TURN_LEFT)
    assert enabled(mask, MapperAction.TURN_RIGHT)
    assert not enabled(mask, MapperAction.WAIT)
    assert not enabled(mask, MapperAction.REACQUIRE_HEADING)
    assert not enabled(mask, MapperAction.BACKTRACK)


def test_contact_masks_repeated_forward_and_keeps_escape_actions() -> None:
    mask = build_action_mask(
        context(
            quality=ObservationQuality.CONTACT,
            last_outcome=MotionOutcome.BLOCKED,
            backtrack_available=True,
        )
    )

    assert not enabled(mask, MapperAction.FORWARD)
    assert enabled(mask, MapperAction.TURN_LEFT)
    assert enabled(mask, MapperAction.TURN_RIGHT)
    assert enabled(mask, MapperAction.BACKTRACK)


def test_heading_loss_requires_active_reacquisition() -> None:
    mask = build_action_mask(
        context(
            quality=ObservationQuality.HEADING_UNAVAILABLE,
            heading_available=False,
        )
    )

    assert not enabled(mask, MapperAction.WAIT)
    assert enabled(mask, MapperAction.REACQUIRE_HEADING)
    assert not enabled(mask, MapperAction.FORWARD)
    assert not enabled(mask, MapperAction.TURN_LEFT)
    assert not enabled(mask, MapperAction.TURN_RIGHT)


def test_camera_obstruction_allows_bounded_passive_wait() -> None:
    first = build_action_mask(
        context(
            quality=ObservationQuality.CAMERA_OBSCURED,
            camera_obscured=True,
            wait_streak=0,
        )
    )
    exhausted = build_action_mask(
        context(
            quality=ObservationQuality.CAMERA_OBSCURED,
            camera_obscured=True,
            wait_streak=2,
        )
    )

    assert enabled(first, MapperAction.WAIT)
    assert enabled(first, MapperAction.REACQUIRE_HEADING)
    assert not enabled(exhausted, MapperAction.WAIT)
    assert enabled(exhausted, MapperAction.REACQUIRE_HEADING)


def test_unknown_pose_cannot_wait_even_when_camera_is_obscured() -> None:
    mask = build_action_mask(
        context(
            quality=ObservationQuality.CAMERA_OBSCURED,
            camera_obscured=True,
            pose_known=False,
        )
    )

    assert not enabled(mask, MapperAction.WAIT)
    assert enabled(mask, MapperAction.REACQUIRE_HEADING)


def test_immediate_turn_reversal_is_masked() -> None:
    mask = build_action_mask(
        context(
            last_action=MapperAction.TURN_LEFT,
            last_outcome=MotionOutcome.TURNED,
            turn_streak=1,
        )
    )

    assert enabled(mask, MapperAction.TURN_LEFT)
    assert not enabled(mask, MapperAction.TURN_RIGHT)


def test_two_turns_force_commitment() -> None:
    mask = build_action_mask(
        context(
            last_action=MapperAction.TURN_LEFT,
            last_outcome=MotionOutcome.TURNED,
            turn_streak=2,
            backtrack_available=True,
        )
    )

    assert enabled(mask, MapperAction.FORWARD)
    assert enabled(mask, MapperAction.BACKTRACK)
    assert not enabled(mask, MapperAction.TURN_LEFT)
    assert not enabled(mask, MapperAction.TURN_RIGHT)
