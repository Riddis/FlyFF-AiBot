from __future__ import annotations

from enum import IntEnum

import pytest
from farming.actions import (
    ACTION_COUNT,
    ACTION_NAMES,
    FarmingAction,
    coerce_farming_action,
)


def test_unified_policy_has_exactly_five_stable_action_values() -> None:
    assert [(action.name, action.value) for action in FarmingAction] == [
        ("RUN_FORWARD", 0),
        ("RUN_FORWARD_LEFT", 1),
        ("RUN_FORWARD_RIGHT", 2),
        ("CAST_EVA", 3),
        ("RUN_FORWARD_JUMP", 4),
    ]
    assert ACTION_NAMES == (
        "RUN_FORWARD",
        "RUN_FORWARD_LEFT",
        "RUN_FORWARD_RIGHT",
        "CAST_EVA",
        "RUN_FORWARD_JUMP",
    )
    assert ACTION_COUNT == 5


def test_unified_action_validation_rejects_unknown_indices_and_booleans() -> None:
    assert coerce_farming_action(2) is FarmingAction.RUN_FORWARD_RIGHT
    assert coerce_farming_action(FarmingAction.CAST_EVA) is FarmingAction.CAST_EVA
    assert coerce_farming_action(4) is FarmingAction.RUN_FORWARD_JUMP

    with pytest.raises(ValueError, match="Invalid unified farming action"):
        coerce_farming_action(5)
    with pytest.raises(ValueError, match="integer action index"):
        coerce_farming_action(True)
    with pytest.raises(ValueError, match="integer action index"):
        coerce_farming_action(0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="integer action index"):
        coerce_farming_action("0")  # type: ignore[arg-type]

    class ForeignAction(IntEnum):
        RUN_FORWARD = 0

    with pytest.raises(ValueError, match="integer action index"):
        coerce_farming_action(ForeignAction.RUN_FORWARD)


def test_every_action_except_eva_is_forward_movement() -> None:
    assert all(
        action.is_movement
        for action in FarmingAction
        if action is not FarmingAction.CAST_EVA
    )
    assert not FarmingAction.CAST_EVA.is_movement
