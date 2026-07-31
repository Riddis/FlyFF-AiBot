from __future__ import annotations

from enum import IntEnum

import pytest
from farming.actions import (
    ACTION_COUNT,
    ACTION_NAMES,
    FarmingAction,
    coerce_farming_action,
)


def test_unified_policy_has_exactly_four_stable_action_values() -> None:
    assert [(action.name, action.value) for action in FarmingAction] == [
        ("RUN_FORWARD", 0),
        ("RUN_FORWARD_LEFT", 1),
        ("RUN_FORWARD_RIGHT", 2),
        ("CAST_EVA", 3),
    ]
    assert ACTION_NAMES == (
        "RUN_FORWARD",
        "RUN_FORWARD_LEFT",
        "RUN_FORWARD_RIGHT",
        "CAST_EVA",
    )
    assert ACTION_COUNT == 4


def test_unified_action_validation_rejects_legacy_indices_and_booleans() -> None:
    assert coerce_farming_action(2) is FarmingAction.RUN_FORWARD_RIGHT
    assert coerce_farming_action(FarmingAction.CAST_EVA) is FarmingAction.CAST_EVA

    with pytest.raises(ValueError, match="Invalid unified farming action"):
        coerce_farming_action(4)
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


def test_only_first_three_actions_are_persistent_movement() -> None:
    assert all(action.is_movement for action in tuple(FarmingAction)[:3])
    assert not FarmingAction.CAST_EVA.is_movement
