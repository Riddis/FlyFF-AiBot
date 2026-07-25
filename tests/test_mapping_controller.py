from __future__ import annotations

import pytest
from libs.HumanKeyboard import VKEY, KeyPressTiming
from mapper.MappingController import MappingController
from mapper.RotationModel import (
    DirectionIdleResponseCurve,
    DirectionRotationProfile,
    IdleResponseCurves,
    RotationTiming,
    StateAwareRotationModel,
    TurnDirection,
    TurnMemoryMode,
    TurnMemoryPolicy,
    TurnTransition,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeKeyboard:
    def __init__(self):
        self.release_batches = []
        self.release_all_calls = 0
        self.presses = []
        self.fail_press = False

    def release_keys(self, keys):
        self.release_batches.append(tuple(keys))

    def release_all(self):
        self.release_all_calls += 1

    def press_key(self, key, press_time):
        self.presses.append((key, press_time))
        if self.fail_press:
            raise RuntimeError("press failed")
        return KeyPressTiming(
            press_time, press_time, press_time + 0.002, press_time + 0.012
        )


def test_persistent_controller_keeps_reversal_after_long_idle() -> None:
    clock = FakeClock()
    keyboard = FakeKeyboard()
    controller = MappingController(
        keyboard,
        turn_memory_policy=TurnMemoryPolicy(TurnMemoryMode.PERSISTENT_OBSERVED, 6.0),
        clock=clock,
    )
    controller.turn_left(0.05)
    clock.advance(100.0)
    result = controller.turn_right(0.05)
    assert result.transition is TurnTransition.REVERSAL


def test_controller_can_install_validated_policy_without_resetting_history() -> None:
    clock = FakeClock()
    keyboard = FakeKeyboard()
    controller = MappingController(keyboard, neutral_after_seconds=7.0, clock=clock)
    controller.turn_left(0.05)
    clock.advance(2.0)
    policy = TurnMemoryPolicy(TurnMemoryMode.PERSISTENT_OBSERVED, 6.0)
    controller.set_turn_memory_policy(policy, reset_history=False)
    assert controller.turn_right(0.05).transition is TurnTransition.REVERSAL


def test_failed_press_releases_keys_and_does_not_record_state() -> None:
    keyboard = FakeKeyboard()
    keyboard.fail_press = True
    controller = MappingController(keyboard)
    with pytest.raises(RuntimeError, match="press failed"):
        controller.turn_right(0.1)
    assert controller.previous_turn_direction is None
    assert keyboard.release_batches == [(VKEY["z"], VKEY["q"], VKEY["d"])] * 2


def test_turn_degrees_compensates_without_waiting() -> None:
    clock = FakeClock()
    keyboard = FakeKeyboard()
    policy = TurnMemoryPolicy(TurnMemoryMode.PERSISTENT_OBSERVED, 2.0)
    controller = MappingController(keyboard, turn_memory_policy=policy, clock=clock)
    curve = DirectionIdleResponseCurve(
        TurnMemoryMode.PERSISTENT_OBSERVED,
        (0.0, 0.5, 1.0, 2.0),
        (0.0, 0.25, 0.60, 0.75),
        8,
        2.0,
        20.0,
        30.0,
        0.0,
    )
    profile = DirectionRotationProfile(
        RotationTiming(300, 0, 4, 0.2),
        RotationTiming(200, 0, 4, 0.2),
        RotationTiming(150, 0, 4, 0.2),
    )
    model = StateAwareRotationModel(
        profile, profile, policy, IdleResponseCurves(curve, curve)
    )
    controller.turn_left(0.05)
    clock.advance(0.75)
    before = clock.now
    result = controller.turn_degrees(
        TurnDirection.RIGHT, 30.0, model, maximum_seconds=1.0
    )
    assert result.transition is TurnTransition.REVERSAL
    assert clock.now == before
    assert keyboard.presses[-1][0] == VKEY["d"]
