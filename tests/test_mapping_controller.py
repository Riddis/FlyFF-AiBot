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
    TurnTransition,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeKeyboard:
    def __init__(self) -> None:
        self.release_batches: list[tuple[int, ...]] = []
        self.release_all_calls = 0
        self.presses: list[tuple[int, float]] = []
        self.fail_press = False

    def release_keys(self, keys) -> None:
        self.release_batches.append(tuple(keys))

    def release_all(self) -> None:
        self.release_all_calls += 1

    def press_key(self, key: int, press_time: float) -> KeyPressTiming:
        self.presses.append((key, press_time))
        if self.fail_press:
            raise RuntimeError("press failed")
        return KeyPressTiming(
            requested_seconds=press_time,
            clamped_seconds=press_time,
            held_seconds=press_time + 0.002,
            elapsed_seconds=press_time + 0.012,
        )


def test_turn_pulses_report_clamped_actual_timing_and_transition_state() -> None:
    keyboard = FakeKeyboard()
    clock = FakeClock()
    controller = MappingController(
        keyboard,
        neutral_after_seconds=2.0,
        clock=clock,
    )

    first = controller.turn_left(0.001)
    clock.advance(0.4)
    repeated = controller.turn_left(0.10)
    clock.advance(0.4)
    reversed_turn = controller.turn_right(0.10)
    clock.advance(2.1)
    neutral = controller.turn_left(0.10)

    assert first.direction is TurnDirection.LEFT
    assert first.transition is TurnTransition.NEUTRAL
    assert first.requested_seconds == pytest.approx(0.001)
    assert first.clamped_seconds == pytest.approx(0.015)
    assert first.held_seconds == pytest.approx(0.017)
    assert repeated.transition is TurnTransition.SAME_DIRECTION
    assert reversed_turn.transition is TurnTransition.REVERSAL
    assert neutral.transition is TurnTransition.NEUTRAL
    assert keyboard.presses[0] == (VKEY["q"], 0.015)

    expected_release = (VKEY["z"], VKEY["q"], VKEY["d"])
    assert keyboard.release_batches == [expected_release] * 8
    assert keyboard.release_all_calls == 8


def test_failed_press_still_releases_every_movement_key() -> None:
    keyboard = FakeKeyboard()
    keyboard.fail_press = True
    controller = MappingController(keyboard)

    with pytest.raises(RuntimeError, match="press failed"):
        controller.turn_right(0.1)

    expected_release = (VKEY["z"], VKEY["q"], VKEY["d"])
    assert keyboard.release_batches == [expected_release, expected_release]
    assert keyboard.release_all_calls == 2
    assert controller.previous_turn_direction is None


def test_validated_neutral_timeout_can_replace_provisional_threshold() -> None:
    keyboard = FakeKeyboard()
    clock = FakeClock()
    controller = MappingController(
        keyboard,
        neutral_after_seconds=7.25,
        clock=clock,
    )

    _ = controller.turn_left(0.05)
    clock.advance(1.8)
    assert controller.turn_idle_seconds == pytest.approx(1.8)

    controller.set_neutral_after_seconds(1.5, reset_history=False)
    result = controller.turn_right(0.05)

    assert controller.neutral_after_seconds == pytest.approx(1.5)
    assert result.transition is TurnTransition.NEUTRAL
    assert result.idle_seconds == pytest.approx(1.8)


def test_turn_idle_is_sampled_after_pre_press_key_release() -> None:
    clock = FakeClock()

    class DelayedReleaseKeyboard(FakeKeyboard):
        def release_keys(self, keys) -> None:
            super().release_keys(keys)
            clock.advance(0.03)

        def release_all(self) -> None:
            super().release_all()
            clock.advance(0.02)

    keyboard = DelayedReleaseKeyboard()
    controller = MappingController(
        keyboard,
        neutral_after_seconds=2.0,
        clock=clock,
    )

    _ = controller.turn_left(0.05)
    clock.advance(0.40)
    repeated = controller.turn_left(0.05)

    assert repeated.idle_seconds == pytest.approx(0.45)


def test_turn_degrees_uses_observed_idle_without_waiting() -> None:
    clock = FakeClock()
    keyboard = FakeKeyboard()
    controller = MappingController(
        keyboard,
        neutral_after_seconds=2.0,
        clock=clock,
    )
    curve = DirectionIdleResponseCurve(
        idle_seconds=(0.0, 0.5, 1.0, 2.0),
        response_progress=(0.0, 0.25, 0.75, 1.0),
        source_sample_count=8,
        stateful_response_degrees=20.0,
        neutral_response_degrees=30.0,
        maximum_monotonic_adjustment_degrees=0.0,
    )
    profile = DirectionRotationProfile(
        neutral=RotationTiming(300.0, 0.0, 4, 0.2),
        same_direction=RotationTiming(200.0, 0.0, 4, 0.2),
        reversal=RotationTiming(150.0, 0.0, 4, 0.2),
    )
    model = StateAwareRotationModel(
        left=profile,
        right=profile,
        neutral_after_seconds=2.0,
        idle_response_curves=IdleResponseCurves(left=curve, right=curve),
    )

    _ = controller.turn_left(0.05)
    clock.advance(0.75)
    before = clock.now
    result = controller.turn_degrees(
        TurnDirection.RIGHT,
        30.0,
        model,
        maximum_seconds=1.0,
    )

    # Progress at 0.75s is 0.5, blending reversal 0.2s to neutral 0.1s.
    assert result.transition is TurnTransition.REVERSAL
    assert result.idle_seconds == pytest.approx(0.75)
    assert keyboard.presses[-1] == (VKEY["d"], pytest.approx(0.15))
    # The controller sampled idle and issued the compensated pulse immediately.
    assert clock.now == before
