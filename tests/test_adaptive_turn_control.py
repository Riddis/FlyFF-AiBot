from __future__ import annotations

import sys
import types

# The production project runs on Windows with pywin32. These unit tests exercise
# only the adaptive controller logic, so lightweight import stubs are sufficient
# on non-Windows CI hosts.
if "win32api" not in sys.modules:
    win32api = types.ModuleType("win32api")
    win32api.PostMessage = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    sys.modules["win32api"] = win32api
if "win32con" not in sys.modules:
    win32con = types.ModuleType("win32con")
    win32con.WM_KEYDOWN = 0x0100  # type: ignore[attr-defined]
    win32con.WM_KEYUP = 0x0101  # type: ignore[attr-defined]
    sys.modules["win32con"] = win32con

from libs.HumanKeyboard import KeyPressTiming
from mapper.AdaptiveMotionModel import AdaptiveMotionModel, TurnDirection
from mapper.AdaptiveTurnControl import AdaptiveTurnController, AdaptiveTurnError
from mapper.MinimapHeading import HeadingReading
from runtime.worker_manager import CancellationToken


class FakeController:
    def __init__(self, model: AdaptiveMotionModel, heading: float = 0.0) -> None:
        self.model = model
        self.heading = heading
        self.reverse_motion = False
        self.reversal_backlash_once = False
        self.backlash_used = False

    def stop(self) -> None:
        return None

    def turn_left(self, seconds: float) -> KeyPressTiming:
        sign = 1.0 if self.reverse_motion else -1.0
        self.heading = (
            self.heading + sign * seconds / self.model.left_seconds_per_degree
        ) % 360.0
        return KeyPressTiming(seconds, seconds, seconds, seconds)

    def turn_right(self, seconds: float) -> KeyPressTiming:
        if self.reversal_backlash_once and not self.backlash_used:
            self.heading = (self.heading - 5.0) % 360.0
            self.backlash_used = True
        else:
            sign = -1.0 if self.reverse_motion else 1.0
            self.heading = (
                self.heading + sign * seconds / self.model.right_seconds_per_degree
            ) % 360.0
        return KeyPressTiming(seconds, seconds, seconds, seconds)


def reading(controller: FakeController, samples: int) -> HeadingReading:
    return HeadingReading(
        angle_deg=controller.heading,
        confidence=0.90,
        center=(0, 0),
        radius=10,
        angular_uncertainty_deg=1.0,
        ambiguity=0.10,
        sample_count=samples,
        motion_angle_deg=controller.heading,
    )


def test_turn_controller_reaches_right_cardinal_and_learns() -> None:
    model = AdaptiveMotionModel()
    controller = FakeController(model)
    turner = AdaptiveTurnController(
        controller,  # type: ignore[arg-type]
        model,
        read_heading=lambda _label, samples: reading(controller, samples),
        cancellation=CancellationToken(),
        neutral_wait_seconds=0.0,
        settle_seconds=0.0,
    )

    result = turner.turn_to_heading(90.0, label="test right")

    assert abs(result.final_reading.angle_deg - 90.0) <= 5.0
    assert len(result.pulses) >= 1
    assert result.model_updates >= 1
    assert model.right_turn_samples >= 1


def test_turn_controller_reaches_left_cardinal_across_zero() -> None:
    model = AdaptiveMotionModel()
    controller = FakeController(model, heading=5.0)
    turner = AdaptiveTurnController(
        controller,  # type: ignore[arg-type]
        model,
        read_heading=lambda _label, samples: reading(controller, samples),
        cancellation=CancellationToken(),
        neutral_wait_seconds=0.0,
        settle_seconds=0.0,
    )

    result = turner.turn_to_heading(270.0, label="test left")

    error = (270.0 - result.final_reading.angle_deg + 180.0) % 360.0 - 180.0
    assert abs(error) <= 5.0
    assert model.left_turn_samples >= 1


def test_opposite_observed_motion_fails_closed() -> None:
    model = AdaptiveMotionModel()
    controller = FakeController(model)
    controller.reverse_motion = True
    turner = AdaptiveTurnController(
        controller,  # type: ignore[arg-type]
        model,
        read_heading=lambda _label, samples: reading(controller, samples),
        cancellation=CancellationToken(),
        neutral_wait_seconds=0.0,
        settle_seconds=0.0,
    )

    try:
        turner.turn_to_heading(90.0, label="wrong direction")
    except AdaptiveTurnError:
        pass
    else:
        raise AssertionError("opposite observed motion should fail closed")


def test_one_reversal_backlash_is_tolerated_then_turn_completes() -> None:
    model = AdaptiveMotionModel()
    controller = FakeController(model)
    controller.reversal_backlash_once = True
    turner = AdaptiveTurnController(
        controller,  # type: ignore[arg-type]
        model,
        read_heading=lambda _label, samples: reading(controller, samples),
        cancellation=CancellationToken(),
        neutral_wait_seconds=0.0,
        settle_seconds=0.0,
    )
    turner._last_pulse_direction = TurnDirection.LEFT

    result = turner.turn_to_heading(90.0, label="reversal backlash")

    error = (90.0 - result.final_reading.angle_deg + 180.0) % 360.0 - 180.0
    assert abs(error) <= 5.0
    assert controller.backlash_used
    assert len(result.pulses) >= 2


def test_heading_recovery_callback_is_used_after_ambiguous_post_pulse_read() -> None:
    model = AdaptiveMotionModel()
    controller = FakeController(model)
    failed_once = False
    recoveries: list[TurnDirection] = []

    def read_with_one_failure(_label: str, samples: int) -> HeadingReading:
        nonlocal failed_once
        if controller.heading > 1.0 and not failed_once:
            failed_once = True
            raise RuntimeError("ambiguous heading")
        return reading(controller, samples)

    def recover(
        direction: TurnDirection,
        _label: str,
        samples: int,
    ) -> HeadingReading:
        recoveries.append(direction)
        return reading(controller, samples)

    turner = AdaptiveTurnController(
        controller,  # type: ignore[arg-type]
        model,
        read_heading=read_with_one_failure,
        recover_heading=recover,
        cancellation=CancellationToken(),
        neutral_wait_seconds=0.0,
        settle_seconds=0.0,
    )

    result = turner.turn_to_heading(90.0, label="recover heading")

    assert failed_once
    assert recoveries
    assert recoveries[0] is TurnDirection.RIGHT
    assert abs(result.final_reading.angle_deg - 90.0) <= 5.0
