from __future__ import annotations

from collections import deque

import pytest
from mapper.MinimapHeading import HeadingReading
from mapper.RotationModel import (
    DirectionRotationProfile,
    RotationTiming,
    StateAwareRotationModel,
    TurnDirection,
    TurnPulseResult,
    TurnTransition,
)
from mapper.TurnControl import (
    ClosedLoopTurnController,
    TurnControlError,
)
from worker_manager import CancellationToken, WorkerCancelled


def _reading(angle: float, uncertainty: float = 1.0) -> HeadingReading:
    return HeadingReading(
        angle_deg=angle,
        confidence=0.9,
        center=(10, 10),
        radius=10,
        angular_uncertainty_deg=uncertainty,
    )


def _model() -> StateAwareRotationModel:
    same = RotationTiming(300.0, 0.0, 6, 1.0)
    reversal = RotationTiming(300.0, 0.04, 6, 1.0)
    profile = DirectionRotationProfile(
        neutral=reversal,
        same_direction=same,
        reversal=reversal,
    )
    return StateAwareRotationModel(
        left=profile,
        right=profile,
        neutral_after_seconds=2.0,
    )


class FakeController:
    def __init__(self) -> None:
        self.previous_turn_direction: TurnDirection | None = None
        self.requested: list[tuple[TurnDirection, float]] = []
        self.stop_calls = 0

    def turn_left(self, seconds: float) -> TurnPulseResult:
        return self._turn(TurnDirection.LEFT, seconds)

    def turn_right(self, seconds: float) -> TurnPulseResult:
        return self._turn(TurnDirection.RIGHT, seconds)

    def turn_degrees(
        self,
        direction: TurnDirection,
        degrees: float,
        rotation_model: StateAwareRotationModel,
        *,
        maximum_seconds: float,
    ) -> TurnPulseResult:
        previous = self.previous_turn_direction
        transition = (
            TurnTransition.NEUTRAL
            if previous is None
            else (
                TurnTransition.SAME_DIRECTION
                if previous is direction
                else TurnTransition.REVERSAL
            )
        )
        seconds = rotation_model.seconds_for(direction, transition, degrees)
        return self._turn(direction, min(maximum_seconds, seconds))

    def _turn(
        self,
        direction: TurnDirection,
        seconds: float,
    ) -> TurnPulseResult:
        previous = self.previous_turn_direction
        transition = (
            TurnTransition.NEUTRAL
            if previous is None
            else (
                TurnTransition.SAME_DIRECTION
                if previous is direction
                else TurnTransition.REVERSAL
            )
        )
        self.previous_turn_direction = direction
        self.requested.append((direction, seconds))
        return TurnPulseResult(
            direction=direction,
            transition=transition,
            requested_seconds=seconds,
            clamped_seconds=seconds,
            held_seconds=seconds,
            elapsed_seconds=seconds,
            idle_seconds=None,
        )

    def stop(self) -> None:
        self.stop_calls += 1


def _turner(
    controller: FakeController,
    readings: list[HeadingReading],
) -> ClosedLoopTurnController:
    queued = deque(readings)
    return ClosedLoopTurnController(
        controller,  # type: ignore[arg-type]
        _model(),
        left_heading_sign=-1,
        right_heading_sign=1,
        read_heading=lambda _context: queued.popleft(),
        cancellation=CancellationToken(),
        settle_seconds=0.0,
    )


def test_closed_loop_turn_confirms_observed_target_before_success() -> None:
    controller = FakeController()
    turner = _turner(
        controller,
        [_reading(90.0), _reading(55.0), _reading(25.0), _reading(2.0)],
    )

    result = turner.turn_to_heading(0.0)

    assert result.final_reading.angle_deg == 2.0
    assert result.corrections == 3
    assert all(direction is TurnDirection.LEFT for direction, _ in controller.requested)
    assert result.pulses[0].transition is TurnTransition.NEUTRAL
    assert result.pulses[1].transition is TurnTransition.SAME_DIRECTION


def test_closed_loop_turn_does_not_pulse_when_already_confirmed() -> None:
    controller = FakeController()
    turner = _turner(controller, [_reading(359.0)])

    result = turner.turn_to_heading(0.0)

    assert result.corrections == 0
    assert controller.requested == []


def test_closed_loop_turn_aborts_on_wrong_direction_motion() -> None:
    controller = FakeController()
    turner = _turner(controller, [_reading(90.0), _reading(110.0)])

    with pytest.raises(TurnControlError, match="opposite"):
        turner.turn_to_heading(0.0)

    assert controller.stop_calls == 1


def test_closed_loop_turn_rejects_uncertain_heading() -> None:
    controller = FakeController()
    turner = _turner(controller, [_reading(90.0, uncertainty=4.0)])

    with pytest.raises(TurnControlError, match="uncertainty"):
        turner.turn_to_heading(0.0)

    assert controller.requested == []


def test_closed_loop_turn_rechecks_cancellation_immediately_before_pulse() -> None:
    controller = FakeController()
    token = CancellationToken()

    def cancel_when_planned(message: str) -> None:
        if "planning a state-aware" in message:
            token.cancel()

    turner = ClosedLoopTurnController(
        controller,  # type: ignore[arg-type]
        _model(),
        left_heading_sign=-1,
        right_heading_sign=1,
        read_heading=lambda _context: _reading(90.0),
        cancellation=token,
        status_callback=cancel_when_planned,
        settle_seconds=0.0,
    )

    with pytest.raises(WorkerCancelled):
        turner.turn_to_heading(0.0)

    assert controller.requested == []
    assert controller.stop_calls == 1
