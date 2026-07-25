from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic

from worker_manager import CancellationToken

from mapper.MappingController import MappingController
from mapper.MinimapHeading import HeadingReading, signed_angle_delta
from mapper.RotationModel import (
    DirectionRotationProfile,
    RotationTiming,
    StateAwareRotationModel,
    TurnDirection,
    TurnPulseResult,
)

HeadingReader = Callable[[str], HeadingReading]
StatusCallback = Callable[[str], None]
FreshnessBarrier = Callable[[float], None]


class TurnControlError(RuntimeError):
    """A closed-loop turn could not be completed safely."""


@dataclass(frozen=True)
class ClosedLoopTurnResult:
    target_heading: float
    final_reading: HeadingReading
    corrections: int
    pulses: tuple[TurnPulseResult, ...]


class ClosedLoopTurnController:
    """
    Shared, state-aware absolute-heading controller.

    Timing is feed-forward only. Every bounded pulse is followed by a settled,
    strict heading read, and map/calibration callers receive success only after
    the observed heading satisfies the configured accuracy gate.
    """

    def __init__(
        self,
        controller: MappingController,
        rotation_model: StateAwareRotationModel,
        *,
        left_heading_sign: int,
        right_heading_sign: int,
        read_heading: HeadingReader,
        cancellation: CancellationToken,
        status_callback: StatusCallback | None = None,
        freshness_barrier: FreshnessBarrier | None = None,
        settle_seconds: float = 0.48,
        tolerance_degrees: float = 3.0,
        maximum_uncertainty_degrees: float = 3.0,
        maximum_step_degrees: float = 35.0,
        maximum_pulse_seconds: float = 0.18,
        maximum_corrections: int = 12,
    ) -> None:
        signs = {
            TurnDirection.LEFT: int(left_heading_sign),
            TurnDirection.RIGHT: int(right_heading_sign),
        }
        if set(signs.values()) != {-1, 1}:
            raise ValueError("left/right heading signs must be opposite ±1 values")
        if tolerance_degrees <= 0.0 or maximum_uncertainty_degrees <= 0.0:
            raise ValueError("heading accuracy limits must be positive")
        if maximum_step_degrees <= 0.0 or maximum_step_degrees > 90.0:
            raise ValueError("maximum_step_degrees must be in (0, 90]")
        if maximum_pulse_seconds < 0.015:
            raise ValueError("maximum_pulse_seconds must be at least 15 ms")
        if maximum_corrections < 1:
            raise ValueError("maximum_corrections must be positive")

        self.controller = controller
        self.rotation_model = rotation_model
        self.signs = signs
        self.read_heading = read_heading
        self.cancellation = cancellation
        self.status = status_callback or (lambda _message: None)
        self.freshness_barrier = freshness_barrier
        self.settle_seconds = max(0.0, float(settle_seconds))
        self.tolerance_degrees = float(tolerance_degrees)
        self.maximum_uncertainty_degrees = float(maximum_uncertainty_degrees)
        self.maximum_step_degrees = float(maximum_step_degrees)
        self.maximum_pulse_seconds = float(maximum_pulse_seconds)
        self.maximum_corrections = int(maximum_corrections)

    def turn_to_heading(
        self,
        target_heading: float,
        *,
        label: str = "Heading turn",
    ) -> ClosedLoopTurnResult:
        target = float(target_heading) % 360.0
        pulses: list[TurnPulseResult] = []
        stalled_reads = 0
        worsening_reads = 0

        try:
            current = self.read_heading(f"{label}: initial heading")
            for correction in range(self.maximum_corrections + 1):
                self._raise_if_cancelled(label)
                self._validate_reading(current, label)
                error = signed_angle_delta(target, current.angle_deg)

                uncertainty = self._uncertainty(current)
                if abs(error) + uncertainty <= self.tolerance_degrees:
                    self.status(
                        f"{label} confirmed at {current.angle_deg:.1f}° "
                        f"(target {target:.1f}°, error {error:+.1f}°, "
                        f"uncertainty {self._uncertainty(current):.1f}°)."
                    )
                    return ClosedLoopTurnResult(
                        target_heading=target,
                        final_reading=current,
                        corrections=correction,
                        pulses=tuple(pulses),
                    )

                if correction >= self.maximum_corrections:
                    break

                direction = self._direction_for_error(error)
                planned_degrees = min(
                    self.maximum_step_degrees,
                    max(2.0, abs(error) * 0.85),
                )
                self.status(
                    f"{label} correction {correction + 1}: "
                    f"{current.angle_deg:.1f}° -> {target:.1f}° "
                    f"(error {error:+.1f}°), planning a state-aware "
                    f"{direction.value} pulse for {planned_degrees:.1f}°."
                )
                self._raise_if_cancelled(label)
                pulse = self.controller.turn_degrees(
                    direction,
                    planned_degrees,
                    self.rotation_model,
                    maximum_seconds=self.maximum_pulse_seconds,
                )
                pulses.append(pulse)
                self.status(
                    f"{label} used {pulse.transition.value} timing: "
                    f"{pulse.held_seconds * 1000.0:.1f} ms held."
                )

                if self.cancellation.wait(self.settle_seconds):
                    self.cancellation.raise_if_cancelled()
                if self.freshness_barrier is not None:
                    self.freshness_barrier(monotonic())

                after = self.read_heading(
                    f"{label}: correction {correction + 1} result"
                )
                self._validate_reading(after, label)
                signed_motion = signed_angle_delta(
                    after.angle_deg,
                    current.angle_deg,
                )
                directed_motion = signed_motion * self.signs[direction]
                motion_guard = max(
                    1.5,
                    self._uncertainty(current) + self._uncertainty(after),
                )
                if directed_motion < -motion_guard:
                    raise TurnControlError(
                        f"{label} aborted: {direction.value} pulse moved "
                        f"opposite its calibrated sign ({signed_motion:+.1f}°)."
                    )

                if directed_motion < 0.75:
                    stalled_reads += 1
                else:
                    stalled_reads = 0
                if stalled_reads >= 3:
                    raise TurnControlError(
                        f"{label} aborted after three pulses without "
                        "reliable observed progress."
                    )

                new_error = signed_angle_delta(target, after.angle_deg)
                if abs(new_error) > abs(error) + motion_guard:
                    worsening_reads += 1
                else:
                    worsening_reads = 0
                if worsening_reads >= 2:
                    raise TurnControlError(
                        f"{label} aborted because strict heading error increased twice."
                    )
                current = after

            final_error = signed_angle_delta(target, current.angle_deg)
            raise TurnControlError(
                f"{label} could not reach {target:.1f}° after "
                f"{self.maximum_corrections} corrections; "
                f"final error {final_error:+.1f}°."
            )
        except Exception:
            self.controller.stop()
            raise

    def _direction_for_error(self, error: float) -> TurnDirection:
        desired_sign = 1 if error > 0.0 else -1
        for direction, sign in self.signs.items():
            if sign == desired_sign:
                return direction
        raise AssertionError("validated turn signs do not cover both directions")

    def _validate_reading(self, reading: HeadingReading, label: str) -> None:
        if reading.is_stale:
            raise TurnControlError(f"{label} received a stale heading reading.")
        uncertainty = self._uncertainty(reading)
        if uncertainty > self.maximum_uncertainty_degrees:
            raise TurnControlError(
                f"{label} heading uncertainty is {uncertainty:.1f}°, "
                f"above the {self.maximum_uncertainty_degrees:.1f}° limit."
            )

    def _raise_if_cancelled(self, label: str) -> None:
        del label
        self.cancellation.raise_if_cancelled()

    @staticmethod
    def _uncertainty(reading: HeadingReading) -> float:
        uncertainty = reading.angular_uncertainty_deg
        return float(uncertainty) if uncertainty is not None else 3.0


def uniform_rotation_model(
    *,
    left_seconds_90: float,
    right_seconds_90: float,
    neutral_after_seconds: float = 2.0,
) -> StateAwareRotationModel:
    """Build a conservative state-neutral model for early calibration stages."""

    def profile(seconds_90: float) -> DirectionRotationProfile:
        if seconds_90 <= 0.0:
            raise ValueError("90-degree timing must be positive")
        timing = RotationTiming(
            rate_degrees_per_second=90.0 / float(seconds_90),
            dead_time_seconds=0.0,
            sample_count=0,
            median_error_degrees=0.0,
            is_fallback=True,
        )
        return DirectionRotationProfile(
            neutral=timing,
            same_direction=timing,
            reversal=timing,
        )

    return StateAwareRotationModel(
        left=profile(left_seconds_90),
        right=profile(right_seconds_90),
        neutral_after_seconds=neutral_after_seconds,
    )
