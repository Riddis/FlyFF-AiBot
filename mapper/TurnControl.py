from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from time import monotonic

from runtime.worker_manager import CancellationToken

from mapper.MappingController import MappingController
from mapper.MinimapHeading import (
    HeadingReading,
    observed_heading_delta,
    signed_angle_delta,
)
from mapper.RotationModel import (
    MAXIMUM_TOLERATED_REVERSAL_BACKLASH_DEGREES,
    DirectionRotationProfile,
    RotationTiming,
    StateAwareRotationModel,
    TurnDirection,
    TurnMemoryPolicy,
    TurnPulseResult,
    TurnTransition,
)

HeadingReader = Callable[[str], HeadingReading]
StatusCallback = Callable[[str], None]
FreshnessBarrier = Callable[[float], None]

MINIMUM_PLAUSIBLE_CLOSED_LOOP_MOTION_DEGREES = 45.0
MAXIMUM_PLAUSIBLE_CLOSED_LOOP_MOTION_DEGREES = 90.0


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
        maximum_reversal_backlash_degrees: float = (
            MAXIMUM_TOLERATED_REVERSAL_BACKLASH_DEGREES
        ),
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
        if (
            maximum_reversal_backlash_degrees <= 0.0
            or maximum_reversal_backlash_degrees > 15.0
        ):
            raise ValueError("maximum_reversal_backlash_degrees must be in (0, 15]")

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
        self.maximum_reversal_backlash_degrees = float(
            maximum_reversal_backlash_degrees
        )

    def turn_to_heading(
        self,
        target_heading: float,
        *,
        label: str = "Heading turn",
        initial_reading: HeadingReading | None = None,
    ) -> ClosedLoopTurnResult:
        target = float(target_heading) % 360.0
        pulses: list[TurnPulseResult] = []
        stalled_reads = 0
        worsening_reads = 0

        try:
            current = initial_reading or self.read_heading(f"{label}: initial heading")
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
                signed_motion = observed_heading_delta(after, current)
                directed_motion = signed_motion * self.signs[direction]
                motion_guard = max(
                    1.5,
                    self._uncertainty(current) + self._uncertainty(after),
                )
                maximum_plausible_motion = maximum_plausible_closed_loop_motion_degrees(
                    planned_degrees,
                    pulse,
                )
                if abs(signed_motion) - motion_guard > maximum_plausible_motion:
                    raise TurnControlError(
                        f"{label} aborted: a pulse planned for "
                        f"{planned_degrees:.1f} degrees appeared to move "
                        f"{signed_motion:+.1f} degrees, beyond the physically "
                        f"plausible {maximum_plausible_motion:.1f}-degree bound. "
                        "The result is probably a minimap heading alias and "
                        "was not adopted as the current heading."
                    )
                if directed_motion < -motion_guard:
                    if (
                        pulse.transition is TurnTransition.REVERSAL
                        and abs(directed_motion)
                        <= self.maximum_reversal_backlash_degrees
                    ):
                        self.status(
                            f"{label} observed {abs(directed_motion):.1f}° of "
                            f"transient opposite motion on a {direction.value} "
                            "reversal pulse; continuing with a bounded "
                            "same-direction correction."
                        )
                    else:
                        raise TurnControlError(
                            f"{label} aborted: {direction.value} pulse moved "
                            "opposite its calibrated sign "
                            f"({signed_motion:+.1f}°)."
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


def maximum_plausible_closed_loop_motion_degrees(
    planned_degrees: float,
    pulse: TurnPulseResult,
) -> float:
    """
    Bound observed motion using the command plan and actual key-hold timing.

    Closed-loop corrections request at most a modest turn. A generous
    two-times overshoot allowance plus ten degrees covers timing/model error,
    while the hard 90-degree ceiling rejects the minimap arrow's common
    120-180-degree visual aliases before they can become controller state.
    """
    planned = float(planned_degrees)
    if not isfinite(planned) or planned <= 0.0:
        raise ValueError("planned turn degrees must be finite and positive")

    intended_hold = max(float(pulse.clamped_seconds), 0.015)
    actual_hold = float(pulse.held_seconds)
    if not isfinite(actual_hold) or actual_hold < 0.0:
        raise ValueError("pulse hold duration must be finite and non-negative")
    timing_scale = max(1.0, actual_hold / intended_hold)
    timing_adjusted_plan = planned * timing_scale
    return min(
        MAXIMUM_PLAUSIBLE_CLOSED_LOOP_MOTION_DEGREES,
        max(
            MINIMUM_PLAUSIBLE_CLOSED_LOOP_MOTION_DEGREES,
            timing_adjusted_plan * 2.0 + 10.0,
        ),
    )


def uniform_rotation_model(
    *,
    left_seconds_90: float,
    right_seconds_90: float,
    neutral_after_seconds: float | None = 2.0,
    turn_memory_policy: TurnMemoryPolicy | None = None,
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

    left = profile(left_seconds_90)
    right = profile(right_seconds_90)
    if turn_memory_policy is not None:
        return StateAwareRotationModel(
            left=left,
            right=right,
            turn_memory_policy=turn_memory_policy,
        )
    if neutral_after_seconds is None:
        raise ValueError(
            "neutral_after_seconds is required without a turn-memory policy"
        )
    return StateAwareRotationModel(
        left=left,
        right=right,
        neutral_after_seconds=neutral_after_seconds,
    )
