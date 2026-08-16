from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

from libs.HumanKeyboard import KeyPressTiming
from worker_manager import CancellationToken

from .AdaptiveMappingController import AdaptiveMappingController
from .AdaptiveMotionModel import AdaptiveMotionModel, TurnDirection
from .MinimapHeading import (
    HeadingReading,
    observed_heading_delta,
    signed_angle_delta,
)

HeadingReader = Callable[[str, int], HeadingReading]
HeadingRecovery = Callable[[TurnDirection, str, int], HeadingReading]
StatusCallback = Callable[[str], None]
ModelUpdateCallback = Callable[[], None]


class AdaptiveTurnError(RuntimeError):
    """A closed-loop adaptive turn could not be completed safely."""


@dataclass(frozen=True)
class AdaptiveTurnResult:
    target_heading: float
    final_reading: HeadingReading
    corrections: int
    pulses: tuple[KeyPressTiming, ...]
    model_updates: int


class AdaptiveTurnController:
    """
    Conservative closed-loop turn controller that learns while it moves.

    Every pulse is deliberately small and separated by a neutral wait. That
    avoids requiring the old pre-run turn-state calibration. The observed
    minimap heading delta updates the next pulse's seconds-per-degree estimate.
    """

    def __init__(
        self,
        controller: AdaptiveMappingController,
        model: AdaptiveMotionModel,
        *,
        read_heading: HeadingReader,
        cancellation: CancellationToken,
        recover_heading: HeadingRecovery | None = None,
        status_callback: StatusCallback | None = None,
        model_update_callback: ModelUpdateCallback | None = None,
        neutral_wait_seconds: float = 0.75,
        settle_seconds: float = 0.35,
        tolerance_degrees: float = 5.0,
        settled_tolerance_degrees: float = 8.0,
        settled_confirmation_seconds: float = 0.18,
        maximum_settled_drift_degrees: float = 3.5,
        maximum_uncertainty_degrees: float = 8.0,
        maximum_step_degrees: float = 28.0,
        maximum_pulse_seconds: float = 0.140,
        maximum_corrections: int = 8,
    ) -> None:
        if neutral_wait_seconds < 0.0 or settle_seconds < 0.0:
            raise ValueError("turn waits cannot be negative")
        if tolerance_degrees <= 0.0:
            raise ValueError("turn tolerance must be positive")
        if not tolerance_degrees <= settled_tolerance_degrees <= 12.0:
            raise ValueError(
                "settled_tolerance_degrees must be between the normal "
                "tolerance and 12 degrees"
            )
        if settled_confirmation_seconds < 0.0:
            raise ValueError("settled confirmation wait cannot be negative")
        if maximum_settled_drift_degrees <= 0.0:
            raise ValueError("maximum settled drift must be positive")
        if maximum_uncertainty_degrees <= 0.0:
            raise ValueError("maximum uncertainty must be positive")
        if not 2.0 <= maximum_step_degrees <= 45.0:
            raise ValueError("maximum_step_degrees must be in [2, 45]")
        if maximum_pulse_seconds < 0.015:
            raise ValueError("maximum_pulse_seconds must be at least 15 ms")
        if maximum_corrections < 1:
            raise ValueError("maximum_corrections must be positive")

        self.controller = controller
        self.model = model
        self.read_heading = read_heading
        self.recover_heading = recover_heading
        self.cancellation = cancellation
        self.status = status_callback or (lambda _message: None)
        self.model_update_callback = model_update_callback
        self.neutral_wait_seconds = float(neutral_wait_seconds)
        self.settle_seconds = float(settle_seconds)
        self.tolerance_degrees = float(tolerance_degrees)
        self.settled_tolerance_degrees = float(settled_tolerance_degrees)
        self.settled_confirmation_seconds = float(settled_confirmation_seconds)
        self.maximum_settled_drift_degrees = float(
            maximum_settled_drift_degrees
        )
        self.maximum_uncertainty_degrees = float(maximum_uncertainty_degrees)
        self.maximum_step_degrees = float(maximum_step_degrees)
        self.maximum_pulse_seconds = float(maximum_pulse_seconds)
        self.maximum_corrections = int(maximum_corrections)
        # Flyff can retain turn state across pulses for several seconds. Track
        # the last requested direction instead of pretending a short idle wait
        # always returns the game to a neutral response.
        self._last_pulse_direction: TurnDirection | None = None

    def turn_to_heading(
        self,
        target_heading: float,
        *,
        label: str,
        initial_reading: HeadingReading | None = None,
    ) -> AdaptiveTurnResult:
        target = float(target_heading) % 360.0
        pulses: list[KeyPressTiming] = []
        model_updates = 0
        stalled_reads = 0
        worsening_reads = 0

        try:
            current = initial_reading or self.read_heading(
                f"{label}: initial heading",
                15,
            )
            for correction in range(self.maximum_corrections + 1):
                self.cancellation.raise_if_cancelled()
                self._validate_reading(current, label)
                error = signed_angle_delta(target, current.angle_deg)

                if abs(error) <= self.tolerance_degrees:
                    final = current
                    if current.sample_count < 12:
                        final = self.read_heading(f"{label}: final confirmation", 15)
                        self._validate_reading(final, label)
                    final_error = signed_angle_delta(target, final.angle_deg)
                    if abs(final_error) <= self.tolerance_degrees:
                        self.status(
                            f"{label} confirmed at {final.angle_deg:.1f}° "
                            f"(target {target:.1f}°, error {final_error:+.1f}°)."
                        )
                        return AdaptiveTurnResult(
                            target_heading=target,
                            final_reading=final,
                            corrections=correction,
                            pulses=tuple(pulses),
                            model_updates=model_updates,
                        )
                    current = final
                    error = final_error

                if correction >= self.maximum_corrections:
                    break

                direction = (
                    TurnDirection.RIGHT if error > 0.0 else TurnDirection.LEFT
                )
                planned_degrees = min(
                    self.maximum_step_degrees,
                    max(4.0, abs(error) * 0.72),
                )
                duration = self.model.seconds_for_turn(
                    direction,
                    planned_degrees,
                    maximum_seconds=self.maximum_pulse_seconds,
                )
                self.status(
                    f"{label} correction {correction + 1}: "
                    f"{current.angle_deg:.1f}° -> {target:.1f}° "
                    f"(error {error:+.1f}°), {direction.value} "
                    f"{duration * 1000.0:.0f} ms."
                )

                if self._last_pulse_direction is None:
                    transition = "initial"
                elif self._last_pulse_direction is direction:
                    transition = "same"
                else:
                    transition = "reversal"

                self.controller.stop()
                if self.cancellation.wait(self.neutral_wait_seconds):
                    self.cancellation.raise_if_cancelled()

                if direction is TurnDirection.LEFT:
                    pulse = self.controller.turn_left(duration)
                else:
                    pulse = self.controller.turn_right(duration)
                pulses.append(pulse)
                self._last_pulse_direction = direction

                if self.cancellation.wait(self.settle_seconds):
                    self.cancellation.raise_if_cancelled()
                result_label = f"{label}: correction {correction + 1} result"
                try:
                    after = self.read_heading(result_label, 9)
                except Exception:
                    self.cancellation.raise_if_cancelled()
                    if self.recover_heading is None:
                        raise
                    self.status(
                        f"{result_label} could not be read; trying bounded "
                        "same-direction heading search pulses."
                    )
                    after = self.recover_heading(direction, result_label, 9)
                self._validate_reading(after, label)

                signed_motion = observed_heading_delta(after, current)
                directed_motion = (
                    -signed_motion
                    if direction is TurnDirection.LEFT
                    else signed_motion
                )
                motion_guard = max(
                    2.0,
                    0.5 * (self._uncertainty(current) + self._uncertainty(after)),
                )
                maximum_plausible = max(45.0, planned_degrees * 2.2 + 8.0)
                if abs(signed_motion) - motion_guard > maximum_plausible:
                    raise AdaptiveTurnError(
                        f"{label} rejected a probable heading alias: "
                        f"{signed_motion:+.1f}° observed for a "
                        f"{planned_degrees:.1f}° plan."
                    )
                tolerated_reversal_backlash = False
                if directed_motion < -motion_guard:
                    tolerated_reversal_backlash = (
                        transition == "reversal"
                        and abs(directed_motion) <= 8.0 + motion_guard
                    )
                    if tolerated_reversal_backlash:
                        self.status(
                            f"{label} observed {abs(directed_motion):.1f}° of "
                            "transient reversal backlash; the next bounded pulse "
                            "will use the now-established direction."
                        )
                    else:
                        raise AdaptiveTurnError(
                            f"{label} aborted because a {direction.value} pulse "
                            f"appeared to move the opposite way "
                            f"({signed_motion:+.1f}°)."
                        )

                updated = False
                if not tolerated_reversal_backlash and transition != "reversal":
                    updated = self.model.observe_turn(
                        direction,
                        held_seconds=pulse.held_seconds,
                        signed_motion_degrees=signed_motion,
                        uncertainty_degrees=motion_guard,
                    )
                if updated:
                    model_updates += 1
                    if self.model_update_callback is not None:
                        self.model_update_callback()

                if tolerated_reversal_backlash:
                    # One opposite transient is a known state transition, not a
                    # failed same-direction command. Give the newly established
                    # direction a clean pulse before applying the stall gate.
                    stalled_reads = 0
                elif directed_motion < max(1.0, motion_guard * 0.5):
                    stalled_reads += 1
                else:
                    stalled_reads = 0
                if stalled_reads >= 3:
                    raise AdaptiveTurnError(
                        f"{label} stopped after three pulses without reliable progress."
                    )

                new_error = signed_angle_delta(target, after.angle_deg)
                if tolerated_reversal_backlash:
                    worsening_reads = 0
                elif abs(new_error) > abs(error) + motion_guard:
                    worsening_reads += 1
                else:
                    worsening_reads = 0
                if worsening_reads >= 2:
                    raise AdaptiveTurnError(
                        f"{label} stopped because heading error increased twice."
                    )
                current = after

            final_error = signed_angle_delta(target, current.angle_deg)
            settled = self._confirm_bounded_settle(
                target=target,
                label=label,
                current=current,
                correction=self.maximum_corrections,
                pulses=pulses,
                model_updates=model_updates,
            )
            if settled is not None:
                return settled
            raise AdaptiveTurnError(
                f"{label} could not reach {target:.1f}° after "
                f"{self.maximum_corrections} corrections; "
                f"final error {final_error:+.1f}°."
            )
        except Exception:
            self.controller.stop()
            raise

    def _confirm_bounded_settle(
        self,
        *,
        target: float,
        label: str,
        current: HeadingReading,
        correction: int,
        pulses: list[KeyPressTiming],
        model_updates: int,
    ) -> AdaptiveTurnResult | None:
        """Accept a stable near-cardinal result instead of oscillating forever.

        Flyff can retain a small amount of turn momentum and reversal backlash.
        Near the target, another minimum key pulse can overshoot farther than the
        remaining error. The normal five-degree band is still preferred. This
        secondary band is used only after the correction budget is exhausted and
        only when a fresh, high-sample heading confirms that the view has settled.
        """

        current_error = signed_angle_delta(target, current.angle_deg)
        if abs(current_error) > self.settled_tolerance_degrees:
            return None

        self.controller.stop()
        if self.cancellation.wait(self.settled_confirmation_seconds):
            self.cancellation.raise_if_cancelled()
        final = self.read_heading(
            f"{label}: bounded final settle confirmation",
            15,
        )
        self._validate_reading(final, label)
        final_error = signed_angle_delta(target, final.angle_deg)
        settled_drift = abs(
            signed_angle_delta(final.angle_deg, current.angle_deg)
        )
        uncertainty_allowance = 0.5 * (
            self._uncertainty(current) + self._uncertainty(final)
        )
        maximum_drift = max(
            self.maximum_settled_drift_degrees,
            uncertainty_allowance,
        )
        if (
            abs(final_error) > self.settled_tolerance_degrees
            or settled_drift > maximum_drift
        ):
            return None

        self.status(
            f"{label} accepted after bounded final settle at "
            f"{final.angle_deg:.1f}° (target {target:.1f}°, error "
            f"{final_error:+.1f}°); the view stayed stable within the "
            f"{self.settled_tolerance_degrees:.1f}° operational band, so no "
            "additional reversal pulse was issued."
        )
        return AdaptiveTurnResult(
            target_heading=target,
            final_reading=final,
            corrections=correction,
            pulses=tuple(pulses),
            model_updates=model_updates,
        )

    def _validate_reading(self, reading: HeadingReading, label: str) -> None:
        if reading.is_stale:
            raise AdaptiveTurnError(f"{label} received a stale heading reading")
        if not isfinite(float(reading.angle_deg)):
            raise AdaptiveTurnError(f"{label} received a non-finite heading")
        uncertainty = self._uncertainty(reading)
        if uncertainty > self.maximum_uncertainty_degrees:
            raise AdaptiveTurnError(
                f"{label} heading uncertainty is {uncertainty:.1f}°, "
                f"above the {self.maximum_uncertainty_degrees:.1f}° limit."
            )

    @staticmethod
    def _uncertainty(reading: HeadingReading) -> float:
        value = reading.angular_uncertainty_deg
        return float(value) if value is not None else 6.0
