from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from threading import Event
from time import perf_counter, sleep
from typing import Callable
import json

from mapper.MappingController import MappingController
from mapper.MinimapHeading import (
    MinimapHeadingDetector,
    signed_angle_delta,
)


@dataclass(frozen=True)
class TurnTrial:
    direction: str
    heading_sign: int
    start_heading: float
    end_heading: float
    measured_degrees: float
    pulse_seconds: float
    normalized_seconds_45: float
    confidence: float


@dataclass(frozen=True)
class CalibrationResult:
    version: int
    created_at: str
    source: str
    left_seconds_90: float
    right_seconds_90: float
    left_heading_sign: int
    right_heading_sign: int
    left_trials: list[dict]
    right_trials: list[dict]
    refinement_trials: list[dict]


class RotationCalibrator:
    """
    Conservative minimap calibration based on short, settled key bursts.

    The calibrator never holds a turn key while waiting for vision. It:
      1. starts with a tiny pulse,
      2. measures the settled minimap change,
      3. increases the pulse gradually,
      4. converges toward a 45-degree turn,
      5. derives the 90-degree duration from several 45-degree trials.

    This prevents a bad heading reading from causing a full uncontrolled spin.
    """

    TARGET_DEGREES = 45.0

    def __init__(
        self,
        bot,
        status_callback: Callable[[str], None] | None = None,
        frame_callback: Callable[[object], None] | None = None,
        *,
        trials_per_direction: int = 3,
        initial_pulse_seconds: float = 0.012,
        minimum_pulse_seconds: float = 0.008,
        maximum_pulse_seconds: float = 0.45,
        settle_seconds: float = 0.28,
        visual_confirmation_callback=None,
    ) -> None:
        if bot.keyboard is None:
            raise RuntimeError("Attach the Flyff window first.")
        self.bot = bot
        self.status = status_callback or print
        self.frame_callback = frame_callback
        self.controller = MappingController(bot.keyboard)
        self.detector = MinimapHeadingDetector()
        self.trials_per_direction = trials_per_direction
        self.initial_pulse_seconds = initial_pulse_seconds
        self.minimum_pulse_seconds = minimum_pulse_seconds
        self.maximum_pulse_seconds = maximum_pulse_seconds
        self.settle_seconds = settle_seconds
        self.visual_confirmation_callback = (
            visual_confirmation_callback
        )
        self.stop_event = Event()

    def stop(self) -> None:
        self.stop_event.set()
        self.controller.stop()

    def run(self, *, manual: bool = True) -> Path:
        # A previous GUI Stop action can leave this calibrator's Event set if
        # the same task object is reused. Always begin a new run from a clean,
        # released-input state.
        self.stop_event.clear()
        self.controller.stop()

        mode = "manual GUI calibration" if manual else "automatic recalibration"
        self.status(
            f"Starting {mode}. Calibration will use small, settled turn "
            "bursts and converge toward 45 degrees."
        )
        self.status(
            f"Minimap detector version: {self.detector.version()}"
        )

        if manual:
            for remaining in range(5, 0, -1):
                if self.stop_event.wait(1.0):
                    raise RuntimeError(
                    "Calibration stopped because the GUI stop event was set."
                )
                self.status(f"Calibration starting in {remaining}...")

        self.status("Reading initial minimap arrow...")
        initial = self._stable_heading(
            "Initial heading before calibration"
        )
        original_heading = initial.angle_deg
        self.status(
            f"Initial minimap heading: {original_heading:.1f}° "
            f"(confidence {initial.confidence:.2f})."
        )

        self.status(
            "Phase 1/2: estimating coarse left/right timing from 45-degree "
            "burst trials."
        )
        left_trials = self._calibrate_direction("left")
        right_trials = self._calibrate_direction("right")

        left_sign = self._trial_sign(left_trials)
        right_sign = self._trial_sign(right_trials)
        if left_sign == right_sign:
            raise RuntimeError(
                "Left and right appeared to move the minimap arrow in the "
                "same direction. Arrow detection or key mapping is unreliable."
            )

        left_seconds_45 = self._robust_duration(left_trials)
        right_seconds_45 = self._robust_duration(right_trials)

        coarse_left_seconds_90 = left_seconds_45 * 2.0
        coarse_right_seconds_90 = right_seconds_45 * 2.0

        self.status(
            "Coarse calibration finished: "
            f"left 90°={coarse_left_seconds_90:.4f}s, "
            f"right 90°={coarse_right_seconds_90:.4f}s."
        )

        if manual:
            self.status(
                "Phase 2/2: starting closed-loop refinement and "
                "demonstration. "
                "Each 90-degree turn will be corrected with progressively "
                "smaller pulses until the minimap is within 6 degrees."
            )
            self.status(
                "Refinement begins in 3 seconds. Use Stop immediately if "
                "movement is clearly incorrect."
            )
            if self.stop_event.wait(3.0):
                raise RuntimeError("Calibration refinement stopped.")
            turns_each_direction = 4
        else:
            self.status(
                "Starting closed-loop calibration refinement."
            )
            turns_each_direction = 2

        self.status(
            "Returning to the heading recorded before coarse calibration "
            f"({original_heading:.1f}°) before the demonstration."
        )
        self._turn_to_absolute_heading(
            target_heading=original_heading,
            left_seconds_90=coarse_left_seconds_90,
            right_seconds_90=coarse_right_seconds_90,
            left_sign=left_sign,
            right_sign=right_sign,
            label="Pre-demonstration restore",
        )

        (
            refined_left_seconds_90,
            refined_right_seconds_90,
            refinement_trials,
        ) = self._refine_and_demonstrate(
            coarse_left_seconds_90,
            coarse_right_seconds_90,
            left_sign,
            right_sign,
            turns_each_direction=turns_each_direction,
        )

        self.status(
            "Eight-turn demonstration complete. Restoring the exact heading "
            f"from the start of calibration ({original_heading:.1f}°)."
        )
        self._turn_to_absolute_heading(
            target_heading=original_heading,
            left_seconds_90=refined_left_seconds_90,
            right_seconds_90=refined_right_seconds_90,
            left_sign=left_sign,
            right_sign=right_sign,
            label="Final heading restore",
        )

        result = CalibrationResult(
            version=5,
            created_at=datetime.now().isoformat(),
            source="minimap_closed_loop_refined",
            left_seconds_90=round(refined_left_seconds_90, 5),
            right_seconds_90=round(refined_right_seconds_90, 5),
            left_heading_sign=left_sign,
            right_heading_sign=right_sign,
            left_trials=[asdict(item) for item in left_trials],
            right_trials=[asdict(item) for item in right_trials],
            refinement_trials=refinement_trials,
        )

        path = Path(__file__).resolve().parent / "calibration.json"
        path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")

        self.status(
            "Refined calibration saved: "
            f"left 90°={result.left_seconds_90:.4f}s, "
            f"right 90°={result.right_seconds_90:.4f}s."
        )

        if manual:
            final_heading = self._stable_heading()
            final_error = signed_angle_delta(
                original_heading,
                final_heading.angle_deg,
            )
            self.status(
                "Demonstration finished and original heading restored: "
                f"{final_heading.angle_deg:.1f}° "
                f"(start {original_heading:.1f}°, error "
                f"{final_error:+.1f}°)."
            )

        return path

    def _calibrate_direction(self, direction: str) -> list[TurnTrial]:
        trials: list[TurnTrial] = []
        pulse = self.initial_pulse_seconds
        learned_sign = 0
        sign_votes: list[int] = []
        attempts = 0
        maximum_attempts = 18

        self.status(
            f"{direction.title()} calibration begins with "
            f"{pulse * 1000.0:.0f} ms bursts."
        )

        while len(trials) < self.trials_per_direction:
            if self.stop_event.is_set():
                raise RuntimeError("Calibration stopped.")
            attempts += 1
            if attempts > maximum_attempts:
                raise RuntimeError(
                    f"Could not converge on a 45-degree {direction} turn."
                )

            before = self._stable_heading()
            self._turn_burst(direction, pulse)
            sleep(max(self.settle_seconds, 0.55))
            after = self._stable_heading()

            signed_change = signed_angle_delta(
                after.angle_deg,
                before.angle_deg,
            )
            magnitude = abs(signed_change)

            self.status(
                f"{direction.title()} burst {pulse * 1000.0:.1f} ms -> "
                f"{signed_change:+.1f}°."
            )

            # No meaningful motion yet: increase cautiously.
            if magnitude < 1.5:
                pulse = min(
                    self.maximum_pulse_seconds,
                    max(pulse + 0.006, pulse * 1.45),
                )
                continue

            sign = 1 if signed_change > 0.0 else -1
            sign_votes.append(sign)

            # Do not lock direction from one small reading. Require a majority
            # across at least three meaningful bursts.
            if learned_sign == 0 and len(sign_votes) >= 3:
                vote_sum = sum(sign_votes)
                if abs(vote_sum) >= 2:
                    learned_sign = 1 if vote_sum > 0 else -1
                    self.status(
                        f"{direction.title()} minimap direction learned from "
                        f"{len(sign_votes)} bursts: "
                        f"{'increasing' if learned_sign > 0 else 'decreasing'} "
                        "heading."
                    )

            if learned_sign == 0:
                # Continue increasing cautiously until direction is known.
                pulse = min(
                    self.maximum_pulse_seconds,
                    max(pulse + 0.006, pulse * 1.25),
                )
                continue

            if sign != learned_sign:
                self.status(
                    f"Ignoring inconsistent {direction} heading reading."
                )
                # Keep the current pulse; do not shrink toward zero because of
                # one vision outlier.
                continue

            # A burst over 75° is unsafe for calibration, but the key is already
            # released. Reduce sharply rather than continuing.
            if magnitude > 75.0:
                pulse = max(
                    self.minimum_pulse_seconds,
                    pulse * (self.TARGET_DEGREES / magnitude) * 0.72,
                )
                continue

            normalized = pulse * self.TARGET_DEGREES / magnitude

            # Accept a broad 30–60° window. Every accepted result is normalized
            # mathematically to the equivalent 45° duration.
            if 30.0 <= magnitude <= 60.0:
                trial = TurnTrial(
                    direction=direction,
                    heading_sign=learned_sign,
                    start_heading=round(before.angle_deg, 3),
                    end_heading=round(after.angle_deg, 3),
                    measured_degrees=round(magnitude, 3),
                    pulse_seconds=round(pulse, 5),
                    normalized_seconds_45=round(normalized, 5),
                    confidence=round(
                        min(before.confidence, after.confidence),
                        3,
                    ),
                )
                trials.append(trial)
                self.status(
                    f"Accepted {direction} trial "
                    f"{len(trials)}/{self.trials_per_direction}: "
                    f"{magnitude:.1f}°; equivalent 45° pulse "
                    f"{normalized:.4f}s."
                )

            # Damped proportional update. Limit each change so convergence is
            # gradual and cannot jump from a tiny pulse to a long spin.
            desired = normalized
            lower = pulse * 0.65
            upper = pulse * 1.35
            pulse = min(
                self.maximum_pulse_seconds,
                max(
                    self.minimum_pulse_seconds,
                    min(upper, max(lower, desired)),
                ),
            )

        return trials

    def _turn_burst(self, direction: str, seconds: float) -> None:
        started = perf_counter()
        if direction == "left":
            self.controller.turn_left(seconds)
        elif direction == "right":
            self.controller.turn_right(seconds)
        else:
            raise ValueError(f"Unknown turn direction: {direction}")
        # Send an explicit release-all after every pulse before any vision
        # measurement. This is intentionally redundant and safety-critical.
        self.controller.stop()
        elapsed = perf_counter() - started
        self.status(
            f"{direction.title()} pulse command completed after "
            f"{elapsed * 1000.0:.1f} ms; all movement keys released."
        )

    def _turn_to_absolute_heading(
        self,
        *,
        target_heading: float,
        left_seconds_90: float,
        right_seconds_90: float,
        left_sign: int,
        right_sign: int,
        label: str,
        tolerance_degrees: float = 4.5,
    ) -> None:
        """
        Return to one absolute minimap heading with closed-loop corrections.

        This compensates for all rotation accumulated during the coarse phase,
        which otherwise leaves the character far from the heading at which the
        calibration began.
        """
        durations = {
            "left": float(left_seconds_90),
            "right": float(right_seconds_90),
        }
        signs = {
            "left": int(left_sign),
            "right": int(right_sign),
        }

        for correction in range(1, 15):
            if self.stop_event.is_set():
                raise RuntimeError(
                    f"{label} stopped because the GUI stop event was set."
                )

            current = self._stable_heading()
            error = signed_angle_delta(
                target_heading,
                current.angle_deg,
            )

            if abs(error) <= tolerance_degrees:
                self.status(
                    f"{label} complete: {current.angle_deg:.1f}°, "
                    f"error {error:+.1f}°."
                )
                return

            # Select the key whose learned heading sign matches the required
            # correction direction.
            desired_sign = 1 if error > 0.0 else -1
            direction = (
                "left"
                if signs["left"] == desired_sign
                else "right"
            )

            degrees = min(abs(error), 42.0)
            pulse_seconds = (
                durations[direction]
                * degrees
                / 90.0
                * 0.86
            )
            pulse_seconds = min(0.180, max(0.018, pulse_seconds))

            self.status(
                f"{label} correction {correction}: current "
                f"{current.angle_deg:.1f}°, target {target_heading:.1f}°, "
                f"error {error:+.1f}°, {direction} "
                f"{pulse_seconds * 1000.0:.0f} ms."
            )
            self._turn_burst(direction, pulse_seconds)
            sleep(max(self.settle_seconds, 0.50))

        final = self._stable_heading()
        final_error = signed_angle_delta(
            target_heading,
            final.angle_deg,
        )
        raise RuntimeError(
            f"{label} could not reach the requested heading. "
            f"Final error: {final_error:+.1f}°."
        )

    def _refine_and_demonstrate(
        self,
        left_seconds_90: float,
        right_seconds_90: float,
        left_sign: int,
        right_sign: int,
        *,
        turns_each_direction: int,
    ) -> tuple[float, float, list[dict]]:
        """
        Execute exact 90-degree target turns with minimap feedback.

        Large movement is split into bounded pulses. After every pulse the
        settled heading is read again and the remaining error determines the
        next pulse. This is slower than a single timed press but prevents timing
        error from accumulating through the map.
        """
        durations = {
            "left": float(left_seconds_90),
            "right": float(right_seconds_90),
        }
        signs = {
            "left": int(left_sign),
            "right": int(right_sign),
        }
        samples: dict[str, list[float]] = {
            "left": [float(left_seconds_90)],
            "right": [float(right_seconds_90)],
        }
        records: list[dict] = []

        sequence = (
            ["left"] * turns_each_direction
            + ["right"] * turns_each_direction
        )

        for index, direction in enumerate(sequence, start=1):
            sign = signs[direction]
            opposite = "right" if direction == "left" else "left"
            start = self._stable_heading()
            target = (start.angle_deg + sign * 90.0) % 360.0

            self.status(
                f"Demonstration turn {index}/{len(sequence)} {direction}: "
                f"start {start.angle_deg:.1f}°, target {target:.1f}°."
            )

            total_requested_seconds = 0.0
            total_directed_motion = 0.0
            previous = start
            corrections = 0
            stalled_reads = 0

            for correction in range(1, 11):
                if self.stop_event.is_set():
                    raise RuntimeError("Calibration refinement stopped.")

                current = previous
                signed_to_target = signed_angle_delta(
                    target,
                    current.angle_deg,
                )
                directed_remaining = signed_to_target * sign

                if abs(signed_to_target) <= 6.0:
                    break

                # If the requested direction overshot the target, use the
                # opposite calibrated key for a small corrective pulse.
                pulse_direction = direction
                pulse_sign = sign
                error_degrees = directed_remaining
                if directed_remaining < 0.0:
                    pulse_direction = opposite
                    pulse_sign = signs[opposite]
                    error_degrees = abs(signed_to_target)

                seconds_per_90 = durations[pulse_direction]
                proposed = (
                    seconds_per_90
                    * min(max(error_degrees, 3.0), 42.0)
                    / 90.0
                    * 0.88
                )
                pulse_seconds = min(0.180, max(0.018, proposed))

                self.status(
                    f"  correction {correction}: remaining "
                    f"{signed_to_target:+.1f}°, "
                    f"{pulse_direction} pulse {pulse_seconds * 1000.0:.0f} ms."
                )

                self._turn_burst(pulse_direction, pulse_seconds)
                sleep(max(self.settle_seconds, 0.48))
                after = self._stable_heading()

                signed_motion = signed_angle_delta(
                    after.angle_deg,
                    current.angle_deg,
                )
                directed_motion = signed_motion * pulse_sign

                if abs(signed_motion) < 2.0:
                    stalled_reads += 1
                    self.status(
                        f"  no reliable motion measured "
                        f"({current.angle_deg:.1f}° -> "
                        f"{after.angle_deg:.1f}°)."
                    )
                    if stalled_reads >= 3:
                        debug_folder = self.detector.save_debug(
                            f"{pulse_direction}_refinement_stalled"
                        )
                        raise RuntimeError(
                            f"{pulse_direction.title()} refinement stalled "
                            "for three pulses. "
                            f"Debug saved to: {debug_folder}"
                        )
                else:
                    stalled_reads = 0

                if directed_motion < -4.0:
                    raise RuntimeError(
                        f"{pulse_direction.title()} pulse moved opposite its "
                        f"learned direction: {signed_motion:+.1f}°."
                    )

                if 4.0 <= directed_motion <= 55.0:
                    equivalent_90 = (
                        pulse_seconds * 90.0 / directed_motion
                    )
                    samples[pulse_direction].append(equivalent_90)
                    durations[pulse_direction] = self._bounded_median(
                        samples[pulse_direction],
                        durations[pulse_direction],
                    )

                if pulse_direction == direction:
                    total_requested_seconds += pulse_seconds
                    total_directed_motion += max(0.0, directed_motion)

                previous = after
                corrections = correction
            else:
                raise RuntimeError(
                    f"Could not bring the {direction} turn within 6° after "
                    "ten corrective pulses."
                )

            final = self._stable_heading()
            final_error = signed_angle_delta(
                target,
                final.angle_deg,
            )
            measured = (
                signed_angle_delta(
                    final.angle_deg,
                    start.angle_deg,
                )
                * sign
            )

            if abs(final_error) > 7.5:
                raise RuntimeError(
                    f"{direction.title()} refinement ended "
                    f"{final_error:+.1f}° from target."
                )

            if total_directed_motion >= 20.0:
                whole_turn_estimate = (
                    total_requested_seconds
                    * 90.0
                    / total_directed_motion
                )
                samples[direction].append(whole_turn_estimate)
                durations[direction] = self._bounded_median(
                    samples[direction],
                    durations[direction],
                )

            record = {
                "direction": direction,
                "start_heading": round(start.angle_deg, 3),
                "target_heading": round(target, 3),
                "final_heading": round(final.angle_deg, 3),
                "measured_degrees": round(measured, 3),
                "final_error_degrees": round(final_error, 3),
                "corrections": corrections,
                "left_seconds_90": round(durations["left"], 5),
                "right_seconds_90": round(durations["right"], 5),
            }
            records.append(record)

            self.status(
                f"Demonstration turn {index}/{len(sequence)} complete: "
                f"measured {measured:.1f}°, target error "
                f"{final_error:+.1f}°. Current timings: "
                f"L={durations['left']:.4f}s, "
                f"R={durations['right']:.4f}s."
            )

            if self.stop_event.wait(0.65):
                raise RuntimeError("Calibration refinement stopped.")

        return durations["left"], durations["right"], records

    @staticmethod
    def _bounded_median(
        values: list[float],
        previous: float,
    ) -> float:
        """
        Update timing conservatively so one bad visual reading cannot cause a
        large timing jump.
        """
        candidate = float(median(values[-9:]))
        lower = previous * 0.82
        upper = previous * 1.18
        return min(upper, max(lower, candidate))

    def _stable_heading(self, context: str = "Heading check"):
        """
        Obtain a strict heading and optionally ask the user to approve it.

        Rejected readings are discarded and reacquired from scratch.
        """
        for attempt in range(12):
            reading = self.detector.read_strict(
                self.bot.get_frame,
                samples=15,
                delay=0.015,
                fresh=True,
            )

            if reading is None or reading.confidence < 0.52:
                if attempt < 11:
                    self.status(
                        "Strict heading was ambiguous; reacquiring "
                        f"({attempt + 1}/12)."
                    )
                    sleep(0.10)
                continue

            self._publish_debug(reading)

            callback = self.visual_confirmation_callback
            if callback is None:
                return reading

            frame = self.bot.get_frame()
            accepted = callback(
                frame,
                reading.angle_deg,
                reading.confidence,
                context,
            )

            if accepted is True:
                self.status(
                    "Visual heading accepted: "
                    f"{reading.angle_deg:.1f}° "
                    f"(confidence {reading.confidence:.2f})."
                )
                return reading

            if accepted is None:
                raise RuntimeError(
                    "Calibration stopped during visual heading validation."
                )

            self.status(
                "Visual heading rejected; reacquiring from scratch."
            )
            self.detector.reset_fast()
            sleep(max(0.10, self.settle_seconds))

        debug_folder = self.detector.save_debug(
            "visual_or_strict_heading_failed"
        )
        debug_text = (
            f" Debug saved to: {debug_folder}"
            if debug_folder is not None
            else ""
        )
        raise RuntimeError(
            "Could not obtain a visually accepted strict heading after "
            f"12 attempts.{debug_text}"
        )


    def _publish_debug(self, reading) -> None:
        if self.frame_callback is None:
            return
        frame = self.bot.get_debug_frame()
        if frame is not None:
            self.frame_callback(
                self.detector.draw_debug(frame, reading)
            )

    @staticmethod
    def _trial_sign(trials: list[TurnTrial]) -> int:
        return 1 if sum(item.heading_sign for item in trials) > 0 else -1

    @staticmethod
    def _robust_duration(trials: list[TurnTrial]) -> float:
        values = [item.normalized_seconds_45 for item in trials]
        center = median(values)
        deviations = [abs(value - center) for value in values]
        mad = median(deviations) or 0.001
        filtered = [
            value
            for value in values
            if abs(value - center) <= max(3.0 * mad, 0.015)
        ]
        return float(median(filtered or values))
