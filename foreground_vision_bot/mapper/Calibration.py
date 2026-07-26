from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from statistics import median
from time import monotonic, monotonic_ns

import cv2 as cv
import numpy as np
from capture_service import FrameSample
from worker_manager import CancellationToken, WorkerCancelled

from .CalibrationSchema import MapperCalibration
from .ForwardCalibration import (
    ForwardCalibrationTrial,
    ForwardMotionModel,
    fit_forward_motion_model,
    forward_flow_coherence_error,
)
from .MappingController import MappingController
from .MinimapHeading import (
    HeadingReading,
    MinimapHeadingDetector,
    observed_heading_delta,
    signed_angle_delta,
)
from .MotionTracker import ForwardMotionOutcome, MotionTracker
from .RotationModel import (
    MAXIMUM_TOLERATED_REVERSAL_BACKLASH_DEGREES,
    NeutralTimeoutFit,
    NeutralTimeoutSample,
    RotationSample,
    StateAwareRotationModel,
    TurnDirection,
    TurnMemoryMode,
    TurnPulseResult,
    TurnTransition,
    fit_neutral_timeout,
    fit_rotation_model,
    validate_neutral_timeout,
)
from .TurnControl import (
    ClosedLoopTurnController,
    maximum_plausible_closed_loop_motion_degrees,
    uniform_rotation_model,
)


@dataclass(frozen=True)
class TurnTrial:
    direction: str
    transition: str
    heading_sign: int
    start_heading: float
    end_heading: float
    measured_degrees: float
    requested_seconds: float
    clamped_seconds: float
    pulse_seconds: float
    elapsed_seconds: float
    idle_seconds: float | None
    normalized_seconds_45: float
    confidence: float


@dataclass(frozen=True)
class CalibrationResult:
    """Legacy serializable calibration payload kept for caller/test compatibility."""

    version: int
    created_at: str
    source: str
    left_seconds_90: float
    right_seconds_90: float
    left_heading_sign: int
    right_heading_sign: int
    left_trials: list[dict[str, object]]
    right_trials: list[dict[str, object]]
    neutral_after_seconds: float | None
    neutral_timeout_trials: list[dict[str, object]]
    neutral_timeout_fit: dict[str, object]
    transition_trials: list[dict[str, object]]
    refinement_trials: list[dict[str, object]]
    rotation_model: dict[str, object]
    forward_trials: list[dict[str, object]]
    forward_model: dict[str, float | int]


@dataclass(frozen=True)
class _HeadingRecoveryPlan:
    target_heading: float
    left_seconds_90: float
    right_seconds_90: float
    left_sign: int
    right_sign: int
    rotation_model: StateAwareRotationModel | None = None


class _ProbeMeasurementError(RuntimeError):
    """One timed probe did not produce a trustworthy heading measurement."""


class _ProbeHeadingConsensusError(_ProbeMeasurementError):
    """One timed probe could not establish a required strict heading."""


class _ProbeStartHeadingError(_ProbeHeadingConsensusError):
    """A timed probe could not read its heading before issuing movement."""


class _ProbeHeadingDeltaError(_ProbeMeasurementError):
    """Two strict probe endpoints imply physically implausible rotation."""


class _ProbeCaptureError(_ProbeMeasurementError):
    """A timed probe could not capture its required fresh frame batch."""


class _ProbeTimingError(_ProbeMeasurementError):
    """A timed probe missed the requested physical idle-state window."""


class _ProbeResponseError(_ProbeMeasurementError):
    """A probe response could not be resolved within its measurement gates."""


class _PersistentHeadingFailure(RuntimeError):
    """Heading remained unreadable, so further calibrated movement is unsafe."""


class RotationCalibrator:
    """
    Conservative mapper calibration based on settled vision measurements.

    The calibrator never holds a movement key while waiting for vision. It:
      1. starts with a tiny pulse,
      2. measures the settled minimap change,
      3. measures how long direction-dependent turn state persists,
      4. separates repeated and reversed turn response,
      5. verifies closed-loop absolute-heading control,
      6. fits a relative forward-distance scale at the fixed camera setup.

    The previous calibration file is replaced only after every required phase
    succeeds.
    """

    TARGET_DEGREES = 45.0
    TURN_TOLERANCE_DEGREES = 3.0
    MAXIMUM_HEADING_UNCERTAINTY_DEGREES = 3.0
    MAXIMUM_HEADING_AMBIGUITY = 0.70
    FORWARD_NOMINAL_SECONDS = 0.12
    # Three bracketed durations, each repeated at a different physical
    # location and in reverse order. This keeps the existing six pulses while
    # separating duration response from one-off scene texture.
    FORWARD_TRIAL_SECONDS = (0.090, 0.120, 0.150, 0.150, 0.120, 0.090)
    # Start with a replicated short/horizon experiment. Only a detector that
    # actually converges at the horizon needs the intermediate decay scan.
    # The observed game state remained direction-dependent at six seconds, so
    # running every intermediate delay unconditionally added minutes without
    # answering a different question.
    NEUTRAL_PROBE_MINIMUM_SHORT_IDLE_SECONDS = 0.22
    NEUTRAL_PROBE_REFINEMENT_IDLE_SECONDS = (
        0.95,
        1.70,
        2.80,
        4.30,
    )
    NEUTRAL_PROBE_HORIZON_SECONDS = 6.00
    NEUTRAL_PROBE_SCAN_REPEATS = 2
    NEUTRAL_PROBE_MAXIMUM_SECONDS = 6.25
    NEUTRAL_PROBE_SAFETY_MARGIN_SECONDS = 0.25
    NEUTRAL_PROBE_MAXIMUM_OVERSHOOT_SECONDS = 0.18
    # A turn measurement subtracts two independently accepted headings. Each
    # endpoint remains subject to the strict three-degree gate above, so the
    # conservative uncertainty bound for their difference is the sum of both
    # endpoint bounds, not the single-heading limit.
    MAXIMUM_TURN_MEASUREMENT_UNCERTAINTY_DEGREES = (
        2.0 * MAXIMUM_HEADING_UNCERTAINTY_DEGREES
    )
    NEUTRAL_PROBE_VALIDATION_REPEATS = 2
    PROBE_HEADING_FRAME_COUNT = 5
    PROBE_FRAME_CAPTURE_LEAD_SECONDS = 0.10
    TRANSITION_MATRIX_REPEATS = 2
    PROBE_HEADING_ATTEMPTS = 3
    MAXIMUM_CONSECUTIVE_PROBE_START_FAILURES = 3
    COARSE_ATTEMPTS_PER_REQUIRED_TRIAL = 8
    COARSE_MINIMUM_MAXIMUM_ATTEMPTS = 12
    COARSE_MAXIMUM_IMPLAUSIBLE_READINGS = 3
    MAXIMUM_PLAUSIBLE_SINGLE_PULSE_DELTA_DEGREES = 120.0

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
        cancellation: CancellationToken | None = None,
    ) -> None:
        if bot.keyboard is None:
            raise RuntimeError("Attach the Flyff window first.")
        self.bot = bot
        self.status = status_callback or print
        self.frame_callback = frame_callback
        self.controller = MappingController(
            bot.keyboard,
            # This threshold is deliberately above the bounded probe window.
            # It is provisional only and is replaced by a validated fit before
            # any state-specific rotation samples are collected.
            neutral_after_seconds=self.NEUTRAL_PROBE_MAXIMUM_SECONDS + 1.0,
        )
        self.detector = MinimapHeadingDetector(automatic_debug=True)
        self.trials_per_direction = trials_per_direction
        self.initial_pulse_seconds = initial_pulse_seconds
        self.minimum_pulse_seconds = minimum_pulse_seconds
        self.maximum_pulse_seconds = maximum_pulse_seconds
        self.settle_seconds = settle_seconds
        self.visual_confirmation_callback = visual_confirmation_callback
        self.cancellation = cancellation or CancellationToken()
        self._rotation_samples: list[RotationSample] = []
        self._failure_recovery: _HeadingRecoveryPlan | None = None

    def stop(self) -> None:
        self.cancellation.cancel()
        self.controller.stop()

    def run(self, *, manual: bool = True) -> Path:
        primary_error: BaseException | None = None
        try:
            return self._run(manual=manual)
        except BaseException as error:
            primary_error = error
            if isinstance(error, Exception):
                try:
                    self._best_effort_restore_after_failure(error)
                except Exception as recovery_error:  # noqa: BLE001
                    error.add_note(
                        "Calibration failure recovery also raised unexpectedly: "
                        f"{recovery_error}"
                    )
            raise
        finally:
            try:
                self.controller.stop()
            except Exception as release_error:
                if primary_error is None:
                    raise
                primary_error.add_note(
                    "Calibration also failed to release all movement keys during "
                    f"final cleanup: {release_error}"
                )

    def _run(self, *, manual: bool) -> Path:
        self.controller.stop()
        self.controller.reset_turn_history()
        self._rotation_samples.clear()
        self._failure_recovery = None

        mode = "manual GUI calibration" if manual else "automatic recalibration"
        self.status(
            f"Starting {mode}. Calibration will measure heading accuracy, "
            "turn-state persistence, neutral/repeated/reversed response, and "
            "fixed-camera forward travel."
        )
        self.status(f"Minimap detector version: {self.detector.version()}")

        if manual:
            for remaining in range(5, 0, -1):
                if self.cancellation.wait(1.0):
                    self.cancellation.raise_if_cancelled()
                self.status(f"Calibration starting in {remaining}...")

        self.status("Reading initial minimap arrow...")
        initial = self._stable_heading("Initial heading before calibration")
        original_heading = initial.angle_deg
        self.status(
            f"Initial minimap heading: {original_heading:.1f}° "
            f"(confidence {initial.confidence:.2f})."
        )

        self.status(
            "Phase 1/4: estimating coarse left/right timing from 45-degree "
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
        self._failure_recovery = _HeadingRecoveryPlan(
            target_heading=original_heading,
            left_seconds_90=coarse_left_seconds_90,
            right_seconds_90=coarse_right_seconds_90,
            left_sign=left_sign,
            right_sign=right_sign,
        )

        self.status(
            "Coarse calibration finished: "
            f"left 90°={coarse_left_seconds_90:.4f}s, "
            f"right 90°={coarse_right_seconds_90:.4f}s."
        )

        self.status(
            "Phase 2/4: measuring the turn-state neutral timeout with bounded "
            "idle-delay probes in both directions."
        )
        neutral_timeout_fit, neutral_timeout_trials = self._calibrate_neutral_timeout(
            left_seconds_90=coarse_left_seconds_90,
            right_seconds_90=coarse_right_seconds_90,
            left_sign=left_sign,
            right_sign=right_sign,
            restore_heading=original_heading,
        )

        self.status(
            "Phase 3/4: measuring same-direction and reversal response at "
            "three safe pulse durations per direction."
        )
        transition_trials = self._measure_transition_matrix(
            left_seconds_90=coarse_left_seconds_90,
            right_seconds_90=coarse_right_seconds_90,
            left_sign=left_sign,
            right_sign=right_sign,
        )
        # Phase 3 is the first point at which SAME/REVERSAL timing has been
        # measured at several durations. Use that evidence immediately for
        # restore/refinement; falling back to one uniform coarse duration here
        # recreated the exact direction-memory error this calibration measures.
        provisional_rotation_model = fit_rotation_model(
            self._rotation_samples,
            fallback_left_seconds_90=coarse_left_seconds_90,
            fallback_right_seconds_90=coarse_right_seconds_90,
            idle_response_curves=neutral_timeout_fit.idle_response_curves,
            turn_memory_policy=neutral_timeout_fit.turn_memory_policy,
        )
        self._failure_recovery = _HeadingRecoveryPlan(
            target_heading=original_heading,
            left_seconds_90=coarse_left_seconds_90,
            right_seconds_90=coarse_right_seconds_90,
            left_sign=left_sign,
            right_sign=right_sign,
            rotation_model=provisional_rotation_model,
        )

        if manual:
            self.status(
                "Continuing Phase 3/4 with state-aware closed-loop refinement and "
                "demonstration. "
                "Each 90-degree turn will be corrected with progressively "
                "smaller pulses until the strict minimap heading is within "
                f"{self.TURN_TOLERANCE_DEGREES:.0f} degrees."
            )
            self.status(
                "Refinement begins in 3 seconds. Use Stop immediately if "
                "movement is clearly incorrect."
            )
            if self.cancellation.wait(3.0):
                self.cancellation.raise_if_cancelled()
            turns_each_direction = 4
        else:
            self.status("Starting closed-loop calibration refinement.")
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
            rotation_model=provisional_rotation_model,
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
            rotation_model=provisional_rotation_model,
        )

        rotation_model = fit_rotation_model(
            self._rotation_samples,
            fallback_left_seconds_90=refined_left_seconds_90,
            fallback_right_seconds_90=refined_right_seconds_90,
            idle_response_curves=neutral_timeout_fit.idle_response_curves,
            turn_memory_policy=neutral_timeout_fit.turn_memory_policy,
        )
        self._failure_recovery = _HeadingRecoveryPlan(
            target_heading=original_heading,
            left_seconds_90=refined_left_seconds_90,
            right_seconds_90=refined_right_seconds_90,
            left_sign=left_sign,
            right_sign=right_sign,
            rotation_model=rotation_model,
        )

        self.status(
            "State-balanced demonstration complete. Restoring the exact heading "
            f"from the start of calibration ({original_heading:.1f}°)."
        )
        self._turn_to_absolute_heading(
            target_heading=original_heading,
            left_seconds_90=refined_left_seconds_90,
            right_seconds_90=refined_right_seconds_90,
            left_sign=left_sign,
            right_sign=right_sign,
            label="Final heading restore",
            rotation_model=rotation_model,
        )

        forward_model, forward_trials = self._calibrate_forward_motion(
            original_heading=initial,
            manual=manual,
        )

        result = MapperCalibration(
            version=MapperCalibration.CURRENT_VERSION,
            created_at=datetime.now(timezone.utc).isoformat(),
            source="vision_fitted_turn_state_rotation_and_forward",
            left_seconds_90=round(refined_left_seconds_90, 5),
            right_seconds_90=round(refined_right_seconds_90, 5),
            left_heading_sign=left_sign,
            right_heading_sign=right_sign,
            left_trials=[asdict(item) for item in left_trials],
            right_trials=[asdict(item) for item in right_trials],
            neutral_after_seconds=(
                round(neutral_timeout_fit.neutral_after_seconds, 6)
                if neutral_timeout_fit.neutral_after_seconds is not None
                else None
            ),
            neutral_timeout_trials=neutral_timeout_trials,
            neutral_timeout_fit=neutral_timeout_fit.to_dict(),
            transition_trials=transition_trials,
            refinement_trials=refinement_trials,
            rotation_model=rotation_model,
            forward_trials=[asdict(item) for item in forward_trials],
            forward_model=forward_model,
        )

        final_heading = self._stable_heading("Final calibration validation")
        final_error = observed_heading_delta(final_heading, initial)
        if not self._heading_delta_satisfied(
            final_error,
            before_reading=initial,
            after_reading=final_heading,
        ):
            combined_uncertainty = self._reading_uncertainty(
                initial
            ) + self._reading_uncertainty(final_heading)
            raise RuntimeError(
                "Final heading validation failed "
                f"(error {final_error:+.1f}°, combined endpoint uncertainty "
                f"{combined_uncertainty:.1f}°). "
                "The previous calibration was preserved."
            )

        path = Path(__file__).resolve().parent / "calibration.json"
        self.cancellation.raise_if_cancelled()
        self._atomic_write_json(path, result.to_dict())

        self.status(
            "Validated mapper calibration saved: "
            f"left 90°={result.left_seconds_90:.4f}s, "
            f"right 90°={result.right_seconds_90:.4f}s, "
            f"forward scale={forward_model.pixels_per_cell:.2f}px/cell."
        )

        if manual:
            self.status(
                "Mapper calibration finished; original heading preserved: "
                f"{final_heading.angle_deg:.1f}° "
                f"(start {original_heading:.1f}°, error "
                f"{final_error:+.1f}°)."
            )

        return path

    def _best_effort_restore_after_failure(self, primary_error: Exception) -> None:
        plan = self._failure_recovery
        if isinstance(primary_error, _PersistentHeadingFailure):
            self._status_without_masking(
                "Original-heading recovery was skipped because the minimap "
                "heading remained unreadable; issuing a blind recovery turn "
                "would be unsafe.",
                primary_error,
            )
            return
        if (
            plan is None
            or isinstance(primary_error, WorkerCancelled)
            or self.cancellation.cancelled
        ):
            return

        try:
            self.controller.stop()
        except Exception as release_error:  # noqa: BLE001 - preserve primary failure.
            primary_error.add_note(
                "Original-heading recovery was skipped because calibration "
                f"could not confirm that movement keys were released: {release_error}"
            )
            self._status_without_masking(
                "Calibration failed, and original-heading recovery was skipped "
                "because movement keys could not be confirmed released.",
                primary_error,
            )
            return

        previous_confirmation_callback = self.visual_confirmation_callback
        self.visual_confirmation_callback = None
        try:
            recovery_heading = self._noninteractive_heading(
                "Failure recovery detector health check",
                timeout=0.65,
                samples=5,
            )
        except Exception as health_error:  # noqa: BLE001 - recovery must not move.
            primary_error.add_note(
                "Original-heading recovery was skipped because its single "
                f"bounded detector health check failed: {health_error}"
            )
            self._status_without_masking(
                "Calibration failed, and original-heading recovery was skipped "
                "because one short detector health check could not establish "
                "a safe heading.",
                primary_error,
            )
            self.visual_confirmation_callback = previous_confirmation_callback
            return

        self._status_without_masking(
            "Calibration failed after movement began. Attempting a bounded "
            "return to the heading recorded at the start.",
            primary_error,
        )
        try:
            self._turn_to_absolute_heading(
                target_heading=plan.target_heading,
                left_seconds_90=plan.left_seconds_90,
                right_seconds_90=plan.right_seconds_90,
                left_sign=plan.left_sign,
                right_sign=plan.right_sign,
                label="Failure recovery heading restore",
                rotation_model=plan.rotation_model,
                initial_reading=recovery_heading,
                noninteractive=True,
            )
        except Exception as restore_error:  # noqa: BLE001 - best-effort recovery.
            primary_error.add_note(
                f"The best-effort original-heading restore also failed: {restore_error}"
            )
            self._status_without_masking(
                "Calibration failed, and the best-effort original-heading "
                f"restore also failed: {restore_error}",
                primary_error,
            )
        finally:
            self.visual_confirmation_callback = previous_confirmation_callback

    def _status_without_masking(
        self,
        message: str,
        primary_error: Exception,
    ) -> None:
        try:
            self.status(message)
        except Exception as status_error:  # noqa: BLE001 - preserve primary failure.
            primary_error.add_note(
                "A recovery status callback also failed: "
                f"{type(status_error).__name__}: {status_error}"
            )

    def _calibrate_direction(self, direction: str) -> list[TurnTrial]:
        trials: list[TurnTrial] = []
        pulse = self.initial_pulse_seconds
        learned_sign = 0
        sign_votes: list[int] = []
        attempts = 0
        implausible_readings = 0
        outlier_debug_folder: Path | None = None
        maximum_attempts = max(
            self.COARSE_MINIMUM_MAXIMUM_ATTEMPTS,
            self.trials_per_direction * self.COARSE_ATTEMPTS_PER_REQUIRED_TRIAL,
        )

        self.status(
            f"{direction.title()} calibration begins with "
            f"{pulse * 1000.0:.0f} ms bursts."
        )
        self.controller.stop()
        self.status(
            f"{direction.title()} coarse calibration: movement is released. "
            "No neutral delay is assumed before turn-memory has been measured; "
            "early bursts establish the requested direction state."
        )

        while len(trials) < self.trials_per_direction:
            self.cancellation.raise_if_cancelled()
            if attempts >= maximum_attempts:
                debug_folder = outlier_debug_folder or self.detector.save_debug(
                    f"{direction}_coarse_nonconvergence"
                )
                debug_text = (
                    f" Debug saved to: {debug_folder}"
                    if debug_folder is not None
                    else ""
                )
                raise RuntimeError(
                    f"Could not converge on a 45-degree {direction} turn: "
                    f"accepted {len(trials)}/{self.trials_per_direction} trials "
                    f"after {attempts} bursts and discarded "
                    f"{implausible_readings} implausible heading readings."
                    f"{debug_text}"
                )
            attempts += 1

            before = self._stable_heading()
            pulse_result = self._turn_burst(direction, pulse)
            self._wait_or_cancel(max(self.settle_seconds, 0.55))
            after = self._stable_heading()

            signed_change = observed_heading_delta(after, before)
            magnitude = abs(signed_change)
            motion_uncertainty = self._reading_uncertainty(
                before
            ) + self._reading_uncertainty(after)

            self.status(
                f"{direction.title()} burst {pulse * 1000.0:.1f} ms -> "
                f"{signed_change:+.1f}°."
            )

            # A settled coarse pulse cannot rotate more than 120 degrees while
            # we are cautiously converging on 45 degrees. Such jumps are
            # characteristic of the minimap arrow's opposite-orientation
            # visual alias. Never let one of them vote on direction or alter
            # the next pulse duration.
            if self._single_pulse_delta_is_implausible(
                signed_change,
                motion_uncertainty,
            ):
                implausible_readings += 1
                if outlier_debug_folder is None:
                    outlier_debug_folder = self.detector.save_debug(
                        f"{direction}_coarse_implausible_delta"
                    )
                debug_text = (
                    f" Debug saved to: {outlier_debug_folder}."
                    if outlier_debug_folder is not None
                    else ""
                )
                self.status(
                    f"Discarding implausible {direction} heading jump "
                    f"({signed_change:+.1f}° with "
                    f"{motion_uncertainty:.1f}° endpoint uncertainty); "
                    f"retrying the unchanged {pulse * 1000.0:.1f} ms pulse."
                    f"{debug_text}"
                )
                if implausible_readings > self.COARSE_MAXIMUM_IMPLAUSIBLE_READINGS:
                    repeated_debug = self.detector.save_debug(
                        f"{direction}_coarse_repeated_implausible_delta"
                    )
                    debug_folder = repeated_debug or outlier_debug_folder
                    debug_text = (
                        f" Debug saved to: {debug_folder}"
                        if debug_folder is not None
                        else ""
                    )
                    raise RuntimeError(
                        f"Could not calibrate {direction}: the minimap heading "
                        f"jumped by more than "
                        f"{self.MAXIMUM_PLAUSIBLE_SINGLE_PULSE_DELTA_DEGREES:.0f} "
                        f"degrees in "
                        f"{self.COARSE_MAXIMUM_IMPLAUSIBLE_READINGS + 1} "
                        "coarse measurements. No timing was learned from those "
                        f"readings.{debug_text}"
                    )
                continue

            # No meaningful motion yet: increase cautiously.
            if magnitude <= max(1.5, motion_uncertainty):
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
                self.status(f"Ignoring inconsistent {direction} heading reading.")
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

            normalized = pulse_result.held_seconds * self.TARGET_DEGREES / magnitude

            # Accept a broad 30–60° window. Every accepted result is normalized
            # mathematically to the equivalent 45° duration.
            if 30.0 <= magnitude <= 60.0:
                if pulse_result.transition is not TurnTransition.SAME_DIRECTION:
                    self.status(
                        f"Discarding otherwise usable {direction} burst because "
                        f"it was {pulse_result.transition.value}; coarse timing "
                        "is learned only after same-direction state is established."
                    )
                    continue
                trial = TurnTrial(
                    direction=direction,
                    transition=pulse_result.transition.value,
                    heading_sign=learned_sign,
                    start_heading=round(before.angle_deg, 3),
                    end_heading=round(after.angle_deg, 3),
                    measured_degrees=round(magnitude, 3),
                    requested_seconds=round(
                        pulse_result.requested_seconds,
                        5,
                    ),
                    clamped_seconds=round(
                        pulse_result.clamped_seconds,
                        5,
                    ),
                    pulse_seconds=round(pulse_result.held_seconds, 5),
                    elapsed_seconds=round(
                        pulse_result.elapsed_seconds,
                        5,
                    ),
                    idle_seconds=(
                        round(pulse_result.idle_seconds, 5)
                        if pulse_result.idle_seconds is not None
                        else None
                    ),
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

    @classmethod
    def _single_pulse_delta_is_implausible(
        cls,
        signed_change: float,
        motion_uncertainty: float,
    ) -> bool:
        """Identify non-finite or visually aliased single-pulse deltas."""
        if not isfinite(signed_change) or not isfinite(motion_uncertainty):
            return True
        if motion_uncertainty < 0.0:
            return True
        return (
            abs(signed_change) - motion_uncertainty
            > cls.MAXIMUM_PLAUSIBLE_SINGLE_PULSE_DELTA_DEGREES
        )

    @classmethod
    def _raise_for_implausible_probe_delta(
        cls,
        directed_motion: float,
        motion_uncertainty: float,
        *,
        context: str,
    ) -> None:
        """Reject a cross-endpoint visual alias before fitting probe data."""
        if (
            not isfinite(directed_motion)
            or not isfinite(motion_uncertainty)
            or motion_uncertainty < 0.0
        ):
            raise _ProbeHeadingDeltaError(
                f"{context}: the endpoint delta or its uncertainty was invalid."
            )
        if not cls._single_pulse_delta_is_implausible(
            directed_motion,
            motion_uncertainty,
        ):
            return
        message = f"{context}: two strict endpoint headings implied "
        message += f"{directed_motion:+.1f}° with "
        message += f"{motion_uncertainty:.1f}° uncertainty. This exceeds the "
        message += f"{cls.MAXIMUM_PLAUSIBLE_SINGLE_PULSE_DELTA_DEGREES:.0f}° "
        message += "single-pulse bound and is probably the minimap arrow's "
        message += "opposite-orientation visual alias."
        raise _ProbeHeadingDeltaError(message)

    def _advance_probe_start_failure_count(
        self,
        consecutive_failures: int,
        *,
        phase: str,
        probe_description: str,
    ) -> int:
        """Abort a phase that can no longer reach its pre-movement heading gate."""
        consecutive_failures += 1
        if consecutive_failures < self.MAXIMUM_CONSECUTIVE_PROBE_START_FAILURES:
            return consecutive_failures

        debug_folder = self.detector.save_debug("repeated_probe_start_heading_failure")
        debug_text = (
            f" Debug saved to: {debug_folder}" if debug_folder is not None else ""
        )
        raise _PersistentHeadingFailure(
            "The minimap heading could not be read before movement on "
            f"{consecutive_failures} consecutive {phase} attempts "
            f"(last probe: {probe_description}). The remaining phase was stopped "
            f"before issuing more blind pulses.{debug_text}"
        )

    def _prepare_neutral_transition(self, label: str) -> None:
        self.controller.stop()
        wait_seconds = self.controller.neutral_after_seconds
        if wait_seconds is None:
            self.status(
                f"{label}: turn direction state remained persistent through "
                f"the validated {self.controller.turn_memory_policy.observed_horizon_seconds:.2f}s "
                "observation horizon. Preserving direction history and using "
                "state-aware timing without an invented neutral wait."
            )
            return
        self.status(
            f"{label}: releasing movement and waiting "
            f"{wait_seconds:.2f}s for neutral turn state."
        )
        self._wait_or_cancel(wait_seconds)
        self.controller.reset_turn_history()

    @classmethod
    def _build_idle_probe_schedule(
        cls,
        acquisition_floor_seconds: float,
    ) -> tuple[float, ...]:
        """Return the two delays needed to classify persistent vs decaying state."""
        floor = max(0.0, float(acquisition_floor_seconds))
        short = max(
            cls.NEUTRAL_PROBE_MINIMUM_SHORT_IDLE_SECONDS,
            floor,
        )
        if short >= cls.NEUTRAL_PROBE_HORIZON_SECONDS:
            raise ValueError("turn-memory acquisition floor exceeds probe horizon")
        return (
            round(short, 3),
            cls.NEUTRAL_PROBE_HORIZON_SECONDS,
        )

    @classmethod
    def _build_turn_memory_probe_plan(
        cls,
        schedule: tuple[float, ...],
        *,
        repeats: int | None = None,
    ) -> tuple[tuple[TurnDirection, TurnTransition, float, int], ...]:
        """
        Counterbalance direction, delay and transition for replicated probes.

        Every delay receives both SAME and REVERSAL evidence in both
        directions. Reversing each axis on alternate repeats prevents physical
        path order from being perfectly correlated with one state.
        """
        if not schedule:
            raise ValueError("turn-memory probe schedule cannot be empty")
        repeat_count = cls.NEUTRAL_PROBE_SCAN_REPEATS if repeats is None else repeats
        if repeat_count < 1:
            raise ValueError("turn-memory probe repeats must be positive")

        plan: list[tuple[TurnDirection, TurnTransition, float, int]] = []
        base_directions = tuple(TurnDirection)
        base_transitions = (
            TurnTransition.REVERSAL,
            TurnTransition.SAME_DIRECTION,
        )
        for repeat_index in range(1, repeat_count + 1):
            directions = (
                base_directions
                if repeat_index % 2
                else tuple(reversed(base_directions))
            )
            delays = schedule if repeat_index % 2 else tuple(reversed(schedule))
            for delay_index, requested_idle_seconds in enumerate(delays):
                transitions = (
                    base_transitions
                    if (repeat_index + delay_index) % 2
                    else tuple(reversed(base_transitions))
                )
                for direction in directions:
                    for transition in transitions:
                        plan.append(
                            (
                                direction,
                                transition,
                                requested_idle_seconds,
                                repeat_index,
                            )
                        )
        return tuple(plan)

    def _calibrate_neutral_timeout(
        self,
        *,
        left_seconds_90: float,
        right_seconds_90: float,
        left_sign: int,
        right_sign: int,
        restore_heading: float,
    ) -> tuple[NeutralTimeoutFit, list[dict[str, object]]]:
        """
        Measure the direction-memory timeout and validate it before use.

        The scan uses only strict non-interactive heading reads. A visual
        confirmation dialog could remain open past the requested idle delay and
        would invalidate the state being measured.
        """
        seconds_90 = {
            TurnDirection.LEFT: float(left_seconds_90),
            TurnDirection.RIGHT: float(right_seconds_90),
        }
        signs = {
            TurnDirection.LEFT: int(left_sign),
            TurnDirection.RIGHT: int(right_sign),
        }
        scan_samples: list[NeutralTimeoutSample] = []
        validation_samples: list[NeutralTimeoutSample] = []
        records: list[dict[str, object]] = []
        failure: Exception | None = None
        phase_rotation_model: StateAwareRotationModel | None = None
        consecutive_start_heading_failures = 0

        try:

            def collect_scan_plan(
                plan: tuple[
                    tuple[TurnDirection, TurnTransition, float, int],
                    ...,
                ],
                *,
                phase: str,
            ) -> None:
                nonlocal consecutive_start_heading_failures
                for (
                    direction,
                    conditioning_transition,
                    requested_idle_seconds,
                    repeat_index,
                ) in plan:
                    conditioning_direction = (
                        direction
                        if conditioning_transition is TurnTransition.SAME_DIRECTION
                        else direction.opposite
                    )
                    for attempt in range(1, self.PROBE_HEADING_ATTEMPTS + 1):
                        try:
                            sample, record = self._measure_neutral_timeout_probe(
                                direction=direction,
                                conditioning_transition=conditioning_transition,
                                requested_idle_seconds=requested_idle_seconds,
                                target_pulse_seconds=min(
                                    0.18,
                                    max(0.055, seconds_90[direction] * 0.28),
                                ),
                                conditioning_pulse_seconds=min(
                                    0.14,
                                    max(
                                        0.040,
                                        seconds_90[conditioning_direction] * 0.20,
                                    ),
                                ),
                                heading_sign=signs[direction],
                                conditioning_heading_sign=signs[conditioning_direction],
                                phase=phase,
                            )
                        except _ProbeMeasurementError as error:
                            self.controller.stop()
                            rejection_reason = str(error)
                            debug_text = ""
                            if attempt == self.PROBE_HEADING_ATTEMPTS and isinstance(
                                error, _ProbeHeadingDeltaError
                            ):
                                debug_folder = self.detector.save_debug(
                                    f"{direction.value}_neutral_timeout_{phase}_implausible_delta"
                                )
                                if debug_folder is not None:
                                    debug_text = f" Debug saved to: {debug_folder}"
                                    rejection_reason += debug_text
                            records.append(
                                {
                                    "phase": phase,
                                    "direction": direction.value,
                                    "conditioning_transition": (
                                        conditioning_transition.value
                                    ),
                                    "requested_idle_seconds": (requested_idle_seconds),
                                    "repeat": repeat_index,
                                    "attempt": attempt,
                                    "skipped": True,
                                    "reason": rejection_reason,
                                }
                            )
                            if isinstance(error, _ProbeStartHeadingError):
                                consecutive_start_heading_failures = (
                                    self._advance_probe_start_failure_count(
                                        consecutive_start_heading_failures,
                                        phase=f"Phase 2 {phase}",
                                        probe_description=(
                                            f"{direction.value} "
                                            f"{conditioning_transition.value} "
                                            f"at {requested_idle_seconds:.3f}s"
                                        ),
                                    )
                                )
                            else:
                                consecutive_start_heading_failures = 0
                            if attempt < self.PROBE_HEADING_ATTEMPTS:
                                self.status(
                                    f"Rejected {direction.value} "
                                    f"{conditioning_transition.value} {phase} "
                                    f"probe measurement: {error} "
                                    "Reconditioning and retrying."
                                )
                                continue
                            self.status(
                                f"Could not complete {direction.value} "
                                f"{conditioning_transition.value} {phase} probe "
                                f"at {requested_idle_seconds:.3f}s after "
                                f"{self.PROBE_HEADING_ATTEMPTS} rejected heading "
                                f"measurements.{debug_text}"
                            )
                            raise RuntimeError(
                                "Turn-memory classification could not complete "
                                "the required replicated cell "
                                f"({direction.value} "
                                f"{conditioning_transition.value}, "
                                f"{requested_idle_seconds:.3f}s, repeat "
                                f"{repeat_index}) after "
                                f"{self.PROBE_HEADING_ATTEMPTS} attempts. "
                                f"Last rejection: {rejection_reason}"
                            ) from error
                        else:
                            consecutive_start_heading_failures = 0
                            record["repeat"] = repeat_index
                            scan_samples.append(sample)
                            records.append(record)
                            break

            def fit_scan() -> NeutralTimeoutFit:
                try:
                    return fit_neutral_timeout(
                        scan_samples,
                        safety_margin_seconds=(
                            self.NEUTRAL_PROBE_SAFETY_MARGIN_SECONDS
                        ),
                        maximum_idle_seconds=(self.NEUTRAL_PROBE_MAXIMUM_SECONDS),
                        maximum_sample_uncertainty_degrees=(
                            self.MAXIMUM_TURN_MEASUREMENT_UNCERTAINTY_DEGREES
                        ),
                    )
                except ValueError as error:
                    raise RuntimeError(
                        "Turn-state behavior could not be established from "
                        f"the replicated bidirectional probes: {error}. The "
                        "previous calibration was preserved."
                    ) from error

            # Five raw-geometry frames take roughly one tenth of a second at
            # the verified capture rate. Start with one reachable short delay
            # and the six-second horizon, each fully replicated.
            acquisition_floor = self.PROBE_FRAME_CAPTURE_LEAD_SECONDS + 0.08
            schedule = self._build_idle_probe_schedule(acquisition_floor)
            collect_scan_plan(
                self._build_turn_memory_probe_plan(schedule),
                phase="classification",
            )
            fit = fit_scan()

            if fit.turn_memory_policy.mode is TurnMemoryMode.DECAYS_TO_NEUTRAL:
                # Only a genuinely converged long-horizon experiment needs the
                # intermediate samples used to shape the decay curve.
                refinement_delays = tuple(
                    delay
                    for delay in self.NEUTRAL_PROBE_REFINEMENT_IDLE_SECONDS
                    if schedule[0] < delay < schedule[-1]
                )
                refinement_plan = tuple(
                    (direction, TurnTransition.REVERSAL, delay, repeat)
                    for repeat in range(1, self.NEUTRAL_PROBE_SCAN_REPEATS + 1)
                    for delay in (
                        refinement_delays
                        if repeat % 2
                        else tuple(reversed(refinement_delays))
                    )
                    for direction in (
                        tuple(TurnDirection)
                        if repeat % 2
                        else tuple(reversed(tuple(TurnDirection)))
                    )
                )
                self.status(
                    "The replicated horizon responses converged. Measuring "
                    "only the intermediate reversal delays needed for a decay "
                    "curve."
                )
                collect_scan_plan(refinement_plan, phase="decay_scan")
                fit = fit_scan()

            mode = fit.turn_memory_policy.mode
            if mode is TurnMemoryMode.DECAYS_TO_NEUTRAL:
                neutral_after_seconds = fit.neutral_after_seconds
                assert neutral_after_seconds is not None
                self.status(
                    "Turn memory converged to neutral by "
                    f"{neutral_after_seconds:.3f}s. Validating SAME and "
                    "REVERSAL tails in both directions."
                )
            else:
                neutral_after_seconds = None
                self.status(
                    "Turn memory remained direction-dependent through the "
                    f"{fit.turn_memory_policy.observed_horizon_seconds:.3f}s "
                    "observation horizon. Validating persistent mode."
                )
            validation_idle = (
                neutral_after_seconds
                if neutral_after_seconds is not None
                else fit.turn_memory_policy.observed_horizon_seconds
            )
            for direction in TurnDirection:
                for conditioning_transition in (
                    TurnTransition.SAME_DIRECTION,
                    TurnTransition.REVERSAL,
                ):
                    conditioning_direction = (
                        direction
                        if conditioning_transition is TurnTransition.SAME_DIRECTION
                        else direction.opposite
                    )
                    for repeat_index in range(
                        1,
                        self.NEUTRAL_PROBE_VALIDATION_REPEATS + 1,
                    ):
                        for attempt in range(1, self.PROBE_HEADING_ATTEMPTS + 1):
                            try:
                                sample, record = self._measure_neutral_timeout_probe(
                                    direction=direction,
                                    conditioning_transition=conditioning_transition,
                                    requested_idle_seconds=validation_idle,
                                    target_pulse_seconds=min(
                                        0.18,
                                        max(0.055, seconds_90[direction] * 0.28),
                                    ),
                                    conditioning_pulse_seconds=min(
                                        0.14,
                                        max(
                                            0.040,
                                            seconds_90[conditioning_direction] * 0.20,
                                        ),
                                    ),
                                    heading_sign=signs[direction],
                                    conditioning_heading_sign=signs[
                                        conditioning_direction
                                    ],
                                    phase="validation",
                                )
                            except _ProbeMeasurementError as error:
                                self.controller.stop()
                                rejection_reason = str(error)
                                if (
                                    attempt == self.PROBE_HEADING_ATTEMPTS
                                    and isinstance(error, _ProbeHeadingDeltaError)
                                ):
                                    debug_folder = self.detector.save_debug(
                                        f"{direction.value}_neutral_timeout_validation_implausible_delta"
                                    )
                                    if debug_folder is not None:
                                        rejection_reason += (
                                            f" Debug saved to: {debug_folder}"
                                        )
                                records.append(
                                    {
                                        "phase": "validation",
                                        "direction": direction.value,
                                        "conditioning_transition": conditioning_transition.value,
                                        "requested_idle_seconds": validation_idle,
                                        "repeat": repeat_index,
                                        "attempt": attempt,
                                        "skipped": True,
                                        "reason": rejection_reason,
                                    }
                                )
                                if isinstance(error, _ProbeStartHeadingError):
                                    consecutive_start_heading_failures = (
                                        self._advance_probe_start_failure_count(
                                            consecutive_start_heading_failures,
                                            phase="Phase 2 validation",
                                            probe_description=(
                                                f"{direction.value} "
                                                f"{conditioning_transition.value} "
                                                f"repeat {repeat_index}"
                                            ),
                                        )
                                    )
                                else:
                                    consecutive_start_heading_failures = 0
                                if attempt < self.PROBE_HEADING_ATTEMPTS:
                                    retry_message = f"Rejected {direction.value} "
                                    retry_message += f"{conditioning_transition.value} "
                                    retry_message += (
                                        f"validation {repeat_index} heading "
                                    )
                                    retry_message += f"measurement: {error} "
                                    retry_message += "Reconditioning and retrying."
                                    self.status(retry_message)
                                    continue
                                failure_message = f"{direction.value.title()} "
                                failure_message += f"{conditioning_transition.value} "
                                failure_message += (
                                    f"validation {repeat_index} could not produce "
                                )
                                failure_message += (
                                    "a trustworthy heading measurement after "
                                )
                                failure_message += (
                                    f"{self.PROBE_HEADING_ATTEMPTS} attempts. "
                                )
                                failure_message += f"Last rejection: {rejection_reason}"
                                raise RuntimeError(failure_message) from error
                            else:
                                consecutive_start_heading_failures = 0
                                validation_samples.append(sample)
                                records.append(record)
                                break

            try:
                validate_neutral_timeout(fit, validation_samples)
            except ValueError as error:
                raise RuntimeError(
                    "Turn-memory model failed repeated SAME/REVERSAL tail "
                    f"validation: {error}. The previous calibration was preserved."
                ) from error

            neutral_rotation_samples = [
                sample.as_neutral_rotation_sample()
                for sample in (*scan_samples, *validation_samples)
                if fit.turn_memory_policy.mode is TurnMemoryMode.DECAYS_TO_NEUTRAL
                and neutral_after_seconds is not None
                and sample.observed_idle_seconds >= neutral_after_seconds - 0.01
            ]
            for direction in TurnDirection:
                if (
                    fit.turn_memory_policy.mode is TurnMemoryMode.DECAYS_TO_NEUTRAL
                    and sum(
                        sample.direction is direction
                        for sample in neutral_rotation_samples
                    )
                    < 3
                ):
                    raise RuntimeError(
                        f"{direction.value.title()} calibration did not retain "
                        "three physically neutral timing samples. The previous "
                        "calibration was preserved."
                    )
            state_rotation_samples = [
                sample.as_state_rotation_sample()
                for sample in (*scan_samples, *validation_samples)
                if (
                    fit.turn_memory_policy.mode is TurnMemoryMode.PERSISTENT_OBSERVED
                    or sample.observed_idle_seconds
                    <= fit.fit_for(sample.direction).last_stateful_seconds + 0.05
                )
            ]
            self._rotation_samples.extend(
                (*state_rotation_samples, *neutral_rotation_samples)
            )
            phase_rotation_model = fit_rotation_model(
                self._rotation_samples,
                fallback_left_seconds_90=left_seconds_90,
                fallback_right_seconds_90=right_seconds_90,
                idle_response_curves=fit.idle_response_curves,
                turn_memory_policy=fit.turn_memory_policy,
            )

            self.controller.set_turn_memory_policy(
                fit.turn_memory_policy,
                # The last successful validation pulse established real game
                # state. Installing the fitted policy must not erase that
                # physical direction history and pretend the next pulse is
                # neutral.
                reset_history=False,
            )
            self.status(
                "Validated turn-memory model: "
                f"{fit.turn_memory_policy.mode.value} through "
                f"{fit.turn_memory_policy.observed_horizon_seconds:.3f}s. "
                "Its measured idle-response curves will be used immediately "
                "without runtime waiting."
            )
            return fit, records
        except Exception as error:
            failure = error
            raise
        finally:
            release_succeeded = True
            try:
                self.controller.stop()
            except Exception as release_error:
                release_succeeded = False
                if failure is None:
                    raise
                failure.add_note(
                    "Neutral-timeout cleanup also failed to release all "
                    f"movement keys: {release_error}"
                )
                self._status_without_masking(
                    "Neutral-timeout calibration also failed to release all "
                    f"movement keys during recovery: {release_error}",
                    failure,
                )

            if failure is not None:
                if self.cancellation.cancelled:
                    self._status_without_masking(
                        "Neutral-timeout calibration was cancelled; no recovery "
                        "turn was issued after the stop request.",
                        failure,
                    )
                elif not release_succeeded:
                    self._status_without_masking(
                        "Original-heading recovery was skipped because movement "
                        "keys could not be confirmed released.",
                        failure,
                    )
                elif isinstance(failure, _PersistentHeadingFailure):
                    self._status_without_masking(
                        "Neutral-timeout calibration stopped after repeated "
                        "pre-movement heading failures. Movement keys are "
                        "released; no blind recovery turn will be attempted.",
                        failure,
                    )
                else:
                    self._status_without_masking(
                        "Neutral-timeout calibration failed; original-heading "
                        "recovery is being handled once by the calibration "
                        "failure boundary.",
                        failure,
                    )
            elif self.cancellation.cancelled:
                self.cancellation.raise_if_cancelled()
            elif release_succeeded:
                self.status(
                    "Restoring the heading recorded at the start of calibration "
                    "after neutral-timeout probes."
                )
                self._turn_to_absolute_heading(
                    target_heading=restore_heading,
                    left_seconds_90=left_seconds_90,
                    right_seconds_90=right_seconds_90,
                    left_sign=left_sign,
                    right_sign=right_sign,
                    label="Post-neutral-timeout heading restore",
                    rotation_model=phase_rotation_model,
                )

    def _measure_neutral_timeout_probe(
        self,
        *,
        direction: TurnDirection,
        conditioning_transition: TurnTransition = TurnTransition.REVERSAL,
        requested_idle_seconds: float,
        target_pulse_seconds: float,
        conditioning_pulse_seconds: float,
        heading_sign: int,
        conditioning_heading_sign: int,
        phase: str,
    ) -> tuple[NeutralTimeoutSample, dict[str, object]]:
        conditioning_direction = (
            direction
            if conditioning_transition is TurnTransition.SAME_DIRECTION
            else direction.opposite
        )
        self.status(
            f"Turn-memory {phase}: condition {conditioning_direction.value} "
            f"with two pulses ({conditioning_transition.value}), then probe "
            f"{direction.value} after "
            f"{requested_idle_seconds:.3f}s idle."
        )
        try:
            conditioning_start = self._noninteractive_heading(
                f"{direction.value} neutral-timeout {phase} conditioning start",
                timeout=1.25,
                samples=7,
            )
        except _ProbeHeadingConsensusError as error:
            raise _ProbeStartHeadingError(str(error)) from error
        conditioning_prime, conditioning_pulse = self._establish_turn_state(
            conditioning_direction.value,
            conditioning_pulse_seconds,
        )
        current_idle_seconds = self.controller.turn_idle_seconds
        if current_idle_seconds is None:
            raise _ProbeTimingError(
                "Turn-state history disappeared during neutral-timeout probing."
            )
        capture_target_idle = max(
            0.0,
            requested_idle_seconds - self.PROBE_FRAME_CAPTURE_LEAD_SECONDS,
        )
        wait_before_capture = capture_target_idle - current_idle_seconds
        if wait_before_capture > 0.0:
            self._wait_or_cancel(wait_before_capture)
        before_frames = self._collect_probe_frames(
            count=self.PROBE_HEADING_FRAME_COUNT,
            not_before=monotonic(),
            timeout=0.80,
        )

        current_idle_seconds = self.controller.turn_idle_seconds
        if current_idle_seconds is None:
            raise _ProbeTimingError(
                "Turn-state history disappeared during neutral-timeout probing."
            )
        remaining_idle_seconds = requested_idle_seconds - current_idle_seconds
        if remaining_idle_seconds > 0.0:
            self._wait_or_cancel(remaining_idle_seconds)
        elif -remaining_idle_seconds > self.NEUTRAL_PROBE_MAXIMUM_OVERSHOOT_SECONDS:
            raise _ProbeTimingError(
                "The pre-pulse frame batch exceeded the requested "
                "neutral-timeout probe delay by "
                f"{-remaining_idle_seconds:.3f}s. The scan was discarded "
                "instead of using a mistimed state."
            )

        pulse = self._turn_burst(
            direction.value,
            target_pulse_seconds,
        )
        observed_idle_seconds = pulse.idle_seconds
        if observed_idle_seconds is None:
            raise _ProbeTimingError(
                "Neutral-timeout probe did not report its observed idle delay."
            )
        idle_overshoot = observed_idle_seconds - requested_idle_seconds
        if (
            idle_overshoot < -0.01
            or idle_overshoot > self.NEUTRAL_PROBE_MAXIMUM_OVERSHOOT_SECONDS
        ):
            raise _ProbeTimingError(
                f"Neutral-timeout {phase} pulse missed its requested idle "
                f"delay by {idle_overshoot:+.3f}s. The measurement was discarded."
            )

        before = self._noninteractive_heading_from_frames(
            before_frames,
            f"{direction.value} neutral-timeout {phase} start",
        )
        conditioning_motion = (
            observed_heading_delta(before, conditioning_start)
            * conditioning_heading_sign
        )
        conditioning_uncertainty = self._reading_uncertainty(
            conditioning_start
        ) + self._reading_uncertainty(before)
        conditioning_response, conditioning_response_resolved = (
            self._validate_conditioning_response(
                conditioning_direction,
                conditioning_motion,
                conditioning_uncertainty,
            )
        )

        self._wait_or_cancel(max(self.settle_seconds, 0.48))
        after = self._noninteractive_heading(
            f"{direction.value} neutral-timeout {phase} result",
            timeout=1.75,
        )
        signed_motion = observed_heading_delta(after, before)
        directed_motion = signed_motion * heading_sign
        uncertainty = self._reading_uncertainty(before) + self._reading_uncertainty(
            after
        )
        confidence = min(before.confidence, after.confidence)
        self._raise_for_implausible_probe_delta(
            directed_motion,
            uncertainty,
            context=f"{direction.value.title()} neutral-timeout {phase} probe",
        )
        try:
            measured_response, response_resolved = self._normalize_turn_probe_response(
                directed_motion,
                uncertainty,
                allow_bounded_opposite=(pulse.transition is TurnTransition.REVERSAL),
            )
        except ValueError as error:
            raise _ProbeResponseError(
                f"{direction.value.title()} neutral-timeout {phase} probe "
                f"measured {directed_motion:+.1f}° with {uncertainty:.1f}° "
                f"uncertainty: {error}. The measurement was discarded."
            ) from error

        sample = NeutralTimeoutSample(
            direction=direction,
            requested_idle_seconds=float(requested_idle_seconds),
            observed_idle_seconds=float(observed_idle_seconds),
            measured_degrees=measured_response,
            uncertainty_degrees=float(uncertainty),
            confidence=float(confidence),
            conditioning_transition=conditioning_transition,
            requested_seconds=pulse.requested_seconds,
            clamped_seconds=pulse.clamped_seconds,
            held_seconds=pulse.held_seconds,
        )
        record: dict[str, object] = {
            "phase": phase,
            "direction": direction.value,
            "conditioning_direction": conditioning_direction.value,
            "conditioning_transition": conditioning_transition.value,
            "conditioning_prime_transition": conditioning_prime.transition.value,
            "conditioning_prime_requested_seconds": round(
                conditioning_prime.requested_seconds,
                5,
            ),
            "conditioning_prime_clamped_seconds": round(
                conditioning_prime.clamped_seconds,
                5,
            ),
            "conditioning_prime_held_seconds": round(
                conditioning_prime.held_seconds,
                5,
            ),
            "conditioning_prime_elapsed_seconds": round(
                conditioning_prime.elapsed_seconds,
                5,
            ),
            "conditioning_requested_seconds": round(
                conditioning_pulse.requested_seconds,
                5,
            ),
            "conditioning_clamped_seconds": round(
                conditioning_pulse.clamped_seconds,
                5,
            ),
            "conditioning_held_seconds": round(
                conditioning_pulse.held_seconds,
                5,
            ),
            "conditioning_elapsed_seconds": round(
                conditioning_pulse.elapsed_seconds,
                5,
            ),
            "requested_idle_seconds": round(requested_idle_seconds, 6),
            "observed_idle_seconds": round(observed_idle_seconds, 6),
            "conditioning_start_heading": round(
                conditioning_start.angle_deg,
                3,
            ),
            "conditioning_end_heading": round(before.angle_deg, 3),
            "conditioning_start_motion_angle": self._optional_motion_angle(
                conditioning_start
            ),
            "conditioning_end_motion_angle": self._optional_motion_angle(before),
            "conditioning_measured_degrees": round(
                conditioning_motion,
                3,
            ),
            "conditioning_normalized_degrees": round(
                conditioning_response,
                3,
            ),
            "conditioning_response_resolved": conditioning_response_resolved,
            "conditioning_uncertainty_degrees": round(
                conditioning_uncertainty,
                3,
            ),
            "start_heading": round(before.angle_deg, 3),
            "end_heading": round(after.angle_deg, 3),
            "start_motion_angle": self._optional_motion_angle(before),
            "end_motion_angle": self._optional_motion_angle(after),
            "requested_seconds": round(pulse.requested_seconds, 5),
            "clamped_seconds": round(pulse.clamped_seconds, 5),
            "held_seconds": round(pulse.held_seconds, 5),
            "elapsed_seconds": round(pulse.elapsed_seconds, 5),
            "controller_transition": pulse.transition.value,
            "raw_measured_degrees": round(directed_motion, 3),
            "measured_degrees": round(measured_response, 3),
            "response_resolved": response_resolved,
            "uncertainty_degrees": round(uncertainty, 3),
            "confidence": round(confidence, 3),
        }
        if response_resolved:
            self.status(
                f"Neutral-timeout {phase} {direction.value}: "
                f"idle {observed_idle_seconds:.3f}s -> "
                f"{measured_response:.1f}° "
                f"(uncertainty {uncertainty:.1f}°)."
            )
        else:
            self.status(
                f"Neutral-timeout {phase} {direction.value}: "
                f"idle {observed_idle_seconds:.3f}s produced no resolvable "
                f"turn (raw {directed_motion:+.1f}°, uncertainty "
                f"{uncertainty:.1f}°); recording a conservative 0° "
                "suppressed response."
            )
        return sample, record

    @classmethod
    def _normalize_turn_probe_response(
        cls,
        directed_motion: float,
        uncertainty: float,
        *,
        allow_bounded_opposite: bool = False,
    ) -> tuple[float, bool]:
        """Retain a suppressed turn as data without accepting bad motion."""
        directed_motion = float(directed_motion)
        uncertainty = float(uncertainty)
        if (
            not isfinite(directed_motion)
            or not isfinite(uncertainty)
            or uncertainty < 0.0
        ):
            raise ValueError("the measured response was not finite")
        if uncertainty > cls.MAXIMUM_TURN_MEASUREMENT_UNCERTAINTY_DEGREES:
            raise ValueError(
                "combined endpoint uncertainty exceeded the turn-probe limit"
            )

        resolution_floor = max(2.0, uncertainty)
        if directed_motion < -resolution_floor:
            if (
                allow_bounded_opposite
                and abs(directed_motion) <= MAXIMUM_TOLERATED_REVERSAL_BACKLASH_DEGREES
            ):
                return 0.0, False
            raise ValueError(
                "the pulse moved reliably opposite its requested direction"
            )
        if directed_motion > 70.0:
            raise ValueError("the pulse exceeded the safe fitting range")
        if directed_motion <= resolution_floor:
            return 0.0, False
        return directed_motion, True

    def _validate_conditioning_response(
        self,
        direction: TurnDirection,
        directed_motion: float,
        uncertainty: float,
    ) -> tuple[float, bool]:
        """Accept suppressed conditioning while rejecting unsafe measurements."""
        self._raise_for_implausible_probe_delta(
            directed_motion,
            uncertainty,
            context=f"{direction.value.title()} conditioning sequence",
        )
        try:
            response, resolved = self._normalize_turn_probe_response(
                directed_motion,
                uncertainty,
                allow_bounded_opposite=True,
            )
        except ValueError as error:
            raise _ProbeResponseError(
                f"{direction.value.title()} conditioning sequence moved "
                f"{directed_motion:+.1f}° with {uncertainty:.1f}° uncertainty: "
                f"{error}. The requested turn state was discarded."
            ) from error
        if not resolved:
            self.status(
                f"{direction.value.title()} conditioning sequence produced no "
                "resolvable rotation, but both key pulses completed and the "
                "final pulse was same-direction; retaining the commanded turn "
                "state."
            )
        return response, resolved

    def _measure_transition_matrix(
        self,
        *,
        left_seconds_90: float,
        right_seconds_90: float,
        left_sign: int,
        right_sign: int,
    ) -> list[dict[str, object]]:
        """Collect varied-duration samples for both repeated and reversed turns."""
        seconds_90 = {
            TurnDirection.LEFT: float(left_seconds_90),
            TurnDirection.RIGHT: float(right_seconds_90),
        }
        signs = {
            TurnDirection.LEFT: int(left_sign),
            TurnDirection.RIGHT: int(right_sign),
        }
        records: list[dict[str, object]] = []
        duration_factors = (0.20, 0.32, 0.44)
        consecutive_start_heading_failures = 0

        plan: list[tuple[TurnDirection, TurnTransition, float, int]] = []
        base_directions = tuple(TurnDirection)
        base_transitions = (
            TurnTransition.SAME_DIRECTION,
            TurnTransition.REVERSAL,
        )
        for repeat_index in range(1, self.TRANSITION_MATRIX_REPEATS + 1):
            directions = (
                base_directions
                if repeat_index % 2
                else tuple(reversed(base_directions))
            )
            transitions = (
                base_transitions
                if repeat_index % 2
                else tuple(reversed(base_transitions))
            )
            factors = (
                duration_factors
                if repeat_index % 2
                else tuple(reversed(duration_factors))
            )
            for factor in factors:
                for direction in directions:
                    for expected_transition in transitions:
                        plan.append(
                            (
                                direction,
                                expected_transition,
                                factor,
                                repeat_index,
                            )
                        )

        for direction, expected_transition, factor, repeat_index in plan:
            for attempt in range(1, self.PROBE_HEADING_ATTEMPTS + 1):
                try:
                    sample, record = self._measure_transition_probe(
                        direction=direction,
                        expected_transition=expected_transition,
                        factor=factor,
                        seconds_90=seconds_90,
                        signs=signs,
                    )
                except _ProbeMeasurementError as error:
                    self.controller.stop()
                    records.append(
                        {
                            "phase": "transition_matrix",
                            "direction": direction.value,
                            "transition": expected_transition.value,
                            "factor_of_coarse_90": factor,
                            "repeat": repeat_index,
                            "attempt": attempt,
                            "skipped": True,
                            "reason": str(error),
                        }
                    )
                    if isinstance(error, _ProbeStartHeadingError):
                        consecutive_start_heading_failures = (
                            self._advance_probe_start_failure_count(
                                consecutive_start_heading_failures,
                                phase="Phase 3",
                                probe_description=(
                                    f"{direction.value} "
                                    f"{expected_transition.value} "
                                    f"factor {factor:.2f}"
                                ),
                            )
                        )
                    else:
                        consecutive_start_heading_failures = 0
                    if attempt < self.PROBE_HEADING_ATTEMPTS:
                        self.status(
                            f"Rejected {direction.value} "
                            f"{expected_transition.value} Phase 3 "
                            f"measurement: {error} Reconditioning and retrying."
                        )
                        continue
                    raise RuntimeError(
                        f"{direction.value.title()} "
                        f"{expected_transition.value} Phase 3 sample "
                        f"at factor {factor:.2f} could not produce a "
                        "trustworthy measurement after "
                        f"{self.PROBE_HEADING_ATTEMPTS} attempts. "
                        f"Last rejection: {error}"
                    ) from error
                else:
                    consecutive_start_heading_failures = 0
                    record["repeat"] = repeat_index
                    self._rotation_samples.append(sample)
                    records.append(record)
                    break

        return records

    def _measure_transition_probe(
        self,
        *,
        direction: TurnDirection,
        expected_transition: TurnTransition,
        factor: float,
        seconds_90: dict[TurnDirection, float],
        signs: dict[TurnDirection, int],
    ) -> tuple[RotationSample, dict[str, object]]:
        """Measure one Phase 3 cell behind the shared typed retry boundary."""
        conditioning_direction = (
            direction
            if expected_transition is TurnTransition.SAME_DIRECTION
            else direction.opposite
        )
        conditioning_seconds = min(
            0.12,
            max(0.030, seconds_90[conditioning_direction] * 0.18),
        )
        self.status(
            f"Preparing {direction.value} {expected_transition.value} sample "
            f"with a short two-pulse {conditioning_direction.value} "
            "conditioning sequence."
        )
        try:
            conditioning_start = self._noninteractive_heading(
                f"{direction.value} {expected_transition.value} conditioning start",
                timeout=1.25,
                samples=7,
            )
        except _ProbeHeadingConsensusError as error:
            raise _ProbeStartHeadingError(str(error)) from error

        conditioning_prime, conditioning_pulse = self._establish_turn_state(
            conditioning_direction.value,
            conditioning_seconds,
        )
        before_frames = self._collect_probe_frames(
            count=self.PROBE_HEADING_FRAME_COUNT,
            not_before=monotonic(),
            timeout=0.80,
        )
        observed_idle = self.controller.turn_idle_seconds
        neutral_after_seconds = self.controller.neutral_after_seconds
        if observed_idle is None:
            raise _ProbeTimingError(
                "Turn-state history disappeared while capturing "
                f"{direction.value} {expected_transition.value} pre-pulse frames."
            )
        if (
            neutral_after_seconds is not None
            and observed_idle >= neutral_after_seconds - 0.05
        ):
            raise _ProbeTimingError(
                "The fitted decaying turn-state window expired while capturing "
                f"{direction.value} {expected_transition.value} pre-pulse frames."
            )

        requested_seconds = min(
            self.maximum_pulse_seconds,
            max(
                self.minimum_pulse_seconds,
                seconds_90[direction] * factor,
            ),
        )
        pulse = self._turn_burst(direction.value, requested_seconds)
        if pulse.transition is not expected_transition:
            raise _ProbeTimingError(
                "Turn transition changed before its measurement "
                f"(expected {expected_transition.value}, observed "
                f"{pulse.transition.value}). The sample was discarded instead "
                "of fitting the wrong state."
            )

        before = self._noninteractive_heading_from_frames(
            before_frames,
            f"{direction.value} {expected_transition.value} trial start",
        )
        conditioning_motion = (
            observed_heading_delta(before, conditioning_start)
            * signs[conditioning_direction]
        )
        conditioning_uncertainty = self._reading_uncertainty(
            conditioning_start
        ) + self._reading_uncertainty(before)
        conditioning_response, conditioning_response_resolved = (
            self._validate_conditioning_response(
                conditioning_direction,
                conditioning_motion,
                conditioning_uncertainty,
            )
        )
        self._wait_or_cancel(max(self.settle_seconds, 0.48))
        after = self._noninteractive_heading(
            f"{direction.value} {expected_transition.value} trial result",
            timeout=1.75,
        )
        signed_motion = observed_heading_delta(after, before)
        directed_motion = signed_motion * signs[direction]
        confidence = min(before.confidence, after.confidence)
        motion_uncertainty = self._reading_uncertainty(
            before
        ) + self._reading_uncertainty(after)
        self._raise_for_implausible_probe_delta(
            directed_motion,
            motion_uncertainty,
            context=(f"{direction.value.title()} {expected_transition.value} trial"),
        )
        try:
            measured_response, response_resolved = self._normalize_turn_probe_response(
                directed_motion,
                motion_uncertainty,
                allow_bounded_opposite=(pulse.transition is TurnTransition.REVERSAL),
            )
        except ValueError as error:
            raise _ProbeResponseError(
                f"{direction.value.title()} {expected_transition.value} trial "
                f"measured {directed_motion:+.1f} degrees with "
                f"{motion_uncertainty:.1f} degrees uncertainty: {error}. "
                "The measurement was discarded."
            ) from error

        sample = pulse.as_sample(
            measured_degrees=measured_response,
            confidence=confidence,
        )
        record: dict[str, object] = {
            "direction": direction.value,
            "transition": expected_transition.value,
            "conditioning_direction": conditioning_direction.value,
            "conditioning_prime_transition": conditioning_prime.transition.value,
            "conditioning_prime_requested_seconds": round(
                conditioning_prime.requested_seconds,
                5,
            ),
            "conditioning_prime_clamped_seconds": round(
                conditioning_prime.clamped_seconds,
                5,
            ),
            "conditioning_prime_held_seconds": round(
                conditioning_prime.held_seconds,
                5,
            ),
            "conditioning_prime_elapsed_seconds": round(
                conditioning_prime.elapsed_seconds,
                5,
            ),
            "conditioning_transition": conditioning_pulse.transition.value,
            "conditioning_start_heading": round(
                conditioning_start.angle_deg,
                3,
            ),
            "conditioning_end_heading": round(before.angle_deg, 3),
            "conditioning_start_motion_angle": self._optional_motion_angle(
                conditioning_start
            ),
            "conditioning_end_motion_angle": self._optional_motion_angle(before),
            "conditioning_requested_seconds": round(
                conditioning_pulse.requested_seconds,
                5,
            ),
            "conditioning_clamped_seconds": round(
                conditioning_pulse.clamped_seconds,
                5,
            ),
            "conditioning_held_seconds": round(
                conditioning_pulse.held_seconds,
                5,
            ),
            "conditioning_elapsed_seconds": round(
                conditioning_pulse.elapsed_seconds,
                5,
            ),
            "conditioning_measured_degrees": round(
                conditioning_motion,
                3,
            ),
            "conditioning_normalized_degrees": round(
                conditioning_response,
                3,
            ),
            "conditioning_response_resolved": conditioning_response_resolved,
            "conditioning_uncertainty_degrees": round(
                conditioning_uncertainty,
                3,
            ),
            "factor_of_coarse_90": factor,
            "start_heading": round(before.angle_deg, 3),
            "end_heading": round(after.angle_deg, 3),
            "start_motion_angle": self._optional_motion_angle(before),
            "end_motion_angle": self._optional_motion_angle(after),
            "requested_seconds": round(pulse.requested_seconds, 5),
            "clamped_seconds": round(pulse.clamped_seconds, 5),
            "held_seconds": round(pulse.held_seconds, 5),
            "elapsed_seconds": round(pulse.elapsed_seconds, 5),
            "idle_seconds": (
                round(pulse.idle_seconds, 5) if pulse.idle_seconds is not None else None
            ),
            "raw_measured_degrees": round(directed_motion, 3),
            "measured_degrees": round(measured_response, 3),
            "response_resolved": response_resolved,
            "uncertainty_degrees": round(motion_uncertainty, 3),
            "confidence": round(confidence, 3),
        }
        if response_resolved:
            self.status(
                f"Measured {direction.value} {expected_transition.value}: "
                f"{pulse.held_seconds * 1000.0:.1f} ms -> "
                f"{measured_response:.1f} degrees."
            )
        else:
            self.status(
                f"Measured {direction.value} {expected_transition.value}: "
                f"{pulse.held_seconds * 1000.0:.1f} ms produced no resolvable "
                "turn; recording a conservative zero-degree suppressed "
                "response for the dead-time fit."
            )
        return sample, record

    def _noninteractive_heading(
        self,
        context: str,
        *,
        timeout: float,
        samples: int = 9,
    ) -> HeadingReading:
        """Read a strict heading without invoking a user confirmation modal."""
        self.cancellation.raise_if_cancelled()
        reading = self.detector.read_strict(
            self._heading_frame_sample,
            samples=samples,
            delay=0.015,
            fresh=True,
            require_distinct_frames=True,
            fresh_frame_timeout=timeout,
            maximum_uncertainty_deg=self.MAXIMUM_HEADING_UNCERTAINTY_DEGREES,
            maximum_ambiguity=self.MAXIMUM_HEADING_AMBIGUITY,
        )
        self.cancellation.raise_if_cancelled()
        if reading is None or reading.confidence < 0.52:
            raise _ProbeHeadingConsensusError(
                f"{context}: a fresh heading could not be measured before "
                "the bounded non-interactive read expired."
            )
        return reading

    def _collect_probe_frames(
        self,
        *,
        count: int,
        not_before: float,
        timeout: float,
    ) -> list[FrameSample]:
        """
        Capture distinct fresh frames without running CV in the idle window.

        The frames are processed only after the target pulse begins, so detector
        work cannot silently lengthen a short neutral-timeout probe.
        """
        deadline = monotonic() + max(0.1, timeout)
        samples: list[FrameSample] = []
        identities: set[tuple[int, int]] = set()
        generation: int | None = None
        while len(samples) < count and monotonic() < deadline:
            self.cancellation.raise_if_cancelled()
            try:
                sample = self.bot.get_frame_sample()
            except WorkerCancelled:
                raise
            except Exception as error:
                raise _ProbeCaptureError(
                    "The capture supplier failed during a timed probe frame batch."
                ) from error
            if (
                sample is not None
                and sample.identity not in identities
                and sample.captured_at >= not_before
                and monotonic() - sample.captured_at <= 0.35
            ):
                if generation is None:
                    generation = sample.generation
                elif sample.generation != generation:
                    raise _ProbeCaptureError(
                        "Capture restarted during a neutral-timeout probe."
                    )
                samples.append(sample)
                identities.add(sample.identity)
            if len(samples) < count and self.cancellation.wait(0.005):
                self.cancellation.raise_if_cancelled()

        if len(samples) != count:
            raise _ProbeCaptureError(
                f"Only {len(samples)}/{count} distinct frames were captured "
                "inside the neutral-timeout probe window."
            )
        return samples

    def _noninteractive_heading_from_frames(
        self,
        samples: list[FrameSample],
        context: str,
    ) -> HeadingReading:
        """Run strict heading consensus over an explicitly captured frame batch."""

        def acquire(batch: list[FrameSample]) -> HeadingReading | None:
            frames = iter(sample.frame for sample in batch)

            def supply_frame():
                self.cancellation.raise_if_cancelled()
                return next(frames, None)

            self.cancellation.raise_if_cancelled()
            return self.detector.read_strict(
                supply_frame,
                samples=len(batch),
                delay=0.0,
                fresh=True,
                require_distinct_frames=False,
                fresh_frame_timeout=0.80,
                maximum_uncertainty_deg=self.MAXIMUM_HEADING_UNCERTAINTY_DEGREES,
                maximum_ambiguity=self.MAXIMUM_HEADING_AMBIGUITY,
            )

        reading = acquire(samples)
        if reading is None and len(samples) > 5:
            # A conditioning turn can still be settling at the beginning of
            # the capture window. The final five frames are closest to target
            # key-down, so accept them only if they independently satisfy the
            # complete strict five-sample gate.
            reading = acquire(samples[-5:])
            if reading is not None:
                self.status(
                    f"{context}: the full saved batch was still settling; "
                    "using its final five-frame strict consensus."
                )
        self.cancellation.raise_if_cancelled()
        if reading is None or reading.confidence < 0.52:
            raise _ProbeHeadingConsensusError(
                f"{context}: the captured pre-pulse frames did not produce "
                "a strict heading consensus."
            )
        return reading

    def _establish_turn_state(
        self,
        direction: str,
        seconds: float,
    ) -> tuple[TurnPulseResult, TurnPulseResult]:
        """
        Prime a requested direction, then reinforce it with a known SAME pulse.

        A first pulse after reversing can be completely suppressed by Flyff.
        Repeating the requested conditioning direction gives the probe a
        physically established state without pretending that an idle delay
        neutralized it.
        """
        prime = self._turn_burst(direction, seconds)
        reinforcing = self._turn_burst(direction, seconds)
        if reinforcing.transition is not TurnTransition.SAME_DIRECTION:
            raise RuntimeError(
                "The repeated conditioning pulse was not classified as "
                "same-direction. Turn-state history is inconsistent."
            )
        return prime, reinforcing

    def _turn_burst(
        self,
        direction: str,
        seconds: float,
    ) -> TurnPulseResult:
        self.cancellation.raise_if_cancelled()
        if direction == "left":
            result = self.controller.turn_left(seconds)
        elif direction == "right":
            result = self.controller.turn_right(seconds)
        else:
            raise ValueError(f"Unknown turn direction: {direction}")
        # Send an explicit release-all after every pulse before any vision
        # measurement. This is intentionally redundant and safety-critical.
        self.controller.stop()
        self.cancellation.raise_if_cancelled()
        self.status(
            f"{direction.title()} {result.transition.value} pulse held for "
            f"{result.held_seconds * 1000.0:.1f} ms "
            f"(requested {result.requested_seconds * 1000.0:.1f} ms, "
            f"command elapsed {result.elapsed_seconds * 1000.0:.1f} ms); "
            "all movement keys released."
        )
        return result

    def _turn_degrees_burst(
        self,
        direction: str,
        degrees: float,
        rotation_model: StateAwareRotationModel,
        *,
        maximum_seconds: float,
    ) -> TurnPulseResult:
        """Issue one state-aware bounded turn and retain the normal safety log."""
        self.cancellation.raise_if_cancelled()
        result = self.controller.turn_degrees(
            TurnDirection(direction),
            degrees,
            rotation_model,
            maximum_seconds=maximum_seconds,
        )
        self.controller.stop()
        self.cancellation.raise_if_cancelled()
        self.status(
            f"{direction.title()} {result.transition.value} pulse held for "
            f"{result.held_seconds * 1000.0:.1f} ms "
            f"(planned {degrees:.1f} degrees, requested "
            f"{result.requested_seconds * 1000.0:.1f} ms, command elapsed "
            f"{result.elapsed_seconds * 1000.0:.1f} ms); all movement keys "
            "released."
        )
        return result

    def _turn_to_absolute_heading(
        self,
        *,
        target_heading: float,
        left_seconds_90: float,
        right_seconds_90: float,
        left_sign: int,
        right_sign: int,
        label: str,
        rotation_model: StateAwareRotationModel | None = None,
        initial_reading: HeadingReading | None = None,
        noninteractive: bool = False,
    ) -> None:
        model = rotation_model or uniform_rotation_model(
            left_seconds_90=left_seconds_90,
            right_seconds_90=right_seconds_90,
            turn_memory_policy=self.controller.turn_memory_policy,
        )
        if noninteractive:

            def read_heading(context: str) -> HeadingReading:
                return self._noninteractive_heading(
                    context,
                    timeout=0.85,
                    samples=7,
                )

            heading_reader = read_heading
        else:
            heading_reader = self._stable_heading

        turner = ClosedLoopTurnController(
            self.controller,
            model,
            left_heading_sign=left_sign,
            right_heading_sign=right_sign,
            read_heading=heading_reader,
            cancellation=self.cancellation,
            status_callback=self.status,
            freshness_barrier=self._wait_for_freshness,
            settle_seconds=max(self.settle_seconds, 0.48),
            tolerance_degrees=self.TURN_TOLERANCE_DEGREES,
            maximum_uncertainty_degrees=(self.MAXIMUM_HEADING_UNCERTAINTY_DEGREES),
        )
        _ = turner.turn_to_heading(
            target_heading,
            label=label,
            initial_reading=initial_reading,
        )

    def _wait_for_freshness(self, not_before: float) -> None:
        _ = self._wait_for_frame_sample(not_before=not_before)

    def _refine_and_demonstrate(
        self,
        left_seconds_90: float,
        right_seconds_90: float,
        left_sign: int,
        right_sign: int,
        *,
        turns_each_direction: int,
        rotation_model: StateAwareRotationModel,
    ) -> tuple[float, float, list[dict[str, object]]]:
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
        records: list[dict[str, object]] = []

        sequence = self._balanced_refinement_sequence(turns_each_direction)

        for index, (direction, neutral_before) in enumerate(sequence, start=1):
            if neutral_before:
                self._prepare_neutral_transition(
                    f"Demonstration turn {index}/{len(sequence)}"
                )
            sign = signs[direction]
            opposite = "right" if direction == "left" else "left"
            start = self._stable_heading()
            target = (start.angle_deg + sign * 90.0) % 360.0

            self.status(
                f"Demonstration turn {index}/{len(sequence)} {direction}: "
                f"start {start.angle_deg:.1f}°, target {target:.1f}°."
            )

            total_held_seconds = 0.0
            total_directed_motion = 0.0
            previous = start
            corrections = 0
            stalled_reads = 0
            pulse_records: list[dict[str, object]] = []

            for correction in range(1, 11):
                self.cancellation.raise_if_cancelled()

                current = previous
                signed_to_target = signed_angle_delta(
                    target,
                    current.angle_deg,
                )
                directed_remaining = signed_to_target * sign

                if self._heading_target_satisfied(
                    signed_to_target,
                    current,
                ):
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

                planned_degrees = min(max(error_degrees, 3.0), 42.0) * 0.88

                self.status(
                    f"  correction {correction}: remaining "
                    f"{signed_to_target:+.1f}°, "
                    f"{pulse_direction} pulse planned for "
                    f"{planned_degrees:.1f}° using the fitted transition model."
                )

                pulse_result = self._turn_degrees_burst(
                    pulse_direction,
                    planned_degrees,
                    rotation_model,
                    maximum_seconds=0.180,
                )
                self._wait_or_cancel(max(self.settle_seconds, 0.48))
                after = self._stable_heading()

                signed_motion = observed_heading_delta(after, current)
                directed_motion = signed_motion * pulse_sign
                motion_uncertainty = self._reading_uncertainty(
                    current
                ) + self._reading_uncertainty(after)
                confidence = min(
                    current.confidence,
                    after.confidence,
                )
                maximum_plausible_motion = maximum_plausible_closed_loop_motion_degrees(
                    planned_degrees,
                    pulse_result,
                )
                if abs(signed_motion) - motion_uncertainty > maximum_plausible_motion:
                    debug_folder = self.detector.save_debug(
                        f"{pulse_direction}_refinement_heading_alias"
                    )
                    raise RuntimeError(
                        f"{pulse_direction.title()} refinement rejected a "
                        f"{signed_motion:+.1f}-degree heading jump after a "
                        f"{planned_degrees:.1f}-degree planned pulse; the "
                        f"physically plausible bound was "
                        f"{maximum_plausible_motion:.1f} degrees. The aliased "
                        "reading was not adopted as current state. "
                        f"Debug saved to: {debug_folder}"
                    )
                pulse_records.append(
                    {
                        "direction": pulse_direction,
                        "transition": pulse_result.transition.value,
                        "requested_seconds": round(
                            pulse_result.requested_seconds,
                            5,
                        ),
                        "clamped_seconds": round(
                            pulse_result.clamped_seconds,
                            5,
                        ),
                        "held_seconds": round(
                            pulse_result.held_seconds,
                            5,
                        ),
                        "elapsed_seconds": round(
                            pulse_result.elapsed_seconds,
                            5,
                        ),
                        "idle_seconds": (
                            round(pulse_result.idle_seconds, 5)
                            if pulse_result.idle_seconds is not None
                            else None
                        ),
                        "measured_degrees": round(
                            directed_motion,
                            3,
                        ),
                        "confidence": round(confidence, 3),
                    }
                )

                if abs(signed_motion) <= max(2.0, motion_uncertainty):
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

                if directed_motion < -max(4.0, motion_uncertainty):
                    raise RuntimeError(
                        f"{pulse_direction.title()} pulse moved opposite its "
                        f"learned direction: {signed_motion:+.1f}°."
                    )

                if max(4.0, motion_uncertainty) <= directed_motion <= 55.0:
                    equivalent_90 = pulse_result.held_seconds * 90.0 / directed_motion
                    samples[pulse_direction].append(equivalent_90)
                    durations[pulse_direction] = self._bounded_median(
                        samples[pulse_direction],
                        durations[pulse_direction],
                    )
                    self._rotation_samples.append(
                        pulse_result.as_sample(
                            measured_degrees=directed_motion,
                            confidence=confidence,
                        )
                    )

                if pulse_direction == direction:
                    total_held_seconds += pulse_result.held_seconds
                    total_directed_motion += max(0.0, directed_motion)

                previous = after
                corrections = correction
            else:
                raise RuntimeError(
                    f"Could not bring the {direction} turn within "
                    f"{self.TURN_TOLERANCE_DEGREES:.1f}° after ten "
                    "corrective pulses."
                )

            final = self._stable_heading()
            final_error = signed_angle_delta(
                target,
                final.angle_deg,
            )
            measured = observed_heading_delta(final, start) * sign

            if not self._heading_target_satisfied(final_error, final):
                raise RuntimeError(
                    f"{direction.title()} refinement ended "
                    f"{final_error:+.1f}° from target with "
                    f"{self._reading_uncertainty(final):.1f}° uncertainty."
                )

            if total_directed_motion >= 20.0:
                whole_turn_estimate = total_held_seconds * 90.0 / total_directed_motion
                samples[direction].append(whole_turn_estimate)
                durations[direction] = self._bounded_median(
                    samples[direction],
                    durations[direction],
                )

            record: dict[str, object] = {
                "direction": direction,
                "start_heading": round(start.angle_deg, 3),
                "target_heading": round(target, 3),
                "final_heading": round(final.angle_deg, 3),
                "measured_degrees": round(measured, 3),
                "final_error_degrees": round(final_error, 3),
                "corrections": corrections,
                "pulses": pulse_records,
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

            if self.cancellation.wait(0.65):
                self.cancellation.raise_if_cancelled()

        return durations["left"], durations["right"], records

    @staticmethod
    def _balanced_refinement_sequence(
        turns_each_direction: int,
    ) -> list[tuple[str, bool]]:
        """
        Cover both reversal directions without grouping all left turns first.

        The boolean marks turns that follow an intentional neutral wait. Coarse
        trials already provide repeated-direction samples; the short automatic
        sequence therefore prioritizes both reversal transitions.
        """
        if turns_each_direction == 2:
            return [
                ("left", True),
                ("right", False),
                ("right", True),
                ("left", False),
            ]
        if turns_each_direction == 4:
            return [
                ("left", True),
                ("left", False),
                ("right", False),
                ("left", False),
                ("right", False),
                ("right", True),
                ("right", False),
                ("left", False),
            ]
        raise ValueError("turns_each_direction must be 2 (automatic) or 4 (manual)")

    def _calibrate_forward_motion(
        self,
        *,
        original_heading: HeadingReading,
        manual: bool,
    ) -> tuple[ForwardMotionModel, list[ForwardCalibrationTrial]]:
        """
        Fit relative forward distance from several safe pulse durations.

        The camera is fixed by project convention. Calibration therefore stores
        the capture resolution and a visual pixels-per-cell scale. It cannot
        return the character to the starting position because the mapper has no
        calibrated reverse action; this is reported before movement begins.
        """
        self.status(
            "Phase 4/4: calibrating forward travel. Stand on a clear, textured "
            "straight path. This phase moves the character forward and cannot "
            "restore the original position; press Stop now if the path is not clear."
        )
        countdown = 5 if manual else 2
        for remaining in range(countdown, 0, -1):
            if self.cancellation.wait(1.0):
                self.cancellation.raise_if_cancelled()
            self.status(f"Forward calibration begins in {remaining}...")

        tracker = MotionTracker()
        trials: list[ForwardCalibrationTrial] = []
        diagnostic_records: list[dict[str, object]] = []
        debug_folder = (
            Path(__file__).resolve().parents[1]
            / "debug"
            / "forward_calibration"
            / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        )
        debug_folder.mkdir(parents=True, exist_ok=False)
        self._write_forward_diagnostics(
            debug_folder,
            diagnostic_records,
            requested_schedule=self.FORWARD_TRIAL_SECONDS,
        )
        try:
            baseline_heading = self._stable_heading(
                "Heading before forward calibration"
            )
        except WorkerCancelled:
            self._write_forward_diagnostics(
                debug_folder,
                diagnostic_records,
                requested_schedule=self.FORWARD_TRIAL_SECONDS,
                failure="Calibration cancelled before the forward heading baseline.",
            )
            raise
        except Exception as error:
            failure = (
                "Could not establish a heading baseline before forward "
                f"calibration: {type(error).__name__}: {error}"
            )
            self._write_forward_diagnostics(
                debug_folder,
                diagnostic_records,
                requested_schedule=self.FORWARD_TRIAL_SECONDS,
                failure=failure,
            )
            raise RuntimeError(f"{failure} Debug saved to: {debug_folder}") from error
        diagnostic_records.append(
            {
                "stage": "heading_baseline",
                "angle_deg": baseline_heading.angle_deg,
                "motion_angle_deg": baseline_heading.motion_angle_deg,
                "uncertainty_deg": self._reading_uncertainty(baseline_heading),
                "confidence": baseline_heading.confidence,
            }
        )
        baseline_target_error = observed_heading_delta(
            baseline_heading,
            original_heading,
        )
        if not self._heading_delta_satisfied(
            baseline_target_error,
            before_reading=original_heading,
            after_reading=baseline_heading,
        ):
            failure = (
                "The heading restored before forward calibration did not match "
                f"the original target ({baseline_target_error:+.3f} degrees)."
            )
            self._write_forward_diagnostics(
                debug_folder,
                diagnostic_records,
                requested_schedule=self.FORWARD_TRIAL_SECONDS,
                failure=failure,
            )
            raise RuntimeError(f"{failure} Debug saved to: {debug_folder}")
        self._write_forward_diagnostics(
            debug_folder,
            diagnostic_records,
            requested_schedule=self.FORWARD_TRIAL_SECONDS,
        )
        frame_width: int | None = None
        frame_height: int | None = None

        for index, requested_seconds in enumerate(
            self.FORWARD_TRIAL_SECONDS,
            start=1,
        ):
            try:
                before = self._wait_for_frame_sample(not_before=monotonic())
            except WorkerCancelled:
                self._write_forward_diagnostics(
                    debug_folder,
                    diagnostic_records,
                    requested_schedule=self.FORWARD_TRIAL_SECONDS,
                    failure=f"Calibration cancelled before forward trial {index}.",
                )
                raise
            except Exception as error:
                failure = (
                    f"Could not capture the frame before forward trial {index}: "
                    f"{type(error).__name__}: {error}"
                )
                self._write_forward_diagnostics(
                    debug_folder,
                    diagnostic_records,
                    requested_schedule=self.FORWARD_TRIAL_SECONDS,
                    failure=failure,
                )
                raise RuntimeError(
                    f"{failure} Debug saved to: {debug_folder}"
                ) from error
            height, width = before.frame.shape[:2]
            if frame_width is None:
                frame_width = width
                frame_height = height
            elif (width, height) != (frame_width, frame_height):
                self._write_forward_diagnostics(
                    debug_folder,
                    diagnostic_records,
                    requested_schedule=self.FORWARD_TRIAL_SECONDS,
                    failure="Capture resolution changed during forward calibration.",
                )
                raise RuntimeError(
                    "Capture resolution changed during forward calibration. "
                    f"Debug saved to: {debug_folder}"
                )

            record: dict[str, object] = {
                "trial": index,
                "requested_seconds": float(requested_seconds),
                "before_identity": before.identity,
                "after_identity": None,
                "capture_generation": before.generation,
                "before_captured_at": before.captured_at,
                "after_captured_at": None,
                "frame_width": width,
                "frame_height": height,
                "outcome": "pending_command",
            }
            diagnostic_records.append(record)
            self._write_forward_diagnostics(
                debug_folder,
                diagnostic_records,
                requested_schedule=self.FORWARD_TRIAL_SECONDS,
                trial_index=index,
                before_frame=before.frame,
            )

            self.cancellation.raise_if_cancelled()
            try:
                timing = self.controller.forward(requested_seconds)
            except WorkerCancelled:
                record["outcome"] = "cancelled_during_command"
                self._write_forward_diagnostics(
                    debug_folder,
                    diagnostic_records,
                    requested_schedule=self.FORWARD_TRIAL_SECONDS,
                    failure=f"Calibration cancelled during forward trial {index}.",
                )
                raise
            except Exception as error:
                record["outcome"] = "command_failed"
                failure = (
                    "Forward command failed before measurement: "
                    f"{type(error).__name__}: {error}"
                )
                self._write_forward_diagnostics(
                    debug_folder,
                    diagnostic_records,
                    requested_schedule=self.FORWARD_TRIAL_SECONDS,
                    failure=failure,
                )
                raise RuntimeError(
                    f"{failure} Debug saved to: {debug_folder}"
                ) from error
            record.update(
                {
                    "clamped_seconds": timing.clamped_seconds,
                    "held_seconds": timing.held_seconds,
                    "command_elapsed_seconds": timing.elapsed_seconds,
                    "outcome": "pending_capture",
                }
            )
            self._write_forward_diagnostics(
                debug_folder,
                diagnostic_records,
                requested_schedule=self.FORWARD_TRIAL_SECONDS,
            )

            try:
                self._wait_or_cancel(max(self.settle_seconds, 0.20))
                after = self._wait_for_frame_sample(
                    after_identity=before.identity,
                    generation=before.generation,
                    not_before=monotonic(),
                )
                motion = tracker.compare(
                    before.frame,
                    after.frame,
                    commanded_forward=True,
                )
            except WorkerCancelled:
                record["outcome"] = "cancelled_during_measurement"
                self._write_forward_diagnostics(
                    debug_folder,
                    diagnostic_records,
                    requested_schedule=self.FORWARD_TRIAL_SECONDS,
                    failure=f"Calibration cancelled during forward trial {index}.",
                )
                raise
            except Exception as error:
                record["outcome"] = "measurement_failed"
                failure = (
                    "Forward measurement failed after the command: "
                    f"{type(error).__name__}: {error}"
                )
                self._write_forward_diagnostics(
                    debug_folder,
                    diagnostic_records,
                    requested_schedule=self.FORWARD_TRIAL_SECONDS,
                    failure=failure,
                )
                raise RuntimeError(
                    f"{failure} Debug saved to: {debug_folder}"
                ) from error
            record.update(
                {
                    "after_identity": after.identity,
                    "after_captured_at": after.captured_at,
                    "change_score": motion.change_score,
                    "flow_dx_px": motion.directional_flow.scene_dx_px,
                    "flow_dy_px": motion.directional_flow.scene_dy_px,
                    "flow_magnitude_px": motion.directional_flow.magnitude_px,
                    "flow_dispersion_px": motion.directional_flow.dispersion_px,
                    "tracked_points": motion.directional_flow.tracked_points,
                    "flow_inlier_ratio": motion.directional_flow.inlier_ratio,
                    "flow_confidence": motion.directional_flow.confidence,
                    "teleport_likely": motion.teleport_likely,
                    "collision_likely": motion.collision_likely,
                    "outcome": motion.forward_distance.outcome.value,
                }
            )
            self._write_forward_diagnostics(
                debug_folder,
                diagnostic_records,
                requested_schedule=self.FORWARD_TRIAL_SECONDS,
                trial_index=index,
                before_frame=before.frame,
                after_frame=after.frame,
            )

            if motion.teleport_likely:
                failure = (
                    "Scene discontinuity detected during forward calibration; "
                    "the measurements were discarded."
                )
                self._write_forward_diagnostics(
                    debug_folder,
                    diagnostic_records,
                    requested_schedule=self.FORWARD_TRIAL_SECONDS,
                    failure=failure,
                )
                raise RuntimeError(f"{failure} Debug saved to: {debug_folder}")

            outcome = motion.forward_distance.outcome
            coherence_error = forward_flow_coherence_error(
                reference_flow_px=motion.directional_flow.magnitude_px,
                dispersion_px=motion.directional_flow.dispersion_px,
                inlier_ratio=motion.directional_flow.inlier_ratio,
            )
            if coherence_error is not None:
                failure = (
                    "Forward calibration produced an incoherent optical-flow "
                    f"field on trial {index}: {coherence_error}. The previous "
                    "calibration was preserved."
                )
                self._write_forward_diagnostics(
                    debug_folder,
                    diagnostic_records,
                    requested_schedule=self.FORWARD_TRIAL_SECONDS,
                    failure=failure,
                )
                raise RuntimeError(f"{failure} Debug saved to: {debug_folder}")
            if outcome is ForwardMotionOutcome.BLOCKED:
                failure = (
                    "Forward calibration encountered an obstacle. Move to a "
                    "clear textured path and run Calibrate Mapper again."
                )
                self._write_forward_diagnostics(
                    debug_folder,
                    diagnostic_records,
                    requested_schedule=self.FORWARD_TRIAL_SECONDS,
                    failure=failure,
                )
                raise RuntimeError(f"{failure} Debug saved to: {debug_folder}")
            if outcome is not ForwardMotionOutcome.MOVED:
                failure = (
                    "Forward travel could not be measured reliably "
                    f"(trial {index}, confidence "
                    f"{motion.forward_distance.confidence:.2f}, "
                    f"tracked points {motion.tracked_points}). "
                    "The previous calibration was preserved."
                )
                self._write_forward_diagnostics(
                    debug_folder,
                    diagnostic_records,
                    requested_schedule=self.FORWARD_TRIAL_SECONDS,
                    failure=failure,
                )
                raise RuntimeError(f"{failure} Debug saved to: {debug_folder}")

            trial = ForwardCalibrationTrial(
                requested_seconds=float(requested_seconds),
                actual_seconds=timing.held_seconds,
                distance_px=motion.forward_distance.distance_px,
                confidence=motion.forward_distance.confidence,
                tracked_points=motion.tracked_points,
                dispersion_px=motion.directional_flow.dispersion_px,
                inlier_ratio=motion.directional_flow.inlier_ratio,
            )
            trials.append(trial)
            self.status(
                f"Forward trial {index}/{len(self.FORWARD_TRIAL_SECONDS)}: "
                f"held {timing.held_seconds * 1000.0:.1f} ms, "
                f"flow {trial.distance_px:.2f}px, "
                f"confidence {trial.confidence:.2f}."
            )

        assert frame_width is not None
        assert frame_height is not None
        try:
            model = fit_forward_motion_model(
                trials,
                nominal_seconds=self.FORWARD_NOMINAL_SECONDS,
                frame_width=frame_width,
                frame_height=frame_height,
            )
        except ValueError as error:
            self._write_forward_diagnostics(
                debug_folder,
                diagnostic_records,
                requested_schedule=self.FORWARD_TRIAL_SECONDS,
                failure=str(error),
            )
            raise RuntimeError(
                f"Forward calibration model was rejected: {error} "
                f"Debug saved to: {debug_folder}"
            ) from error

        try:
            final_heading = self._stable_heading("Heading after forward calibration")
        except WorkerCancelled:
            self._write_forward_diagnostics(
                debug_folder,
                diagnostic_records,
                requested_schedule=self.FORWARD_TRIAL_SECONDS,
                model=model,
                failure="Calibration cancelled before the final heading check.",
            )
            raise
        except Exception as error:
            failure = (
                "Could not verify heading after forward calibration: "
                f"{type(error).__name__}: {error}"
            )
            self._write_forward_diagnostics(
                debug_folder,
                diagnostic_records,
                requested_schedule=self.FORWARD_TRIAL_SECONDS,
                model=model,
                failure=failure,
            )
            raise RuntimeError(f"{failure} Debug saved to: {debug_folder}") from error
        diagnostic_records.append(
            {
                "stage": "heading_final",
                "angle_deg": final_heading.angle_deg,
                "motion_angle_deg": final_heading.motion_angle_deg,
                "uncertainty_deg": self._reading_uncertainty(final_heading),
                "confidence": final_heading.confidence,
            }
        )
        heading_error = observed_heading_delta(final_heading, baseline_heading)
        if not self._heading_delta_satisfied(
            heading_error,
            before_reading=baseline_heading,
            after_reading=final_heading,
        ):
            combined_uncertainty = self._reading_uncertainty(
                baseline_heading
            ) + self._reading_uncertainty(final_heading)
            self._write_forward_diagnostics(
                debug_folder,
                diagnostic_records,
                requested_schedule=self.FORWARD_TRIAL_SECONDS,
                model=model,
                failure=(
                    f"Heading changed by {heading_error:+.3f} degrees; "
                    f"combined endpoint uncertainty was "
                    f"{combined_uncertainty:.3f} degrees."
                ),
            )
            raise RuntimeError(
                "Heading drifted during forward calibration "
                f"({heading_error:+.1f}°, combined uncertainty "
                f"{combined_uncertainty:.1f}°); forward measurements were "
                "discarded. "
                f"Debug saved to: {debug_folder}"
            )

        self._write_forward_diagnostics(
            debug_folder,
            diagnostic_records,
            requested_schedule=self.FORWARD_TRIAL_SECONDS,
            model=model,
        )
        self.status(
            "Forward calibration validated: "
            f"{model.pixels_per_cell:.2f}px per relative map cell, "
            f"RMSE {model.rmse_px:.2f}px, R² {model.r_squared:.2f}."
        )
        return model, trials

    @classmethod
    def _write_forward_diagnostics(
        cls,
        folder: Path,
        records: list[dict[str, object]],
        *,
        requested_schedule: tuple[float, ...],
        trial_index: int | None = None,
        before_frame: np.ndarray | None = None,
        after_frame: np.ndarray | None = None,
        model: ForwardMotionModel | None = None,
        failure: str | None = None,
    ) -> None:
        """Persist bounded Phase-4 evidence after every attempted pulse."""
        folder.mkdir(parents=True, exist_ok=True)
        if trial_index is not None:
            if before_frame is not None:
                _ = cv.imwrite(
                    str(folder / f"trial_{trial_index:02d}_before.png"),
                    before_frame,
                )
            if after_frame is not None:
                _ = cv.imwrite(
                    str(folder / f"trial_{trial_index:02d}_after.png"),
                    after_frame,
                )
        cls._atomic_write_json(
            folder / "session.json",
            {
                "estimator_version": MotionTracker.VERSION,
                "requested_schedule_seconds": list(requested_schedule),
                "completed": model is not None and failure is None,
                "failure": failure,
                "model": model.to_dict() if model is not None else None,
                "trials": records,
            },
        )

    def _wait_for_frame_sample(
        self,
        *,
        after_identity: tuple[int, int] | None = None,
        generation: int | None = None,
        not_before: float | None = None,
        timeout: float = 2.0,
    ) -> FrameSample:
        deadline = monotonic() + max(0.1, timeout)
        while monotonic() < deadline:
            self.cancellation.raise_if_cancelled()
            sample = self.bot.get_frame_sample()
            if (
                sample is not None
                and sample.identity != after_identity
                and (generation is None or sample.generation == generation)
                and (not_before is None or sample.captured_at >= not_before)
                and monotonic() - sample.captured_at <= 0.35
            ):
                return sample
            if self.cancellation.wait(0.01):
                self.cancellation.raise_if_cancelled()
        raise RuntimeError("No fresh game frame is available for calibration.")

    def _heading_target_satisfied(
        self,
        error_degrees: float,
        reading: HeadingReading,
    ) -> bool:
        return (
            abs(error_degrees) + self._reading_uncertainty(reading)
            <= self.TURN_TOLERANCE_DEGREES
        )

    def _heading_delta_satisfied(
        self,
        error_degrees: float,
        *,
        before_reading: HeadingReading,
        after_reading: HeadingReading,
    ) -> bool:
        """Apply the three-degree bound to both independently read endpoints."""
        return (
            abs(error_degrees)
            + self._reading_uncertainty(before_reading)
            + self._reading_uncertainty(after_reading)
            <= self.TURN_TOLERANCE_DEGREES
        )

    @staticmethod
    def _reading_uncertainty(reading: HeadingReading) -> float:
        value = reading.angular_uncertainty_deg
        return float(value) if value is not None else 3.0

    @staticmethod
    def _optional_motion_angle(reading: HeadingReading) -> float | None:
        value = reading.motion_angle_deg
        return round(float(value), 3) if value is not None else None

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

    def _stable_heading(self, context: str = "Heading check") -> HeadingReading:
        """
        Obtain a strict heading and optionally ask the user to approve it.

        Rejected readings are discarded and reacquired from scratch.
        """
        for attempt in range(12):
            self.cancellation.raise_if_cancelled()
            reading = self.detector.read_strict(
                self._heading_frame_sample,
                samples=15,
                delay=0.015,
                fresh=True,
                require_distinct_frames=True,
                maximum_uncertainty_deg=(self.MAXIMUM_HEADING_UNCERTAINTY_DEGREES),
                maximum_ambiguity=self.MAXIMUM_HEADING_AMBIGUITY,
            )
            self.cancellation.raise_if_cancelled()

            if reading is None or reading.confidence < 0.52:
                if attempt < 11:
                    self.status(
                        f"Strict heading was ambiguous; reacquiring ({attempt + 1}/12)."
                    )
                    self._wait_or_cancel(0.10)
                continue

            self._publish_debug(reading)
            self.cancellation.raise_if_cancelled()

            callback = self.visual_confirmation_callback
            if callback is None:
                return reading

            self.cancellation.raise_if_cancelled()
            frame = self.bot.get_frame()
            accepted = callback(
                frame,
                reading.angle_deg,
                reading.confidence,
                context,
            )
            self.cancellation.raise_if_cancelled()

            if accepted is True:
                self.status(
                    "Visual heading accepted: "
                    f"{reading.angle_deg:.1f}° "
                    f"(confidence {reading.confidence:.2f})."
                )
                return reading

            if accepted is None:
                self.cancellation.cancel()
                self.cancellation.raise_if_cancelled()

            self.status("Visual heading rejected; reacquiring from scratch.")
            self.detector.reset_fast()
            self._wait_or_cancel(max(0.10, self.settle_seconds))

        debug_folder = self.detector.save_debug("visual_or_strict_heading_failed")
        debug_text = (
            f" Debug saved to: {debug_folder}" if debug_folder is not None else ""
        )
        raise RuntimeError(
            "Could not obtain a visually accepted strict heading after "
            f"12 attempts.{debug_text}"
        )

    def _heading_frame_sample(self) -> FrameSample | None:
        self.cancellation.raise_if_cancelled()
        return self.bot.get_frame_sample()

    def _publish_debug(self, reading) -> None:
        if self.frame_callback is None:
            return
        frame = self.bot.get_debug_frame()
        if frame is not None:
            self.frame_callback(self.detector.draw_debug(frame, reading))

    def _wait_or_cancel(self, seconds: float) -> None:
        if self.cancellation.wait(seconds):
            self.cancellation.raise_if_cancelled()

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
        """
        Replace calibration only after a complete JSON file reaches disk.

        A cancelled or failed calibration therefore leaves the last known-good
        calibration in place.
        """
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{monotonic_ns()}.tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
                handle.flush()
                _ = os.fsync(handle.fileno())
            _ = temporary.replace(path)
        finally:
            _ = temporary.unlink(missing_ok=True)

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
            value for value in values if abs(value - center) <= max(3.0 * mad, 0.015)
        ]
        return float(median(filtered or values))
