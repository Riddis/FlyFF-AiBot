from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import monotonic, monotonic_ns

from capture_service import FrameSample
from worker_manager import CancellationToken, WorkerCancelled

from .CalibrationSchema import MapperCalibration
from .ForwardCalibration import (
    ForwardCalibrationTrial,
    ForwardMotionModel,
    fit_forward_motion_model,
)
from .MappingController import MappingController
from .MinimapHeading import (
    HeadingReading,
    MinimapHeadingDetector,
    signed_angle_delta,
)
from .MotionTracker import ForwardMotionOutcome, MotionTracker
from .RotationModel import (
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
from .TurnControl import ClosedLoopTurnController, uniform_rotation_model


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
    neutral_after_seconds: float
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
    FORWARD_TRIAL_SECONDS = (0.075, 0.105, 0.135, 0.090, 0.150, 0.120)
    NEUTRAL_PROBE_IDLE_SECONDS = (
        0.95,
        1.15,
        1.35,
        1.70,
        2.20,
        2.80,
        3.50,
        4.30,
        5.10,
        6.00,
    )
    NEUTRAL_PROBE_MAXIMUM_SECONDS = 6.25
    NEUTRAL_PROBE_SAFETY_MARGIN_SECONDS = 0.25
    NEUTRAL_PROBE_MAXIMUM_OVERSHOOT_SECONDS = 0.18
    NEUTRAL_PROBE_MAXIMUM_UNCERTAINTY_DEGREES = 3.0
    NEUTRAL_PROBE_VALIDATION_REPEATS = 2

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
        self.detector = MinimapHeadingDetector()
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

        rotation_model = fit_rotation_model(
            self._rotation_samples,
            fallback_left_seconds_90=refined_left_seconds_90,
            fallback_right_seconds_90=refined_right_seconds_90,
            neutral_after_seconds=neutral_timeout_fit.neutral_after_seconds,
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
            original_heading=original_heading,
            manual=manual,
        )

        result = MapperCalibration(
            version=9,
            created_at=datetime.now(timezone.utc).isoformat(),
            source="vision_fitted_turn_state_rotation_and_forward",
            left_seconds_90=round(refined_left_seconds_90, 5),
            right_seconds_90=round(refined_right_seconds_90, 5),
            left_heading_sign=left_sign,
            right_heading_sign=right_sign,
            left_trials=[asdict(item) for item in left_trials],
            right_trials=[asdict(item) for item in right_trials],
            neutral_after_seconds=round(
                neutral_timeout_fit.neutral_after_seconds,
                6,
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
        final_error = signed_angle_delta(
            original_heading,
            final_heading.angle_deg,
        )
        if not self._heading_target_satisfied(final_error, final_heading):
            raise RuntimeError(
                "Final heading validation failed "
                f"(error {final_error:+.1f}°, uncertainty "
                f"{self._reading_uncertainty(final_heading):.1f}°). "
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
        maximum_attempts = 18

        self.status(
            f"{direction.title()} calibration begins with "
            f"{pulse * 1000.0:.0f} ms bursts."
        )
        self._prepare_neutral_transition(f"{direction.title()} coarse calibration")

        while len(trials) < self.trials_per_direction:
            self.cancellation.raise_if_cancelled()
            attempts += 1
            if attempts > maximum_attempts:
                raise RuntimeError(
                    f"Could not converge on a 45-degree {direction} turn."
                )

            before = self._stable_heading()
            pulse_result = self._turn_burst(direction, pulse)
            self._wait_or_cancel(max(self.settle_seconds, 0.55))
            after = self._stable_heading()

            signed_change = signed_angle_delta(
                after.angle_deg,
                before.angle_deg,
            )
            magnitude = abs(signed_change)
            motion_uncertainty = self._reading_uncertainty(
                before
            ) + self._reading_uncertainty(after)

            self.status(
                f"{direction.title()} burst {pulse * 1000.0:.1f} ms -> "
                f"{signed_change:+.1f}°."
            )

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

    def _prepare_neutral_transition(self, label: str) -> None:
        self.controller.stop()
        wait_seconds = self.controller.neutral_after_seconds
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
        floor = max(0.0, float(acquisition_floor_seconds))
        early = [
            floor + offset
            for offset in (0.06, 0.18, 0.34)
            if floor + offset < cls.NEUTRAL_PROBE_IDLE_SECONDS[0]
        ]
        return tuple(
            sorted(
                {
                    round(value, 3)
                    for value in (*early, *cls.NEUTRAL_PROBE_IDLE_SECONDS)
                    if value <= cls.NEUTRAL_PROBE_MAXIMUM_SECONDS
                }
            )
        )

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

        try:
            acquisition_floor = max(self.settle_seconds, 0.28)
            schedule = self._build_idle_probe_schedule(acquisition_floor)
            tail_delays = set(schedule[-3:])
            for direction in TurnDirection:
                for requested_idle_seconds in schedule:
                    modes = [TurnTransition.REVERSAL]
                    if requested_idle_seconds in tail_delays:
                        modes.append(TurnTransition.SAME_DIRECTION)
                    for conditioning_transition in modes:
                        conditioning_direction = (
                            direction
                            if conditioning_transition is TurnTransition.SAME_DIRECTION
                            else direction.opposite
                        )
                        try:
                            sample, record = self._measure_neutral_timeout_probe(
                                direction=direction,
                                conditioning_transition=conditioning_transition,
                                requested_idle_seconds=requested_idle_seconds,
                                target_pulse_seconds=min(
                                    0.18, max(0.055, seconds_90[direction] * 0.28)
                                ),
                                conditioning_pulse_seconds=min(
                                    0.14,
                                    max(
                                        0.040, seconds_90[conditioning_direction] * 0.20
                                    ),
                                ),
                                heading_sign=signs[direction],
                                conditioning_heading_sign=signs[conditioning_direction],
                                phase="scan",
                            )
                        except RuntimeError as error:
                            if "exceeded the requested" in str(
                                error
                            ) or "missed its requested idle" in str(error):
                                records.append(
                                    {
                                        "phase": "scan",
                                        "direction": direction.value,
                                        "conditioning_transition": conditioning_transition.value,
                                        "requested_idle_seconds": requested_idle_seconds,
                                        "skipped": True,
                                        "reason": str(error),
                                    }
                                )
                                continue
                            raise
                        scan_samples.append(sample)
                        records.append(record)

            try:
                fit = fit_neutral_timeout(
                    scan_samples,
                    safety_margin_seconds=(self.NEUTRAL_PROBE_SAFETY_MARGIN_SECONDS),
                    maximum_idle_seconds=(self.NEUTRAL_PROBE_MAXIMUM_SECONDS),
                )
            except ValueError as error:
                raise RuntimeError(
                    "Turn-state neutral timeout could not be established from "
                    f"the bounded bidirectional scan: {error}. The previous "
                    "calibration was preserved."
                ) from error

            mode = fit.turn_memory_policy.mode
            if mode is TurnMemoryMode.DECAYS_TO_NEUTRAL:
                self.status(
                    "Turn memory converged to neutral by "
                    f"{fit.neutral_after_seconds:.3f}s. Validating SAME and "
                    "REVERSAL tails in both directions."
                )
            else:
                self.status(
                    "Turn memory remained direction-dependent through the "
                    f"{fit.turn_memory_policy.observed_horizon_seconds:.3f}s "
                    "observation horizon. Validating persistent mode."
                )
            validation_idle = (
                fit.turn_memory_policy.neutral_after_seconds
                or fit.turn_memory_policy.observed_horizon_seconds
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
                    for _ in range(self.NEUTRAL_PROBE_VALIDATION_REPEATS):
                        sample, record = self._measure_neutral_timeout_probe(
                            direction=direction,
                            conditioning_transition=conditioning_transition,
                            requested_idle_seconds=validation_idle,
                            target_pulse_seconds=min(
                                0.18, max(0.055, seconds_90[direction] * 0.28)
                            ),
                            conditioning_pulse_seconds=min(
                                0.14,
                                max(0.040, seconds_90[conditioning_direction] * 0.20),
                            ),
                            heading_sign=signs[direction],
                            conditioning_heading_sign=signs[conditioning_direction],
                            phase="validation",
                        )
                        validation_samples.append(sample)
                        records.append(record)

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
                and sample.observed_idle_seconds >= fit.neutral_after_seconds - 0.01
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
            self._rotation_samples.extend(neutral_rotation_samples)

            self.controller.set_turn_memory_policy(
                fit.turn_memory_policy,
                reset_history=True,
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
            f"({conditioning_transition.value}), "
            f"then probe {direction.value} after "
            f"{requested_idle_seconds:.3f}s idle."
        )
        conditioning_start = self._noninteractive_heading(
            f"{direction.value} neutral-timeout {phase} conditioning start",
            timeout=1.25,
            samples=7,
        )
        conditioning_pulse = self._turn_burst(
            conditioning_direction.value,
            conditioning_pulse_seconds,
        )
        self._wait_or_cancel(max(self.settle_seconds, 0.28))
        before_frames = self._collect_probe_frames(
            count=5,
            not_before=monotonic(),
            timeout=0.80,
        )

        current_idle_seconds = self.controller.turn_idle_seconds
        if current_idle_seconds is None:
            raise RuntimeError(
                "Turn-state history disappeared during neutral-timeout probing."
            )
        remaining_idle_seconds = requested_idle_seconds - current_idle_seconds
        if remaining_idle_seconds > 0.0:
            self._wait_or_cancel(remaining_idle_seconds)
        elif -remaining_idle_seconds > self.NEUTRAL_PROBE_MAXIMUM_OVERSHOOT_SECONDS:
            raise RuntimeError(
                "A strict heading read exceeded the requested neutral-timeout "
                f"probe delay by {-remaining_idle_seconds:.3f}s. The scan was "
                "discarded instead of using a mistimed state."
            )

        pulse = self._turn_burst(
            direction.value,
            target_pulse_seconds,
        )
        observed_idle_seconds = pulse.idle_seconds
        if observed_idle_seconds is None:
            raise RuntimeError(
                "Neutral-timeout probe did not report its observed idle delay."
            )
        idle_overshoot = observed_idle_seconds - requested_idle_seconds
        if (
            idle_overshoot < -0.01
            or idle_overshoot > self.NEUTRAL_PROBE_MAXIMUM_OVERSHOOT_SECONDS
        ):
            raise RuntimeError(
                f"Neutral-timeout {phase} pulse missed its requested idle "
                f"delay by {idle_overshoot:+.3f}s. The measurement was discarded."
            )

        before = self._noninteractive_heading_from_frames(
            before_frames,
            f"{direction.value} neutral-timeout {phase} start",
        )
        conditioning_motion = (
            signed_angle_delta(
                before.angle_deg,
                conditioning_start.angle_deg,
            )
            * conditioning_heading_sign
        )
        conditioning_uncertainty = self._reading_uncertainty(
            conditioning_start
        ) + self._reading_uncertainty(before)
        if (
            conditioning_uncertainty > self.NEUTRAL_PROBE_MAXIMUM_UNCERTAINTY_DEGREES
            or not max(2.0, conditioning_uncertainty) < conditioning_motion <= 70.0
        ):
            raise RuntimeError(
                f"{conditioning_direction.value.title()} conditioning pulse "
                f"moved {conditioning_motion:+.1f}° with "
                f"{conditioning_uncertainty:.1f}° uncertainty. It did not "
                "establish a reliable in-game direction state."
            )

        self._wait_or_cancel(max(self.settle_seconds, 0.48))
        after = self._noninteractive_heading(
            f"{direction.value} neutral-timeout {phase} result",
            timeout=1.75,
        )
        signed_motion = signed_angle_delta(
            after.angle_deg,
            before.angle_deg,
        )
        directed_motion = signed_motion * heading_sign
        uncertainty = self._reading_uncertainty(before) + self._reading_uncertainty(
            after
        )
        confidence = min(before.confidence, after.confidence)
        if (
            uncertainty > self.NEUTRAL_PROBE_MAXIMUM_UNCERTAINTY_DEGREES
            or not max(2.0, uncertainty) < directed_motion <= 70.0
        ):
            raise RuntimeError(
                f"{direction.value.title()} neutral-timeout {phase} probe "
                f"measured {directed_motion:+.1f}° with {uncertainty:.1f}° "
                "uncertainty. The response was outside the reliable range."
            )

        sample = NeutralTimeoutSample(
            direction=direction,
            requested_idle_seconds=float(requested_idle_seconds),
            observed_idle_seconds=float(observed_idle_seconds),
            measured_degrees=float(directed_motion),
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
            "conditioning_measured_degrees": round(
                conditioning_motion,
                3,
            ),
            "conditioning_uncertainty_degrees": round(
                conditioning_uncertainty,
                3,
            ),
            "start_heading": round(before.angle_deg, 3),
            "end_heading": round(after.angle_deg, 3),
            "requested_seconds": round(pulse.requested_seconds, 5),
            "clamped_seconds": round(pulse.clamped_seconds, 5),
            "held_seconds": round(pulse.held_seconds, 5),
            "elapsed_seconds": round(pulse.elapsed_seconds, 5),
            "controller_transition": pulse.transition.value,
            "measured_degrees": round(directed_motion, 3),
            "uncertainty_degrees": round(uncertainty, 3),
            "confidence": round(confidence, 3),
        }
        self.status(
            f"Neutral-timeout {phase} {direction.value}: "
            f"idle {observed_idle_seconds:.3f}s -> "
            f"{directed_motion:.1f}° (uncertainty {uncertainty:.1f}°)."
        )
        return sample, record

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

        for direction in TurnDirection:
            for expected_transition in (
                TurnTransition.SAME_DIRECTION,
                TurnTransition.REVERSAL,
            ):
                for factor in duration_factors:
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
                        f"Preparing {direction.value} "
                        f"{expected_transition.value} sample with a short "
                        f"{conditioning_direction.value} conditioning pulse."
                    )
                    conditioning_start = self._noninteractive_heading(
                        f"{direction.value} "
                        f"{expected_transition.value} conditioning start",
                        timeout=1.25,
                        samples=7,
                    )
                    conditioning_pulse = self._turn_burst(
                        conditioning_direction.value,
                        conditioning_seconds,
                    )
                    self._wait_or_cancel(max(self.settle_seconds, 0.28))
                    before_frames = self._collect_probe_frames(
                        count=5,
                        not_before=monotonic(),
                        timeout=0.80,
                    )
                    observed_idle = self.controller.turn_idle_seconds
                    if (
                        observed_idle is None
                        or observed_idle >= self.controller.neutral_after_seconds - 0.05
                    ):
                        raise RuntimeError(
                            "The fitted turn-state window expired while "
                            f"capturing {direction.value} "
                            f"{expected_transition.value} pre-pulse frames."
                        )

                    requested_seconds = min(
                        self.maximum_pulse_seconds,
                        max(
                            self.minimum_pulse_seconds,
                            seconds_90[direction] * factor,
                        ),
                    )
                    pulse = self._turn_burst(
                        direction.value,
                        requested_seconds,
                    )
                    if pulse.transition is not expected_transition:
                        raise RuntimeError(
                            "Turn transition changed before its measurement "
                            f"(expected {expected_transition.value}, observed "
                            f"{pulse.transition.value}). The calibration was "
                            "discarded instead of fitting the wrong state."
                        )

                    before = self._noninteractive_heading_from_frames(
                        before_frames,
                        f"{direction.value} {expected_transition.value} trial start",
                    )
                    conditioning_motion = (
                        signed_angle_delta(
                            before.angle_deg,
                            conditioning_start.angle_deg,
                        )
                        * signs[conditioning_direction]
                    )
                    conditioning_uncertainty = self._reading_uncertainty(
                        conditioning_start
                    ) + self._reading_uncertainty(before)
                    if (
                        conditioning_uncertainty
                        > self.NEUTRAL_PROBE_MAXIMUM_UNCERTAINTY_DEGREES
                        or not max(2.0, conditioning_uncertainty)
                        < conditioning_motion
                        <= 70.0
                    ):
                        raise RuntimeError(
                            f"{conditioning_direction.value.title()} "
                            f"conditioning pulse moved "
                            f"{conditioning_motion:+.1f}° with "
                            f"{conditioning_uncertainty:.1f}° uncertainty. "
                            "The requested turn state was not established."
                        )
                    self._wait_or_cancel(max(self.settle_seconds, 0.48))
                    after = self._noninteractive_heading(
                        f"{direction.value} {expected_transition.value} trial result",
                        timeout=1.75,
                    )
                    signed_motion = signed_angle_delta(
                        after.angle_deg,
                        before.angle_deg,
                    )
                    directed_motion = signed_motion * signs[direction]
                    confidence = min(before.confidence, after.confidence)
                    motion_uncertainty = self._reading_uncertainty(
                        before
                    ) + self._reading_uncertainty(after)
                    if not max(2.0, motion_uncertainty) <= directed_motion <= 70.0:
                        raise RuntimeError(
                            f"{direction.value.title()} "
                            f"{expected_transition.value} trial measured "
                            f"{directed_motion:+.1f}°. The response was outside "
                            "the safe fitting range."
                        )

                    sample = pulse.as_sample(
                        measured_degrees=directed_motion,
                        confidence=confidence,
                    )
                    self._rotation_samples.append(sample)
                    record: dict[str, object] = {
                        "direction": direction.value,
                        "transition": expected_transition.value,
                        "conditioning_direction": conditioning_direction.value,
                        "conditioning_transition": (
                            conditioning_pulse.transition.value
                        ),
                        "conditioning_start_heading": round(
                            conditioning_start.angle_deg,
                            3,
                        ),
                        "conditioning_end_heading": round(
                            before.angle_deg,
                            3,
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
                        "conditioning_measured_degrees": round(
                            conditioning_motion,
                            3,
                        ),
                        "conditioning_uncertainty_degrees": round(
                            conditioning_uncertainty,
                            3,
                        ),
                        "factor_of_coarse_90": factor,
                        "start_heading": round(before.angle_deg, 3),
                        "end_heading": round(after.angle_deg, 3),
                        "requested_seconds": round(pulse.requested_seconds, 5),
                        "clamped_seconds": round(pulse.clamped_seconds, 5),
                        "held_seconds": round(pulse.held_seconds, 5),
                        "elapsed_seconds": round(pulse.elapsed_seconds, 5),
                        "idle_seconds": (
                            round(pulse.idle_seconds, 5)
                            if pulse.idle_seconds is not None
                            else None
                        ),
                        "measured_degrees": round(directed_motion, 3),
                        "confidence": round(confidence, 3),
                    }
                    records.append(record)
                    self.status(
                        f"Measured {direction.value} "
                        f"{expected_transition.value}: "
                        f"{pulse.held_seconds * 1000.0:.1f} ms -> "
                        f"{directed_motion:.1f}°."
                    )

        return records

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
            raise RuntimeError(
                f"{context}: a fresh heading could not be measured before "
                "the bounded non-interactive read expired. Calibration was "
                "discarded."
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
        Capture distinct post-settle frames without running CV in the idle window.

        The frames are processed only after the target pulse begins, so template
        matching time cannot silently lengthen a short neutral-timeout probe.
        """
        deadline = monotonic() + max(0.1, timeout)
        samples: list[FrameSample] = []
        identities: set[tuple[int, int]] = set()
        generation: int | None = None
        while len(samples) < count and monotonic() < deadline:
            self.cancellation.raise_if_cancelled()
            sample = self.bot.get_frame_sample()
            if (
                sample is not None
                and sample.identity not in identities
                and sample.captured_at >= not_before
                and monotonic() - sample.captured_at <= 0.35
            ):
                if generation is None:
                    generation = sample.generation
                elif sample.generation != generation:
                    raise RuntimeError(
                        "Capture restarted during a neutral-timeout probe."
                    )
                samples.append(sample)
                identities.add(sample.identity)
            if len(samples) < count and self.cancellation.wait(0.005):
                self.cancellation.raise_if_cancelled()

        if len(samples) != count:
            raise RuntimeError(
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
        frames = iter(sample.frame for sample in samples)

        def supply_frame():
            self.cancellation.raise_if_cancelled()
            return next(frames, None)

        self.cancellation.raise_if_cancelled()
        reading = self.detector.read_strict(
            supply_frame,
            samples=len(samples),
            delay=0.0,
            fresh=True,
            require_distinct_frames=False,
            fresh_frame_timeout=0.80,
            maximum_uncertainty_deg=self.MAXIMUM_HEADING_UNCERTAINTY_DEGREES,
            maximum_ambiguity=self.MAXIMUM_HEADING_AMBIGUITY,
        )
        self.cancellation.raise_if_cancelled()
        if reading is None or reading.confidence < 0.52:
            raise RuntimeError(
                f"{context}: the captured pre-pulse frames did not produce "
                "a strict heading consensus. Calibration was discarded."
            )
        return reading

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
    ) -> None:
        model = rotation_model or uniform_rotation_model(
            left_seconds_90=left_seconds_90,
            right_seconds_90=right_seconds_90,
            neutral_after_seconds=self.controller.neutral_after_seconds,
        )
        turner = ClosedLoopTurnController(
            self.controller,
            model,
            left_heading_sign=left_sign,
            right_heading_sign=right_sign,
            read_heading=self._stable_heading,
            cancellation=self.cancellation,
            status_callback=self.status,
            freshness_barrier=self._wait_for_freshness,
            settle_seconds=max(self.settle_seconds, 0.48),
            tolerance_degrees=self.TURN_TOLERANCE_DEGREES,
            maximum_uncertainty_degrees=(self.MAXIMUM_HEADING_UNCERTAINTY_DEGREES),
        )
        _ = turner.turn_to_heading(target_heading, label=label)

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

                seconds_per_90 = durations[pulse_direction]
                proposed = (
                    seconds_per_90 * min(max(error_degrees, 3.0), 42.0) / 90.0 * 0.88
                )
                pulse_seconds = min(0.180, max(0.018, proposed))

                self.status(
                    f"  correction {correction}: remaining "
                    f"{signed_to_target:+.1f}°, "
                    f"{pulse_direction} pulse {pulse_seconds * 1000.0:.0f} ms."
                )

                pulse_result = self._turn_burst(
                    pulse_direction,
                    pulse_seconds,
                )
                self._wait_or_cancel(max(self.settle_seconds, 0.48))
                after = self._stable_heading()

                signed_motion = signed_angle_delta(
                    after.angle_deg,
                    current.angle_deg,
                )
                directed_motion = signed_motion * pulse_sign
                motion_uncertainty = self._reading_uncertainty(
                    current
                ) + self._reading_uncertainty(after)
                confidence = min(
                    current.confidence,
                    after.confidence,
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
            measured = (
                signed_angle_delta(
                    final.angle_deg,
                    start.angle_deg,
                )
                * sign
            )

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
        original_heading: float,
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
        frame_width: int | None = None
        frame_height: int | None = None

        for index, requested_seconds in enumerate(
            self.FORWARD_TRIAL_SECONDS,
            start=1,
        ):
            before = self._wait_for_frame_sample(not_before=monotonic())
            height, width = before.frame.shape[:2]
            if frame_width is None:
                frame_width = width
                frame_height = height
            elif (width, height) != (frame_width, frame_height):
                raise RuntimeError(
                    "Capture resolution changed during forward calibration."
                )

            self.cancellation.raise_if_cancelled()
            timing = self.controller.forward(requested_seconds)
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

            if motion.teleport_likely:
                raise RuntimeError(
                    "Scene discontinuity detected during forward calibration; "
                    "the measurements were discarded."
                )

            outcome = motion.forward_distance.outcome
            if outcome is ForwardMotionOutcome.BLOCKED:
                raise RuntimeError(
                    "Forward calibration encountered an obstacle. Move to a "
                    "clear textured path and run Calibrate Mapper again."
                )
            if outcome is not ForwardMotionOutcome.MOVED:
                raise RuntimeError(
                    "Forward travel could not be measured reliably "
                    f"(trial {index}, confidence "
                    f"{motion.forward_distance.confidence:.2f}, "
                    f"tracked points {motion.tracked_points}). "
                    "The previous calibration was preserved."
                )

            trial = ForwardCalibrationTrial(
                requested_seconds=float(requested_seconds),
                actual_seconds=timing.held_seconds,
                distance_px=motion.forward_distance.distance_px,
                confidence=motion.forward_distance.confidence,
                tracked_points=motion.tracked_points,
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
        model = fit_forward_motion_model(
            trials,
            nominal_seconds=self.FORWARD_NOMINAL_SECONDS,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        final_heading = self._stable_heading("Heading after forward calibration")
        heading_error = signed_angle_delta(
            original_heading,
            final_heading.angle_deg,
        )
        if not self._heading_target_satisfied(heading_error, final_heading):
            raise RuntimeError(
                "Heading drifted during forward calibration "
                f"({heading_error:+.1f}°); forward measurements were discarded."
            )

        self.status(
            "Forward calibration validated: "
            f"{model.pixels_per_cell:.2f}px per relative map cell, "
            f"RMSE {model.rmse_px:.2f}px, R² {model.r_squared:.2f}."
        )
        return model, trials

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

    @staticmethod
    def _reading_uncertainty(reading: HeadingReading) -> float:
        value = reading.angular_uncertainty_deg
        return float(value) if value is not None else 3.0

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
