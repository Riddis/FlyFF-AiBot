from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Protocol

import numpy as np
from capture_service import FrameSample
from libs.HumanKeyboard import HumanKeyboard, KeyPressTiming
from worker_manager import CancellationToken

from .AdaptiveMappingController import AdaptiveMappingController
from .AdaptiveMotionModel import (
    AdaptiveForwardOutcome,
    AdaptiveMotionModel,
    ForwardAssessment,
)
from .AdaptiveMotionTracker import AdaptiveMotionTracker, MotionEstimate
from .AdaptiveTurnControl import AdaptiveTurnController, AdaptiveTurnResult
from .Explorer import Explorer, ExplorerDecision
from .MapLogger import MapLogger
from .MinimapHeading import HeadingReading, MinimapHeadingDetector, signed_angle_delta
from .OccupancyGrid import FREE, UNKNOWN, OccupancyGrid, PoseIntegration
from .PangDetector import PangDetection, PangDetector

StatusCallback = Callable[[str], None]
FrameCallback = Callable[[np.ndarray], None]


class MapperBot(Protocol):
    keyboard: HumanKeyboard | None

    def get_frame_sample(self) -> FrameSample | None: ...


@dataclass(frozen=True)
class MapperConfig:
    """Safety and pacing settings for the adaptive mapper."""

    forward_seconds: float = 0.120
    settle_seconds: float = 0.20
    turn_settle_seconds: float = 0.35
    neutral_turn_wait_seconds: float = 0.75
    heading_tolerance_degrees: float = 5.0
    maximum_heading_uncertainty_degrees: float = 8.0
    maximum_heading_ambiguity: float = 0.85
    minimum_heading_confidence: float = 0.45
    maximum_forward_heading_drift_degrees: float = 8.0
    minimum_step_cells: float = 0.75
    maximum_step_cells: float = 1.25
    minimum_odometry_confidence: float = 0.42
    pang_threshold: float = 0.82
    save_every_steps: int = 5
    blocked_confirmations: int = 2

    def __post_init__(self) -> None:
        if not 0.03 <= self.forward_seconds <= 0.50:
            raise ValueError("forward_seconds is outside the safe mapper range")
        if self.settle_seconds < 0.0 or self.turn_settle_seconds < 0.0:
            raise ValueError("mapper settle times cannot be negative")
        if self.neutral_turn_wait_seconds < 0.0:
            raise ValueError("neutral turn wait cannot be negative")
        if self.heading_tolerance_degrees <= 0.0:
            raise ValueError("heading tolerance must be positive")
        if self.maximum_heading_uncertainty_degrees <= 0.0:
            raise ValueError("heading uncertainty limit must be positive")
        if not 0.0 <= self.maximum_heading_ambiguity <= 1.0:
            raise ValueError("heading ambiguity limit must be between 0 and 1")
        if not 0.0 <= self.minimum_heading_confidence <= 1.0:
            raise ValueError("heading confidence limit must be between 0 and 1")
        if not 0.0 <= self.minimum_odometry_confidence <= 1.0:
            raise ValueError("odometry confidence must be between 0 and 1")
        if self.minimum_step_cells <= 0.0:
            raise ValueError("minimum_step_cells must be positive")
        if self.maximum_step_cells < self.minimum_step_cells:
            raise ValueError("maximum_step_cells must not be below the minimum")
        if self.save_every_steps < 1 or self.blocked_confirmations < 1:
            raise ValueError("mapper counts must be positive")


@dataclass(frozen=True)
class _StepResult:
    frame_sample: FrameSample
    fast_heading: HeadingReading | None
    strict_heading: HeadingReading | None
    motion: MotionEstimate | None
    forward_assessment: ForwardAssessment | None
    key_timing: KeyPressTiming | None
    distance_cells: float | None
    integration: PoseIntegration | None
    pose_known: bool
    turn_result: AdaptiveTurnResult | None = None
    stop_reason: str | None = None
    motion_debug_path: str | None = None


class AdaptiveMapper:
    """
    Autonomous mapper with online motion learning and no mandatory calibration.

    The first run starts from conservative defaults. Small closed-loop turn
    pulses continuously refine left/right timing, while successful nominal
    forward steps teach the expected optical-flow envelope. Uncertain movement
    still fails closed so map drift is not silently accumulated.
    """

    VERSION = "1.1-multi-camera-forward-validation"

    def __init__(
        self,
        bot: MapperBot,
        status_callback: StatusCallback | None = None,
        frame_callback: FrameCallback | None = None,
        config: MapperConfig | None = None,
        cancellation: CancellationToken | None = None,
    ) -> None:
        if bot.keyboard is None:
            raise RuntimeError("Attach the Flyff window first.")

        self.bot = bot
        self.status_callback = status_callback or print
        self.frame_callback = frame_callback
        self.config = config or MapperConfig()
        self.cancellation = cancellation or CancellationToken()

        self.model_path = Path(__file__).resolve().parent / "adaptive_motion.json"
        self.motion_model, model_warning = AdaptiveMotionModel.load_or_default(
            self.model_path,
            forward_seconds=self.config.forward_seconds,
        )
        if model_warning is not None:
            self.status_callback(model_warning)

        self.controller = AdaptiveMappingController(bot.keyboard)
        self.heading_detector = MinimapHeadingDetector()
        self.tracker = AdaptiveMotionTracker()
        self.grid = OccupancyGrid()
        self.explorer = Explorer()

        self._position_known = False
        self._heading_known = False
        self._last_heading_uncertainty_deg: float | None = None
        self._blocked_observations: dict[tuple[int, int], int] = {}
        self._last_map_publish_at = 0.0
        self._map_publish_interval = 0.25
        self._set_pose_reliability(
            position_known=False,
            heading_known=False,
            note="Adaptive mapper pose has not been initialized.",
        )

        self.turner = AdaptiveTurnController(
            self.controller,
            self.motion_model,
            read_heading=self._read_heading_for_turn,
            cancellation=self.cancellation,
            status_callback=self.status_callback,
            model_update_callback=self._save_motion_model,
            neutral_wait_seconds=self.config.neutral_turn_wait_seconds,
            settle_seconds=self.config.turn_settle_seconds,
            tolerance_degrees=self.config.heading_tolerance_degrees,
            maximum_uncertainty_degrees=(
                self.config.maximum_heading_uncertainty_degrees
            ),
        )

        template_path = (
            Path(__file__).resolve().parents[1] / "assets" / "names" / "Pang.png"
        )
        self.pang = PangDetector(template_path, threshold=self.config.pang_threshold)

        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(__file__).resolve().parent / "mapping_runs" / run_id
        self.logger = MapLogger(self.output_dir / "mapping_steps.csv")

    def stop(self) -> None:
        self.cancellation.cancel()
        self.controller.stop()

    def run(self) -> Path:
        primary_error: BaseException | None = None
        try:
            return self._run()
        except BaseException as error:
            primary_error = error
            try:
                self.grid.metadata.termination_reason = (
                    f"{type(error).__name__}: {error}"
                )
            except Exception as metadata_error:  # noqa: BLE001
                error.add_note(
                    "Adaptive mapper also failed to record its termination reason: "
                    f"{metadata_error}"
                )
            raise
        finally:
            cleanup_error: Exception | None = None
            cleanup_actions = (
                ("release movement keys", self.controller.stop),
                ("save learned motion", self._save_motion_model),
                ("save the map", lambda: self.grid.save(self.output_dir)),
                (
                    "save the motion snapshot",
                    lambda: self.motion_model.save_snapshot(
                        self.output_dir / "adaptive_motion_snapshot.json"
                    ),
                ),
                ("close the mapping log", self.logger.close),
            )
            for label, action in cleanup_actions:
                try:
                    action()
                except Exception as error:  # noqa: BLE001 - complete all cleanup.
                    if primary_error is not None:
                        primary_error.add_note(
                            f"Adaptive mapper cleanup could not {label}: {error}"
                        )
                    elif cleanup_error is None:
                        cleanup_error = error
                    else:
                        cleanup_error.add_note(
                            f"Adaptive mapper cleanup also could not {label}: {error}"
                        )
            if primary_error is None and cleanup_error is not None:
                raise cleanup_error

    def _run(self) -> Path:
        self.status_callback(
            "Adaptive mapper starts in 5 seconds. Enter the dungeon, stand at "
            "the known spawn, keep the camera fixed, and leave a clear path "
            "ahead. Behind-character and top-down views are both supported, "
            "but do not change view during a run. No mapper calibration is required."
        )
        self.status_callback(
            f"Adaptive mapper {self.VERSION}; heading detector "
            f"{self.heading_detector.version()}; {self.motion_model.summary()}."
        )
        for remaining in range(5, 0, -1):
            if self.cancellation.wait(1.0):
                self.cancellation.raise_if_cancelled()
            self.status_callback(f"Mapper starting in {remaining}...")

        _ = self._wait_for_frame_sample(not_before=monotonic())
        initial_heading = self._strict_heading("Initial mapper heading", samples=15)
        self.grid.set_continuous_pose(0.0, 0.0, initial_heading.angle_deg)
        self._remember_heading(initial_heading)
        self._set_pose_reliability(
            position_known=True,
            heading_known=True,
            note="Spawn position and stable initial heading confirmed.",
        )
        self._align_to_nearest_cardinal(initial_heading)
        self.heading_detector.reset_fast()
        self._publish_map_preview(force=True)

        step = 0
        while not self.cancellation.cancelled:
            self.cancellation.raise_if_cancelled()
            decision = self.explorer.decide(self.grid)
            if decision.action == "STOP":
                reason = "Mapping completed: no reachable unexplored frontier remains."
                self.grid.metadata.termination_reason = reason
                self.status_callback(reason)
                self._publish_map_preview(force=True)
                self.grid.save(self.output_dir)
                break

            step += 1
            if decision.action == "FORWARD":
                result = self._execute_forward(step)
            else:
                result = self._execute_turn(decision)

            pang = self.pang.detect(result.frame_sample.frame)
            if pang.visible and result.pose_known:
                self.grid.add_pang_sighting(
                    self.grid.pose.x,
                    self.grid.pose.y,
                    pang.score,
                )

            self._write_step(step, decision, result, pang)
            self._report_step(step, decision, result, pang)
            self._publish_map_preview()

            if step % self.config.save_every_steps == 0:
                self.grid.save(self.output_dir)
                self._save_motion_model()

            if result.stop_reason is not None:
                self.grid.metadata.termination_reason = result.stop_reason
                self.controller.stop()
                self.grid.save(self.output_dir)
                self._save_motion_model()
                self.status_callback(result.stop_reason)
                break

        return self.output_dir

    def _align_to_nearest_cardinal(self, reading: HeadingReading) -> None:
        heading_index = self.grid.heading_index_from_degrees(reading.angle_deg)
        target = self.grid.heading_degrees_from_index(heading_index)
        error = signed_angle_delta(target, reading.angle_deg)
        if abs(error) > self.config.heading_tolerance_degrees:
            self.status_callback(
                "Aligning adaptive mapper to the nearest grid cardinal: "
                f"{reading.angle_deg:.1f}° -> {target:.1f}°."
            )
            self._set_pose_reliability(
                heading_known=False,
                note="Initial cardinal alignment is in progress.",
            )
            result = self.turner.turn_to_heading(
                target,
                label="Initial grid alignment",
                initial_reading=reading,
            )
            self.grid.set_heading_degrees(result.final_reading.angle_deg)
            self._remember_heading(result.final_reading)
            self._set_pose_reliability(
                heading_known=True,
                note="Spawn position and aligned heading confirmed.",
            )
        else:
            self.grid.set_heading_degrees(reading.angle_deg)
            self._remember_heading(reading)

    def _execute_turn(self, decision: ExplorerDecision) -> _StepResult:
        current_index = self.grid.pose.heading_index
        if decision.action == "TURN_LEFT":
            target_index = (current_index + 1) % 4
        elif decision.action == "TURN_RIGHT":
            target_index = (current_index - 1) % 4
        else:
            raise ValueError(f"Unsupported mapper turn: {decision.action}")

        target_heading = self.grid.heading_degrees_from_index(target_index)
        self._set_pose_reliability(
            heading_known=False,
            note="Turn command is in progress; position remains last confirmed.",
        )
        turn = self.turner.turn_to_heading(
            target_heading,
            label=f"Mapper {decision.action.lower()}",
        )
        self.grid.set_heading_degrees(turn.final_reading.angle_deg)
        self._remember_heading(turn.final_reading)
        self._set_pose_reliability(
            heading_known=True,
            note="Position and post-turn heading confirmed.",
        )
        if self.grid.pose.heading_index != target_index:
            raise RuntimeError(
                "Validated adaptive turn did not resolve to the intended cardinal."
            )

        sample = self._wait_for_frame_sample(not_before=monotonic())
        self.heading_detector.reset_fast()
        fast_heading = self.heading_detector.read_fast(sample.frame)
        timing = turn.pulses[-1] if turn.pulses else None
        return _StepResult(
            frame_sample=sample,
            fast_heading=fast_heading,
            strict_heading=turn.final_reading,
            motion=None,
            forward_assessment=None,
            key_timing=timing,
            distance_cells=None,
            integration=None,
            pose_known=(self._position_known and self._heading_known),
            turn_result=turn,
        )

    def _execute_forward(self, step: int) -> _StepResult:
        before = self._wait_for_frame_sample(not_before=monotonic())
        start_heading = self.grid.continuous_pose.heading_deg
        step_dx, step_dy = self.grid.DIRECTIONS[self.grid.pose.heading_index]
        target_cell = (self.grid.pose.x + step_dx, self.grid.pose.y + step_dy)
        self._set_pose_reliability(
            position_known=False,
            heading_known=False,
            note="Forward command is in progress; saved pose is last confirmed.",
        )

        self.cancellation.raise_if_cancelled()
        timing = self.controller.forward(self.motion_model.forward_seconds)
        if self.cancellation.wait(self.config.settle_seconds):
            self.cancellation.raise_if_cancelled()
        after = self._wait_for_frame_sample(
            after_identity=before.identity,
            generation=before.generation,
            not_before=monotonic(),
        )

        fast_heading = self.heading_detector.read_fast(after.frame)
        motion = self.tracker.compare(before.frame, after.frame)
        motion_debug_path = (
            self._save_motion_debug(step, before, after, motion)
            if step <= 3
            else None
        )
        if motion.teleport_likely:
            motion_debug_path = motion_debug_path or self._save_motion_debug(
                step, before, after, motion
            )
            self.grid.add_suspected_transition(
                from_x=self.grid.pose.x,
                from_y=self.grid.pose.y,
                attempted_x=target_cell[0],
                attempted_y=target_cell[1],
                heading_deg=start_heading,
                reason="Visual discontinuity after attempted forward movement.",
            )
            return _StepResult(
                frame_sample=after,
                fast_heading=fast_heading,
                strict_heading=None,
                motion=motion,
                forward_assessment=None,
                key_timing=timing,
                distance_cells=None,
                integration=None,
                pose_known=False,
                motion_debug_path=motion_debug_path,
                stop_reason=(
                    "Probable teleport detected. Mapping stopped at the last "
                    f"confirmed pose ({self.grid.pose.x}, {self.grid.pose.y})."
                ),
            )

        strict_heading = self._strict_heading(
            "Heading after forward movement",
            samples=11,
        )
        self._remember_heading(strict_heading)
        self._set_pose_reliability(
            heading_known=True,
            note="Post-forward heading confirmed; position awaits visual validation.",
        )
        heading_change = signed_angle_delta(strict_heading.angle_deg, start_heading)
        if abs(heading_change) > self.config.maximum_forward_heading_drift_degrees:
            motion_debug_path = motion_debug_path or self._save_motion_debug(
                step, before, after, motion
            )
            self.grid.set_heading_degrees(strict_heading.angle_deg)
            return _StepResult(
                frame_sample=after,
                fast_heading=fast_heading,
                strict_heading=strict_heading,
                motion=motion,
                forward_assessment=None,
                key_timing=timing,
                distance_cells=None,
                integration=None,
                pose_known=False,
                motion_debug_path=motion_debug_path,
                stop_reason=(
                    "Forward movement changed heading by "
                    f"{heading_change:+.1f}°. Pose integration was discarded "
                    "before drift could accumulate."
                ),
            )

        assessment = self.motion_model.assess_forward(
            motion.directional_flow,
            change_score=motion.change_score,
            held_seconds=timing.held_seconds,
        )
        if assessment.outcome is AdaptiveForwardOutcome.BLOCKED:
            self.grid.set_heading_degrees(strict_heading.angle_deg)
            self._set_pose_reliability(
                position_known=True,
                heading_known=True,
                note="No forward travel detected; prior position remains confirmed.",
            )
            stop_reason = self._record_blocked_observation(target_cell)
            return _StepResult(
                frame_sample=after,
                fast_heading=fast_heading,
                strict_heading=strict_heading,
                motion=motion,
                forward_assessment=assessment,
                key_timing=timing,
                distance_cells=0.0,
                integration=None,
                pose_known=(self._position_known and self._heading_known),
                stop_reason=stop_reason,
                motion_debug_path=motion_debug_path,
            )

        if assessment.outcome is AdaptiveForwardOutcome.UNCERTAIN:
            motion_debug_path = motion_debug_path or self._save_motion_debug(
                step, before, after, motion
            )
            self.motion_model.record_uncertain_forward()
            self._save_motion_model()
            self.grid.set_heading_degrees(strict_heading.angle_deg)
            return _StepResult(
                frame_sample=after,
                fast_heading=fast_heading,
                strict_heading=strict_heading,
                motion=motion,
                forward_assessment=assessment,
                key_timing=timing,
                distance_cells=None,
                integration=None,
                pose_known=False,
                motion_debug_path=motion_debug_path,
                stop_reason=(
                    "Forward travel was visually uncertain, so mapping stopped "
                    "without guessing the new position. "
                    f"Reason: {assessment.reason}."
                ),
            )

        distance_cells = assessment.distance_cells
        if (
            not assessment.reliable
            or distance_cells is None
            or not (
                self.config.minimum_step_cells
                <= distance_cells
                <= self.config.maximum_step_cells
            )
        ):
            motion_debug_path = motion_debug_path or self._save_motion_debug(
                step, before, after, motion
            )
            self.grid.set_heading_degrees(strict_heading.angle_deg)
            return _StepResult(
                frame_sample=after,
                fast_heading=fast_heading,
                strict_heading=strict_heading,
                motion=motion,
                forward_assessment=assessment,
                key_timing=timing,
                distance_cells=distance_cells,
                integration=None,
                pose_known=False,
                stop_reason="Adaptive forward distance failed its safety gate.",
                motion_debug_path=motion_debug_path,
            )

        self._blocked_observations.pop(target_cell, None)
        midpoint_heading = (start_heading + 0.5 * heading_change) % 360.0
        self.grid.set_heading_degrees(midpoint_heading)
        integration = self.grid.integrate_forward(
            distance_cells,
            confidence=assessment.confidence,
            minimum_confidence=self.config.minimum_odometry_confidence,
            maximum_distance_cells=self.config.maximum_step_cells,
        )
        if not integration.accepted:
            motion_debug_path = motion_debug_path or self._save_motion_debug(
                step, before, after, motion
            )
            self.grid.set_heading_degrees(strict_heading.angle_deg)
            return _StepResult(
                frame_sample=after,
                fast_heading=fast_heading,
                strict_heading=strict_heading,
                motion=motion,
                forward_assessment=assessment,
                key_timing=timing,
                distance_cells=distance_cells,
                integration=integration,
                pose_known=False,
                motion_debug_path=motion_debug_path,
                stop_reason=(
                    "Adaptive motion could not be integrated safely "
                    f"({integration.reason})."
                ),
            )

        self.motion_model.observe_forward(motion.directional_flow)
        self._save_motion_model()
        self.grid.set_heading_degrees(strict_heading.angle_deg)
        self._set_pose_reliability(
            position_known=True,
            heading_known=True,
            note="Forward step and post-step heading confirmed.",
        )
        return _StepResult(
            frame_sample=after,
            fast_heading=fast_heading,
            strict_heading=strict_heading,
            motion=motion,
            forward_assessment=assessment,
            key_timing=timing,
            distance_cells=distance_cells,
            integration=integration,
            pose_known=(self._position_known and self._heading_known),
            motion_debug_path=motion_debug_path,
        )

    def _save_motion_debug(
        self,
        step: int,
        before: FrameSample,
        after: FrameSample,
        motion: MotionEstimate,
    ) -> str:
        path = self.tracker.save_diagnostics(
            self.output_dir / "motion_debug",
            prefix=f"step_{step:04d}_forward",
            before=before.frame,
            after=after.frame,
            estimate=motion,
        )
        try:
            return str(path.relative_to(self.output_dir))
        except ValueError:
            return str(path)

    def _record_blocked_observation(
        self,
        target_cell: tuple[int, int],
    ) -> str | None:
        target_value = self.grid.value(*target_cell)
        if target_value == UNKNOWN:
            observations = self._blocked_observations.get(target_cell, 0) + 1
            self._blocked_observations[target_cell] = observations
            required = max(1, self.config.blocked_confirmations)
            if observations < required:
                self.status_callback(
                    "Potential obstacle ahead needs confirmation "
                    f"({observations}/{required}); map unchanged."
                )
                return None
            blocked_recorded = self.grid.mark_blocked(*target_cell)
            self._blocked_observations.pop(target_cell, None)
        else:
            blocked_recorded = self.grid.mark_blocked(*target_cell)

        if target_value == FREE or not blocked_recorded:
            return (
                "A blocked-motion reading conflicted with an already known "
                "free or out-of-bounds target cell. Existing map preserved."
            )
        return None

    def _set_pose_reliability(
        self,
        *,
        position_known: bool | None = None,
        heading_known: bool | None = None,
        note: str,
    ) -> None:
        if position_known is not None:
            self._position_known = bool(position_known)
        if heading_known is not None:
            self._heading_known = bool(heading_known)
        self.grid.set_pose_reliability(
            position_known=self._position_known,
            heading_known=self._heading_known,
            note=note,
        )

    def _remember_heading(self, reading: HeadingReading) -> None:
        uncertainty = reading.angular_uncertainty_deg
        self._last_heading_uncertainty_deg = (
            float(uncertainty)
            if uncertainty is not None
            else self.config.maximum_heading_uncertainty_degrees
        )
        self.grid.metadata.heading_uncertainty_deg = self._last_heading_uncertainty_deg

    def _read_heading_for_turn(self, context: str, samples: int) -> HeadingReading:
        return self._strict_heading(context, samples=samples)

    def _strict_heading(
        self,
        context: str,
        *,
        samples: int,
    ) -> HeadingReading:
        for attempt in range(3):
            self.cancellation.raise_if_cancelled()
            reading = self.heading_detector.read_strict(
                self._heading_frame_sample,
                samples=max(5, int(samples)),
                delay=0.015,
                fresh=True,
                require_distinct_frames=True,
                fresh_frame_timeout=0.80,
                maximum_uncertainty_deg=(
                    self.config.maximum_heading_uncertainty_degrees
                ),
                maximum_ambiguity=self.config.maximum_heading_ambiguity,
            )
            self.cancellation.raise_if_cancelled()
            if (
                reading is not None
                and reading.confidence >= self.config.minimum_heading_confidence
            ):
                return reading
            if attempt < 2:
                self.status_callback(
                    f"{context} was ambiguous; stable reacquisition {attempt + 1}/3."
                )
                if self.cancellation.wait(0.10):
                    self.cancellation.raise_if_cancelled()

        debug_folder = self.heading_detector.save_debug(
            "adaptive_mapper_heading_failed"
        )
        debug_text = (
            f" Debug saved to: {debug_folder}" if debug_folder is not None else ""
        )
        raise RuntimeError(
            f"Could not obtain a stable fresh minimap heading.{debug_text}"
        )

    def _heading_frame_sample(self) -> FrameSample | None:
        self.cancellation.raise_if_cancelled()
        return self.bot.get_frame_sample()

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
        raise RuntimeError("No fresh game frame is available for mapping.")

    def _save_motion_model(self) -> None:
        self.motion_model.save(self.model_path)

    def _write_step(
        self,
        step: int,
        decision: ExplorerDecision,
        result: _StepResult,
        pang: PangDetection,
    ) -> None:
        motion = result.motion
        flow = motion.directional_flow if motion is not None else None
        assessment = result.forward_assessment
        fast = result.fast_heading
        strict = result.strict_heading
        timing = result.key_timing
        continuous = self.grid.continuous_pose

        expected = assessment.expected_flow_px if assessment is not None else None
        observed = assessment.observed_flow_px if assessment is not None else None
        residual = (
            abs(observed - expected)
            if observed is not None and expected is not None
            else None
        )

        self.logger.write(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(
                    timespec="milliseconds"
                ),
                "step": step,
                "x": self.grid.pose.x,
                "y": self.grid.pose.y,
                "continuous_x": round(continuous.x, 4),
                "continuous_y": round(continuous.y, 4),
                "position_known": self._position_known,
                "heading_known": self._heading_known,
                "pose_known": result.pose_known,
                "heading_index": self.grid.pose.heading_index,
                "heading_deg": round(continuous.heading_deg, 3),
                "action": decision.action,
                "reason": decision.reason,
                "frame_sequence": result.frame_sample.sequence,
                "requested_seconds": (
                    round(timing.requested_seconds, 5) if timing is not None else ""
                ),
                "held_seconds": (
                    round(timing.held_seconds, 5) if timing is not None else ""
                ),
                "change_score": (
                    round(motion.change_score, 5) if motion is not None else ""
                ),
                "flow_dx_px": round(flow.scene_dx_px, 3) if flow is not None else "",
                "flow_dy_px": round(flow.scene_dy_px, 3) if flow is not None else "",
                "median_flow_px": (
                    round(flow.magnitude_px, 3) if flow is not None else ""
                ),
                "flow_dispersion_px": (
                    round(flow.dispersion_px, 3) if flow is not None else ""
                ),
                "flow_confidence": (
                    round(flow.confidence, 3) if flow is not None else ""
                ),
                "flow_inlier_ratio": (
                    round(flow.inlier_ratio, 3) if flow is not None else ""
                ),
                "tracked_points": flow.tracked_points if flow is not None else "",
                "flow_detected_points": (
                    flow.detected_points if flow is not None else ""
                ),
                "flow_valid_tracks": flow.valid_tracks if flow is not None else "",
                "flow_moving_points": (
                    flow.moving_points if flow is not None else ""
                ),
                "flow_moving_ratio": (
                    round(flow.moving_ratio, 3) if flow is not None else ""
                ),
                "flow_spatial_coverage": (
                    round(flow.spatial_coverage, 3) if flow is not None else ""
                ),
                "flow_occupied_regions": (
                    flow.occupied_regions if flow is not None else ""
                ),
                "flow_translation_coherence": (
                    round(flow.translation_coherence, 3)
                    if flow is not None
                    else ""
                ),
                "flow_expansion_coherence": (
                    round(flow.expansion_coherence, 3)
                    if flow is not None
                    else ""
                ),
                "flow_camera_model": flow.camera_model if flow is not None else "",
                "motion_debug_path": result.motion_debug_path or "",
                "motion_outcome": (
                    assessment.outcome.value if assessment is not None else "turn"
                ),
                "distance_cells": (
                    round(result.distance_cells, 4)
                    if result.distance_cells is not None
                    else ""
                ),
                "expected_flow_px": round(expected, 3) if expected is not None else "",
                "observed_motion_px": (
                    round(observed, 3) if observed is not None else ""
                ),
                "flow_residual_px": (
                    round(residual, 3) if residual is not None else ""
                ),
                "maximum_flow_residual_px": "",
                "flow_validation_reason": (
                    assessment.reason if assessment is not None else ""
                ),
                "odometry_integrated": (
                    result.integration.accepted
                    if result.integration is not None
                    else False
                ),
                "collision": (
                    assessment is not None
                    and assessment.outcome is AdaptiveForwardOutcome.BLOCKED
                ),
                "pang_visible": pang.visible,
                "pang_score": round(pang.score, 4),
                "teleport_suspected": (
                    motion.teleport_likely if motion is not None else False
                ),
                "fast_heading": round(fast.angle_deg, 3) if fast is not None else "",
                "fast_heading_confidence": (
                    round(fast.confidence, 3) if fast is not None else ""
                ),
                "fast_heading_uncertainty": (
                    round(fast.angular_uncertainty_deg, 3)
                    if fast is not None and fast.angular_uncertainty_deg is not None
                    else ""
                ),
                "fast_heading_stale": fast.is_stale if fast is not None else True,
                "strict_heading": (
                    round(strict.angle_deg, 3) if strict is not None else ""
                ),
                "strict_heading_confidence": (
                    round(strict.confidence, 3) if strict is not None else ""
                ),
                "strict_heading_uncertainty": (
                    round(strict.angular_uncertainty_deg, 3)
                    if strict is not None and strict.angular_uncertainty_deg is not None
                    else ""
                ),
                "stop_reason": result.stop_reason or "",
            }
        )

    def _report_step(
        self,
        step: int,
        decision: ExplorerDecision,
        result: _StepResult,
        pang: PangDetection,
    ) -> None:
        if result.forward_assessment is not None:
            motion_text = result.forward_assessment.outcome.value
        else:
            motion_text = "turn"
        strict_text = (
            f"{result.strict_heading.angle_deg:.1f}°"
            if result.strict_heading is not None
            else "unavailable"
        )
        camera_text = (
            result.motion.directional_flow.camera_model
            if result.motion is not None
            else "n/a"
        )
        self.status_callback(
            f"map step={step} pose=({self.grid.continuous_pose.x:.2f},"
            f"{self.grid.continuous_pose.y:.2f}) "
            f"heading={self.grid.continuous_pose.heading_deg:.1f}° "
            f"action={decision.action} motion={motion_text} camera={camera_text} "
            f"strict_heading={strict_text} pose_known={result.pose_known} "
            f"pang={pang.visible}; {self.motion_model.summary()}"
        )

    def _publish_map_preview(self, force: bool = False) -> None:
        if self.frame_callback is None:
            return
        now = monotonic()
        if not force and now - self._last_map_publish_at < self._map_publish_interval:
            return
        self.frame_callback(self.grid.render())
        self._last_map_publish_at = now


# Keep the public name expected by RuntimeController and existing imports.
Mapper = AdaptiveMapper
