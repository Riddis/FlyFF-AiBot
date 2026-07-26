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

from .CalibrationSchema import CalibrationSchemaError, MapperCalibration
from .Explorer import Explorer, ExplorerDecision
from .ForwardCalibration import ForwardMotionModel
from .MapLogger import MapLogger
from .MappingController import MappingController
from .MinimapHeading import (
    HeadingReading,
    MinimapHeadingDetector,
    signed_angle_delta,
)
from .MotionTracker import (
    ForwardMotionOutcome,
    MotionEstimate,
    MotionTracker,
)
from .OccupancyGrid import FREE, UNKNOWN, OccupancyGrid, PoseIntegration
from .PangDetector import PangDetection, PangDetector
from .RotationModel import StateAwareRotationModel
from .TurnControl import ClosedLoopTurnController

StatusCallback = Callable[[str], None]
FrameCallback = Callable[[np.ndarray], None]


class MapperBot(Protocol):
    keyboard: HumanKeyboard | None

    def get_frame_sample(self) -> FrameSample | None: ...


def _mapper_calibration_error(message: str) -> RuntimeError:
    return RuntimeError(message)


@dataclass(frozen=True)
class MapperConfig:
    rotation_model: StateAwareRotationModel
    forward_model: ForwardMotionModel
    left_heading_sign: int
    right_heading_sign: int
    settle_seconds: float = 0.20
    turn_settle_seconds: float = 0.48
    heading_tolerance_degrees: float = 3.0
    maximum_heading_uncertainty_degrees: float = 3.0
    maximum_heading_ambiguity: float = 0.70
    minimum_step_cells: float = 0.20
    maximum_step_cells: float = 2.00
    pang_threshold: float = 0.82
    save_every_steps: int = 5
    blocked_confirmations: int = 2


@dataclass(frozen=True)
class _StepResult:
    frame_sample: FrameSample
    fast_heading: HeadingReading | None
    strict_heading: HeadingReading | None
    motion: MotionEstimate | None
    key_timing: KeyPressTiming | None
    distance_cells: float | None
    integration: PoseIntegration | None
    pose_known: bool
    stop_reason: str | None = None


class Mapper:
    """Accuracy-first autonomous mapper using measured heading and travel."""

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
        self.config = config or self._load_config()
        self.cancellation = cancellation or CancellationToken()

        self.controller = MappingController(
            bot.keyboard,
            turn_memory_policy=self.config.rotation_model.turn_memory_policy,
        )
        self.heading_detector = MinimapHeadingDetector()
        self.tracker = MotionTracker(forward_model=self.config.forward_model)
        self.grid = OccupancyGrid()
        self._position_known = False
        self._heading_known = False
        self._last_heading_uncertainty_deg: float | None = None
        self._set_pose_reliability(
            position_known=False,
            heading_known=False,
            note="Mapper pose has not been initialized.",
        )
        self.explorer = Explorer()
        self._blocked_observations: dict[tuple[int, int], int] = {}
        self._last_map_publish_at = 0.0
        self._map_publish_interval = 0.25

        self.turner = ClosedLoopTurnController(
            self.controller,
            self.config.rotation_model,
            left_heading_sign=self.config.left_heading_sign,
            right_heading_sign=self.config.right_heading_sign,
            read_heading=self._strict_heading,
            cancellation=self.cancellation,
            status_callback=self.status_callback,
            freshness_barrier=self._freshness_barrier,
            settle_seconds=self.config.turn_settle_seconds,
            tolerance_degrees=self.config.heading_tolerance_degrees,
            maximum_uncertainty_degrees=(
                self.config.maximum_heading_uncertainty_degrees
            ),
        )

        template_path = (
            Path(__file__).resolve().parents[1] / "assets" / "names" / "Pang.png"
        )
        self.pang = PangDetector(
            template_path,
            threshold=self.config.pang_threshold,
        )

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
                    "Mapper also failed to record its termination reason: "
                    f"{metadata_error}"
                )
            raise
        finally:
            cleanup_error: Exception | None = None
            cleanup_actions = (
                ("release movement keys", self.controller.stop),
                ("save the map", lambda: self.grid.save(self.output_dir)),
                ("close the mapping log", self.logger.close),
            )
            for label, action in cleanup_actions:
                try:
                    action()
                except Exception as error:  # noqa: BLE001 - complete all cleanup.
                    if primary_error is not None:
                        primary_error.add_note(
                            f"Mapper cleanup could not {label}: {error}"
                        )
                    elif cleanup_error is None:
                        cleanup_error = error
                    else:
                        cleanup_error.add_note(
                            f"Mapper cleanup also could not {label}: {error}"
                        )
            if primary_error is None and cleanup_error is not None:
                raise cleanup_error

    def _run(self) -> Path:
        self.status_callback(
            "Mapper starts in 5 seconds. Enter the dungeon, stand at the known "
            "spawn, and make sure the calibrated fixed camera is active."
        )
        for remaining in range(5, 0, -1):
            if self.cancellation.wait(1.0):
                self.cancellation.raise_if_cancelled()
            self.status_callback(f"Mapper starting in {remaining}...")

        initial_frame = self._wait_for_frame_sample(not_before=monotonic())
        if not self.config.forward_model.matches_frame(initial_frame.frame):
            raise RuntimeError(
                "Capture resolution differs from the forward calibration. "
                "Run Calibrate Mapper again at this resolution."
            )

        initial_heading = self._strict_heading("Initial mapper heading")
        self.grid.set_continuous_pose(
            0.0,
            0.0,
            initial_heading.angle_deg,
        )
        self._remember_heading(initial_heading)
        self._set_pose_reliability(
            position_known=True,
            heading_known=True,
            note="Spawn position and strict initial heading confirmed.",
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
                result = self._execute_forward(decision)
            else:
                result = self._execute_turn(decision)

            pang = self.pang.detect(result.frame_sample.frame)
            if pang.visible and result.pose_known:
                self.grid.add_pang_sighting(
                    self.grid.pose.x,
                    self.grid.pose.y,
                    pang.score,
                )

            self._write_step(
                step=step,
                decision=decision,
                result=result,
                pang=pang,
            )
            self._report_step(step, decision, result, pang)
            self._publish_map_preview()

            if step % self.config.save_every_steps == 0:
                self.grid.save(self.output_dir)

            if result.stop_reason is not None:
                self.grid.metadata.termination_reason = result.stop_reason
                self.controller.stop()
                self.grid.save(self.output_dir)
                self.status_callback(result.stop_reason)
                break

        return self.output_dir

    def _align_to_nearest_cardinal(self, reading: HeadingReading) -> None:
        heading_index = self.grid.heading_index_from_degrees(reading.angle_deg)
        target = self.grid.heading_degrees_from_index(heading_index)
        error = signed_angle_delta(target, reading.angle_deg)
        if not self._heading_target_satisfied(error, reading):
            self.status_callback(
                "Aligning mapper pose to the nearest north-up grid cardinal: "
                f"{reading.angle_deg:.1f}° -> {target:.1f}°."
            )
            self._set_pose_reliability(
                heading_known=False,
                note=(
                    "Position is known; initial alignment is in progress and "
                    "heading is awaiting strict confirmation."
                ),
            )
            result = self.turner.turn_to_heading(
                target,
                label="Initial grid alignment",
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
            note=(
                "Position is last confirmed; turn command is in progress and "
                "heading is awaiting strict confirmation."
            ),
        )
        turn = self.turner.turn_to_heading(
            target_heading,
            label=f"Mapper {decision.action.lower()}",
        )

        # Apply orientation only after the strict final reading confirms the
        # requested cardinal. Never pre-mutate the planner pose.
        self.grid.set_heading_degrees(turn.final_reading.angle_deg)
        self._remember_heading(turn.final_reading)
        self._set_pose_reliability(
            heading_known=True,
            note="Position and post-turn heading confirmed.",
        )
        if self.grid.pose.heading_index != target_index:
            raise RuntimeError(
                "Validated turn did not resolve to the intended grid cardinal."
            )

        sample = self._wait_for_frame_sample()
        self.heading_detector.reset_fast()
        fast_heading = self.heading_detector.read_fast(sample.frame)
        return _StepResult(
            frame_sample=sample,
            fast_heading=fast_heading,
            strict_heading=turn.final_reading,
            motion=None,
            key_timing=None,
            distance_cells=None,
            integration=None,
            pose_known=(self._position_known and self._heading_known),
        )

    def _execute_forward(self, decision: ExplorerDecision) -> _StepResult:
        del decision
        before = self._wait_for_frame_sample(not_before=monotonic())
        start_heading = self.grid.continuous_pose.heading_deg
        start_heading_uncertainty = (
            self._last_heading_uncertainty_deg
            if self._last_heading_uncertainty_deg is not None
            else self.config.maximum_heading_uncertainty_degrees
        )
        step_dx, step_dy = self.grid.DIRECTIONS[self.grid.pose.heading_index]
        target_cell = (
            self.grid.pose.x + step_dx,
            self.grid.pose.y + step_dy,
        )
        self._set_pose_reliability(
            position_known=False,
            heading_known=False,
            note=(
                "Saved coordinates are the last confirmed pose before an "
                "in-progress forward command."
            ),
        )
        self.cancellation.raise_if_cancelled()
        timing = self.controller.forward(self.config.forward_model.nominal_seconds)
        if self.cancellation.wait(self.config.settle_seconds):
            self.cancellation.raise_if_cancelled()
        after = self._wait_for_frame_sample(
            after_identity=before.identity,
            generation=before.generation,
            not_before=monotonic(),
        )

        fast_heading = self.heading_detector.read_fast(after.frame)
        motion = self.tracker.compare(
            before.frame,
            after.frame,
            commanded_forward=True,
            actual_forward_seconds=timing.held_seconds,
        )

        if motion.teleport_likely:
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
                key_timing=timing,
                distance_cells=None,
                integration=None,
                pose_known=False,
                stop_reason=(
                    "Probable teleport detected. Movement stopped and the map "
                    f"was saved at last confirmed pose ({self.grid.pose.x}, "
                    f"{self.grid.pose.y})."
                ),
            )

        strict_heading = self._strict_heading("Heading after forward movement")
        self._remember_heading(strict_heading)
        self._set_pose_reliability(
            heading_known=True,
            note=(
                "Post-forward heading is confirmed; position is awaiting "
                "validated odometry."
            ),
        )
        heading_change = signed_angle_delta(
            strict_heading.angle_deg,
            start_heading,
        )
        if not self._heading_delta_satisfied(
            heading_change,
            before_uncertainty_degrees=start_heading_uncertainty,
            after_reading=strict_heading,
        ):
            self.grid.set_heading_degrees(strict_heading.angle_deg)
            return _StepResult(
                frame_sample=after,
                fast_heading=fast_heading,
                strict_heading=strict_heading,
                motion=motion,
                key_timing=timing,
                distance_cells=None,
                integration=None,
                pose_known=False,
                stop_reason=(
                    "Forward movement changed heading by "
                    f"{heading_change:+.1f}°. Pose integration was discarded "
                    "and mapping stopped before drift could accumulate."
                ),
            )

        outcome = motion.forward_distance.outcome
        if outcome is ForwardMotionOutcome.UNAVAILABLE:
            self.grid.set_heading_degrees(strict_heading.angle_deg)
            validation = motion.forward_distance.validation
            validation_detail = (
                f" Validation failed: {validation.reason}."
                if validation is not None and validation.reason is not None
                else ""
            )
            return _StepResult(
                frame_sample=after,
                fast_heading=fast_heading,
                strict_heading=strict_heading,
                motion=motion,
                key_timing=timing,
                distance_cells=None,
                integration=None,
                pose_known=False,
                stop_reason=(
                    "Forward motion could not be measured. The character may "
                    "have moved, so mapping stopped immediately without "
                    f"guessing its new position.{validation_detail}"
                ),
            )

        if outcome is ForwardMotionOutcome.BLOCKED:
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
                key_timing=timing,
                distance_cells=0.0,
                integration=None,
                pose_known=(self._position_known and self._heading_known),
                stop_reason=stop_reason,
            )

        distance_cells = motion.forward_distance.distance_cells
        if (
            outcome is not ForwardMotionOutcome.MOVED
            or not motion.forward_distance.reliable
            or distance_cells is None
        ):
            raise RuntimeError(
                "Motion tracker reported movement without a reliable calibrated "
                "distance."
            )
        if not (
            self.config.minimum_step_cells
            <= distance_cells
            <= self.config.maximum_step_cells
        ):
            self.grid.set_heading_degrees(strict_heading.angle_deg)
            return _StepResult(
                frame_sample=after,
                fast_heading=fast_heading,
                strict_heading=strict_heading,
                motion=motion,
                key_timing=timing,
                distance_cells=distance_cells,
                integration=None,
                pose_known=False,
                stop_reason=(
                    "Measured forward distance was outside the calibrated "
                    f"safety range ({distance_cells:.2f} cells). Pose was not "
                    "changed."
                ),
            )

        midpoint_heading = (start_heading + 0.5 * heading_change) % 360.0
        self._blocked_observations.pop(target_cell, None)
        self.grid.set_heading_degrees(midpoint_heading)
        integration = self.grid.integrate_forward(
            distance_cells,
            confidence=motion.forward_distance.confidence,
            minimum_confidence=self.tracker.minimum_motion_confidence,
        )
        if not integration.accepted:
            self.grid.set_heading_degrees(strict_heading.angle_deg)
            return _StepResult(
                frame_sample=after,
                fast_heading=fast_heading,
                strict_heading=strict_heading,
                motion=motion,
                key_timing=timing,
                distance_cells=distance_cells,
                integration=integration,
                pose_known=False,
                stop_reason=(
                    "Measured motion could not be integrated safely "
                    f"({integration.reason}). Mapping stopped without changing "
                    "position."
                ),
            )
        self.grid.set_heading_degrees(strict_heading.angle_deg)
        self._set_pose_reliability(
            position_known=True,
            heading_known=True,
            note="Forward odometry and post-step heading confirmed.",
        )
        return _StepResult(
            frame_sample=after,
            fast_heading=fast_heading,
            strict_heading=strict_heading,
            motion=motion,
            key_timing=timing,
            distance_cells=distance_cells,
            integration=integration,
            pose_known=(self._position_known and self._heading_known),
        )

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
                    f"({observations}/{required}); the map is unchanged."
                )
                return None
            blocked_recorded = self.grid.mark_blocked(*target_cell)
            self._blocked_observations.pop(target_cell, None)
        else:
            blocked_recorded = self.grid.mark_blocked(*target_cell)

        if target_value == FREE or not blocked_recorded:
            return (
                "A blocked-motion reading conflicted with an already known "
                "free or out-of-bounds target cell. The existing map was "
                "preserved and mapping stopped for review."
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

    def _strict_heading(self, context: str = "Mapper heading") -> HeadingReading:
        for attempt in range(4):
            self.cancellation.raise_if_cancelled()
            reading = self.heading_detector.read_strict(
                self._heading_frame_sample,
                samples=15,
                delay=0.015,
                fresh=True,
                require_distinct_frames=True,
                maximum_uncertainty_deg=(
                    self.config.maximum_heading_uncertainty_degrees
                ),
                maximum_ambiguity=self.config.maximum_heading_ambiguity,
            )
            self.cancellation.raise_if_cancelled()
            if reading is not None and reading.confidence >= 0.52:
                return reading
            if attempt < 3:
                self.status_callback(
                    f"{context} was ambiguous; strict reacquisition {attempt + 1}/4."
                )
                if self.cancellation.wait(0.10):
                    self.cancellation.raise_if_cancelled()

        debug_folder = self.heading_detector.save_debug("mapper_strict_heading_failed")
        debug_text = (
            f" Debug saved to: {debug_folder}" if debug_folder is not None else ""
        )
        raise RuntimeError(
            f"Could not obtain an accurate fresh minimap heading.{debug_text}"
        )

    def _heading_frame_sample(self) -> FrameSample | None:
        self.cancellation.raise_if_cancelled()
        return self.bot.get_frame_sample()

    def _heading_target_satisfied(
        self,
        error_degrees: float,
        reading: HeadingReading,
    ) -> bool:
        uncertainty = reading.angular_uncertainty_deg
        uncertainty_degrees = float(uncertainty) if uncertainty is not None else 3.0
        return (
            abs(error_degrees) + uncertainty_degrees
            <= self.config.heading_tolerance_degrees
        )

    def _heading_delta_satisfied(
        self,
        error_degrees: float,
        *,
        before_uncertainty_degrees: float,
        after_reading: HeadingReading,
    ) -> bool:
        after_uncertainty = after_reading.angular_uncertainty_deg
        after_uncertainty_degrees = (
            float(after_uncertainty)
            if after_uncertainty is not None
            else self.config.maximum_heading_uncertainty_degrees
        )
        return (
            abs(error_degrees)
            + max(0.0, float(before_uncertainty_degrees))
            + after_uncertainty_degrees
            <= self.config.heading_tolerance_degrees
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
        raise RuntimeError("No fresh game frame is available for mapping.")

    def _freshness_barrier(self, not_before: float) -> None:
        _ = self._wait_for_frame_sample(not_before=not_before)

    def _write_step(
        self,
        *,
        step: int,
        decision: ExplorerDecision,
        result: _StepResult,
        pang: PangDetection,
    ) -> None:
        motion = result.motion
        flow = motion.directional_flow if motion is not None else None
        distance = motion.forward_distance if motion is not None else None
        validation = distance.validation if distance is not None else None
        fast = result.fast_heading
        strict = result.strict_heading
        timing = result.key_timing
        continuous = self.grid.continuous_pose

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
                "flow_dx_px": (round(flow.scene_dx_px, 3) if flow is not None else ""),
                "flow_dy_px": (round(flow.scene_dy_px, 3) if flow is not None else ""),
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
                "motion_outcome": distance.outcome.value
                if distance is not None
                else "",
                "distance_cells": (
                    round(result.distance_cells, 4)
                    if result.distance_cells is not None
                    else ""
                ),
                "expected_flow_px": (
                    round(validation.expected_flow_px, 3)
                    if validation is not None
                    and validation.expected_flow_px is not None
                    else ""
                ),
                "observed_motion_px": (
                    round(validation.observed_motion_px, 3)
                    if validation is not None
                    and validation.observed_motion_px is not None
                    else ""
                ),
                "flow_residual_px": (
                    round(validation.residual_px, 3)
                    if validation is not None and validation.residual_px is not None
                    else ""
                ),
                "maximum_flow_residual_px": (
                    round(validation.maximum_residual_px, 3)
                    if validation is not None
                    and validation.maximum_residual_px is not None
                    else ""
                ),
                "flow_validation_reason": (
                    validation.reason
                    if validation is not None and validation.reason is not None
                    else ""
                ),
                "odometry_integrated": (
                    result.integration.accepted
                    if result.integration is not None
                    else False
                ),
                "collision": (motion.collision_likely if motion is not None else False),
                "pang_visible": pang.visible,
                "pang_score": round(pang.score, 4),
                "teleport_suspected": (
                    motion.teleport_likely if motion is not None else False
                ),
                "fast_heading": (round(fast.angle_deg, 3) if fast is not None else ""),
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
        motion_text = "turn"
        if result.motion is not None:
            motion_text = result.motion.forward_distance.outcome.value
        strict_text = (
            f"{result.strict_heading.angle_deg:.1f}°"
            if result.strict_heading is not None
            else "unavailable"
        )
        self.status_callback(
            f"map step={step} pose=({self.grid.continuous_pose.x:.2f},"
            f"{self.grid.continuous_pose.y:.2f}) "
            f"heading={self.grid.continuous_pose.heading_deg:.1f}° "
            f"action={decision.action} motion={motion_text} "
            f"strict_heading={strict_text} pose_known={result.pose_known} "
            f"pang={pang.visible}"
        )

    def _publish_map_preview(self, force: bool = False) -> None:
        if self.frame_callback is None:
            return
        now = monotonic()
        if not force and now - self._last_map_publish_at < self._map_publish_interval:
            return
        self.frame_callback(self.grid.render())
        self._last_map_publish_at = now

    @classmethod
    def _load_config(cls) -> MapperConfig:
        path = Path(__file__).resolve().parent / "calibration.json"
        try:
            calibration = MapperCalibration.load(path)
        except FileNotFoundError as error:
            raise RuntimeError(
                "No mapper calibration found. Run Calibrate Mapper first."
            ) from error
        except CalibrationSchemaError as error:
            raise RuntimeError(
                "Mapper calibration is outdated, incomplete, or inconsistent. "
                "Run Calibrate Mapper again."
            ) from error
        return cls._config_from_validated_calibration(calibration)

    @classmethod
    def _config_from_calibration(
        cls,
        data: dict[str, object],
    ) -> MapperConfig:
        """Validate an in-memory payload through the one calibration schema."""
        version = data.get("version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise _mapper_calibration_error(
                "Mapper calibration has no valid schema version."
            )
        if version != MapperCalibration.CURRENT_VERSION:
            raise RuntimeError(
                "Mapper calibration is outdated "
                f"(found v{version}, need v{MapperCalibration.CURRENT_VERSION}). "
                "Run Calibrate Mapper again."
            )
        try:
            calibration = MapperCalibration.from_dict(data)
        except CalibrationSchemaError as error:
            raise RuntimeError(
                "Mapper calibration is incomplete or inconsistent: "
                f"{error}. Run Calibrate Mapper again."
            ) from error
        return cls._config_from_validated_calibration(calibration)

    @staticmethod
    def _config_from_validated_calibration(
        calibration: MapperCalibration,
    ) -> MapperConfig:
        return MapperConfig(
            rotation_model=calibration.rotation_model,
            forward_model=calibration.forward_model,
            left_heading_sign=calibration.left_heading_sign,
            right_heading_sign=calibration.right_heading_sign,
        )
