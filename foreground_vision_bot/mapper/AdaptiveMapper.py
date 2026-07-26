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
from .AdaptiveHeadingSafety import classify_forward_heading_drift
from .AdaptiveMotionModel import (
    AdaptiveForwardOutcome,
    AdaptiveMotionModel,
    ForwardAssessment,
    TurnDirection,
)
from .AdaptiveMotionTracker import AdaptiveMotionTracker, MotionEstimate
from .AdaptiveRunMotionBaseline import AdaptiveRunMotionBaseline
from .AdaptiveTurnControl import (
    AdaptiveTurnController,
    AdaptiveTurnError,
    AdaptiveTurnResult,
)
from .Explorer import Explorer, ExplorerDecision
from .MapCatalog import MapCatalog, MapProfile
from .MapLogger import MapLogger
from .MinimapHeading import HeadingReading, MinimapHeadingDetector, signed_angle_delta
from .OccupancyGrid import FREE, UNKNOWN, OccupancyGrid, PoseIntegration
from .PangDetector import PangDetection, PangDetector
from .rl.ShadowPlanner import MapperShadowPlanner, ShadowDecision

StatusCallback = Callable[[str], None]
FrameCallback = Callable[[np.ndarray], None]
RecoveryCallback = Callable[[str, str, bool, bool], str | None]


class HeadingAcquisitionError(RuntimeError):
    """A stable heading could not be obtained without guessing."""


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
    maximum_recoverable_forward_heading_drift_degrees: float = 20.0
    post_turn_heading_settle_attempts: int = 3
    post_turn_heading_settle_seconds: float = 0.16
    post_turn_heading_stability_degrees: float = 4.0
    minimum_step_cells: float = 0.75
    maximum_step_cells: float = 1.25
    minimum_odometry_confidence: float = 0.42
    pang_threshold: float = 0.82
    save_every_steps: int = 5
    blocked_confirmations: int = 2
    forward_revalidation_attempts: int = 3
    heading_search_attempts: int = 3
    heading_search_pulse_seconds: float = 0.040

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
        if (
            self.maximum_recoverable_forward_heading_drift_degrees
            < self.maximum_forward_heading_drift_degrees
        ):
            raise ValueError(
                "recoverable forward heading drift must not be below the nominal limit"
            )
        if self.post_turn_heading_settle_attempts < 1:
            raise ValueError("post-turn heading settle attempts must be positive")
        if self.post_turn_heading_settle_seconds < 0.0:
            raise ValueError("post-turn heading settle time cannot be negative")
        if self.post_turn_heading_stability_degrees <= 0.0:
            raise ValueError("post-turn heading stability must be positive")
        if self.minimum_step_cells <= 0.0:
            raise ValueError("minimum_step_cells must be positive")
        if self.maximum_step_cells < self.minimum_step_cells:
            raise ValueError("maximum_step_cells must not be below the minimum")
        if self.save_every_steps < 1 or self.blocked_confirmations < 1:
            raise ValueError("mapper counts must be positive")
        if self.forward_revalidation_attempts < 0 or self.heading_search_attempts < 0:
            raise ValueError("mapper recovery counts cannot be negative")
        if not 0.015 <= self.heading_search_pulse_seconds <= 0.10:
            raise ValueError("heading search pulse must be between 15 and 100 ms")


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
    recovery_reason: str | None = None
    recovery_can_retry_in_place: bool = False
    recovery_requires_spawn_reset: bool = False


class AdaptiveMapper:
    """
    Autonomous mapper with online motion learning and no mandatory calibration.

    The first run starts from conservative defaults. Small closed-loop turn
    pulses continuously refine left/right timing, while successful nominal
    forward steps teach the expected optical-flow envelope. Uncertain movement
    still fails closed so map drift is not silently accumulated.
    """

    VERSION = "1.8-bounded-recovery-and-best-checkpoint"

    def __init__(
        self,
        bot: MapperBot,
        status_callback: StatusCallback | None = None,
        frame_callback: FrameCallback | None = None,
        config: MapperConfig | None = None,
        cancellation: CancellationToken | None = None,
        map_name: str | None = None,
        recovery_callback: RecoveryCallback | None = None,
        rl_shadow_enabled: bool = False,
        rl_policy_path: Path | None = None,
    ) -> None:
        if bot.keyboard is None:
            raise RuntimeError("Attach the Flyff window first.")

        self.bot = bot
        self.status_callback = status_callback or print
        self.frame_callback = frame_callback
        self.config = config or MapperConfig()
        self.cancellation = cancellation or CancellationToken()
        self.recovery_callback = recovery_callback

        self.map_catalog = MapCatalog()
        self.map_profile: MapProfile = self.map_catalog.get(map_name)
        self.map_dir = self.map_catalog.map_directory(self.map_profile.name)

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
        self.run_motion_baseline = AdaptiveRunMotionBaseline()
        canonical_exists = (self.map_dir / "map.json").is_file()
        self.grid, map_warning = OccupancyGrid.load(self.map_dir)
        legacy_import_allowed = self.map_catalog.legacy_import_allowed(
            self.map_profile.name
        )
        if (
            not canonical_exists
            and legacy_import_allowed
            and self.map_profile.name == self.map_catalog.default_name
        ):
            legacy_run = self.map_catalog.best_legacy_run()
            if legacy_run is not None:
                legacy_grid, legacy_warning = OccupancyGrid.load(legacy_run)
                if legacy_warning is None:
                    self.grid = legacy_grid
                    self.status_callback(
                        "Imported the richest legacy mapping run into the "
                        f"persistent map '{self.map_profile.name}': {legacy_run.name}."
                    )
                elif map_warning is None:
                    map_warning = legacy_warning
            self.map_catalog.mark_legacy_import_complete(self.map_profile.name)
        self.grid.metadata.map_name = self.map_profile.name
        if map_warning is not None:
            self.status_callback(map_warning)
        self.explorer = Explorer()

        self._position_known = False
        self._heading_known = False
        self._last_heading_uncertainty_deg: float | None = None
        self._blocked_observations: dict[tuple[int, int], int] = {}
        self._last_map_publish_at = 0.0
        self._map_publish_interval = 0.25
        self._forward_heading_settle_required = False
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
            recover_heading=self._recover_heading_after_turn,
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
        self.output_dir = (
            Path(__file__).resolve().parent
            / "mapping_runs"
            / self.map_profile.slug
            / run_id
        )
        self.logger = MapLogger(self.output_dir / "mapping_steps.csv")
        default_policy_path = (
            Path(__file__).resolve().parents[1]
            / "models"
            / "mapper_explorer_ppo.zip"
        )
        self.shadow_planner = MapperShadowPlanner(
            enabled=rl_shadow_enabled,
            model_path=rl_policy_path or default_policy_path,
            output_path=self.output_dir / "mapper_rl_shadow.jsonl",
        )
        if self.shadow_planner.warning is not None:
            self.status_callback(
                "Mapper RL shadow mode is unavailable; deterministic mapping "
                f"will continue. {self.shadow_planner.warning}"
            )

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
                ("save persistent and run maps", self._save_map_state),
                (
                    "save the motion snapshot",
                    lambda: self.motion_model.save_snapshot(
                        self.output_dir / "adaptive_motion_snapshot.json"
                    ),
                ),
                ("close the mapper RL shadow log", self.shadow_planner.close),
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
        known_cells = self.grid.known_cell_count()
        self.status_callback(
            f"Adaptive mapper starts in 5 seconds for map '{self.map_profile.name}'. "
            "Stand at the known spawn and keep the camera fixed. Behind-character "
            "and top-down views are both supported, but do not change view during "
            "a run. The selected map is persistent across runs; recovery never "
            "teleports automatically."
        )
        self.status_callback(
            f"Adaptive mapper {self.VERSION}; heading detector "
            f"{self.heading_detector.version()}; {self.motion_model.summary()}; "
            f"loaded {known_cells} known map cells."
        )
        self._publish_map_preview(force=True)
        for remaining in range(5, 0, -1):
            if self.cancellation.wait(1.0):
                self.cancellation.raise_if_cancelled()
            self.status_callback(f"Mapper starting in {remaining}...")

        self._initialize_spawn_with_recovery()
        self.grid.metadata.run_count += 1
        self.grid.metadata.termination_reason = None
        self.heading_detector.reset_fast()
        self._publish_map_preview(force=True)
        self._save_map_state()

        step = 0
        while not self.cancellation.cancelled:
            self.cancellation.raise_if_cancelled()
            shadow_decision = self.shadow_planner.recommend(self.grid)
            decision = self.explorer.decide(self.grid)
            if decision.action == "STOP":
                reason = "Mapping completed: no reachable unexplored frontier remains."
                self.grid.metadata.termination_reason = reason
                self.status_callback(reason)
                self._publish_map_preview(force=True)
                self._save_map_state()
                break

            candidate_step = step + 1
            try:
                if decision.action == "FORWARD":
                    result = self._execute_forward(candidate_step)
                else:
                    result = self._execute_turn(decision)
            except (AdaptiveTurnError, HeadingAcquisitionError) as error:
                reason = f"Mapper lost a reliable heading during {decision.action}: {error}"
                if self._attempt_manual_recovery(
                    reason,
                    can_retry_in_place=self._position_known,
                    requires_spawn_reset=not self._position_known,
                ):
                    continue
                self.grid.metadata.termination_reason = reason
                self.status_callback(reason)
                break

            step = candidate_step
            pang = self.pang.detect(result.frame_sample.frame)
            if pang.visible and result.pose_known:
                self.grid.add_pang_sighting(
                    self.grid.pose.x,
                    self.grid.pose.y,
                    pang.score,
                )

            self.shadow_planner.memory.update_after_step(
                actual_action=decision.action,
                forward_outcome=(
                    result.forward_assessment.outcome
                    if result.forward_assessment is not None
                    else None
                ),
                motion=result.motion,
                pose_known=result.pose_known,
                heading_known=self._heading_known,
                known_cell_count=self.grid.known_cell_count(),
            )
            shadow_outcome = (
                result.forward_assessment.outcome.value
                if result.forward_assessment is not None
                else "turn"
            )
            self.shadow_planner.record(
                step=step,
                actual_action=decision.action,
                actual_reason=decision.reason,
                recommendation=shadow_decision,
                outcome=shadow_outcome,
                pose_known=result.pose_known,
            )
            self._write_step(step, decision, result, pang, shadow_decision)
            self._report_step(step, decision, result, pang, shadow_decision)
            self._publish_map_preview()

            if step % self.config.save_every_steps == 0:
                self._save_map_state()
                self._save_motion_model()

            if result.recovery_reason is not None:
                if self._attempt_manual_recovery(
                    result.recovery_reason,
                    can_retry_in_place=result.recovery_can_retry_in_place,
                    requires_spawn_reset=result.recovery_requires_spawn_reset,
                ):
                    continue
                stop_reason = (
                    "Mapping stopped after recovery was declined or cancelled. "
                    f"Last issue: {result.recovery_reason}"
                )
                self.grid.metadata.termination_reason = stop_reason
                self.status_callback(stop_reason)
                break

            if result.stop_reason is not None:
                self.grid.metadata.termination_reason = result.stop_reason
                self.controller.stop()
                self._save_map_state()
                self._save_motion_model()
                self.status_callback(result.stop_reason)
                break

        return self.output_dir

    def _initialize_spawn_with_recovery(self) -> None:
        while True:
            self.cancellation.raise_if_cancelled()
            try:
                self._reset_pose_at_spawn()
                return
            except (AdaptiveTurnError, HeadingAcquisitionError) as error:
                reason = f"Could not initialize the known spawn heading: {error}"
                decision = self._request_recovery_decision(
                    reason,
                    can_retry_in_place=True,
                    requires_spawn_reset=True,
                )
                if decision not in {"retry", "spawn"}:
                    raise HeadingAcquisitionError(reason) from error

    def _reset_pose_at_spawn(self, *, manual: bool = False) -> None:
        _ = self._wait_for_frame_sample(not_before=monotonic())
        initial_heading = self._strict_heading("Known spawn heading", samples=15)
        spawn_x, spawn_y = self.grid.metadata.spawn
        self.grid.set_continuous_pose(
            float(spawn_x),
            float(spawn_y),
            initial_heading.angle_deg,
        )
        self._remember_heading(initial_heading)
        self._set_pose_reliability(
            position_known=True,
            heading_known=True,
            note="Known spawn position and stable heading confirmed.",
        )
        self._align_to_nearest_cardinal(initial_heading)
        self._blocked_observations.clear()
        self.run_motion_baseline.clear()
        self._forward_heading_settle_required = False
        self.heading_detector.reset_fast()
        if manual:
            self.grid.metadata.manual_spawn_resets += 1
        self.status_callback(
            f"Mapper pose reset to the known spawn on '{self.map_profile.name}' "
            "without changing the persistent map."
        )

    def _attempt_manual_recovery(
        self,
        reason: str,
        *,
        can_retry_in_place: bool,
        requires_spawn_reset: bool,
    ) -> bool:
        self.controller.stop()
        self._save_map_state()
        self._save_motion_model()
        while not self.cancellation.cancelled:
            decision = self._request_recovery_decision(
                reason,
                can_retry_in_place=can_retry_in_place,
                requires_spawn_reset=requires_spawn_reset,
            )
            if decision == "stop" or decision is None:
                return False
            try:
                if decision == "spawn":
                    self._reset_pose_at_spawn(manual=True)
                    self._publish_map_preview(force=True)
                    self._save_map_state()
                    return True
                if decision == "retry" and can_retry_in_place:
                    reading = self._strict_heading(
                        "Heading recovery in place",
                        samples=15,
                    )
                    self.grid.set_heading_degrees(reading.angle_deg)
                    self._remember_heading(reading)
                    self._set_pose_reliability(
                        position_known=True,
                        heading_known=True,
                        note="Heading reacquired at the last confirmed position.",
                    )
                    self.heading_detector.reset_fast()
                    self.grid.metadata.heading_recovery_successes += 1
                    self._publish_map_preview(force=True)
                    return True
            except (AdaptiveTurnError, HeadingAcquisitionError) as error:
                reason = f"Recovery attempt still could not establish heading: {error}"
                can_retry_in_place = self._position_known
                requires_spawn_reset = not self._position_known
                continue
        return False

    def _request_recovery_decision(
        self,
        reason: str,
        *,
        can_retry_in_place: bool,
        requires_spawn_reset: bool,
    ) -> str | None:
        self.status_callback(
            "Mapper paused safely with all movement keys released. "
            f"Recovery needed: {reason}"
        )
        if self.recovery_callback is None:
            return None
        return self.recovery_callback(
            self.map_profile.name,
            reason,
            can_retry_in_place,
            requires_spawn_reset,
        )

    def _save_map_state(self) -> None:
        self.grid.metadata.map_name = self.map_profile.name
        self.grid.save(self.map_dir)
        self.grid.save(self.output_dir)

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
        self._forward_heading_settle_required = True
        if self.grid.pose.heading_index != target_index:
            raise AdaptiveTurnError(
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

    def _settle_heading_before_forward(self) -> HeadingReading:
        previous = self._strict_heading(
            "Post-turn heading settle baseline",
            samples=9,
        )
        self.grid.set_heading_degrees(previous.angle_deg)
        self._remember_heading(previous)
        for attempt in range(1, self.config.post_turn_heading_settle_attempts + 1):
            if self.cancellation.wait(self.config.post_turn_heading_settle_seconds):
                self.cancellation.raise_if_cancelled()
            current = self._strict_heading(
                f"Post-turn heading settle {attempt}",
                samples=9,
            )
            drift = signed_angle_delta(current.angle_deg, previous.angle_deg)
            self.grid.set_heading_degrees(current.angle_deg)
            self._remember_heading(current)
            if abs(drift) <= self.config.post_turn_heading_stability_degrees:
                self._forward_heading_settle_required = False
                if attempt > 1:
                    self.status_callback(
                        "Post-turn heading settled before forward movement at "
                        f"{current.angle_deg:.1f}°."
                    )
                return current
            self.status_callback(
                "Post-turn heading was still settling "
                f"({previous.angle_deg:.1f}° -> {current.angle_deg:.1f}°, "
                f"{drift:+.1f}°); waiting before forward movement "
                f"({attempt}/{self.config.post_turn_heading_settle_attempts})."
            )
            previous = current
        raise HeadingAcquisitionError(
            "Heading continued to drift after the completed turn. The character "
            "has not moved forward, so recovery in place is safe."
        )

    def _execute_forward(self, step: int) -> _StepResult:
        if self._forward_heading_settle_required:
            self._settle_heading_before_forward()
        before = self._wait_for_frame_sample(not_before=monotonic())
        start_heading = self.grid.continuous_pose.heading_deg
        travel_heading_index = self.grid.pose.heading_index
        step_dx, step_dy = self.grid.DIRECTIONS[travel_heading_index]
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
                recovery_reason=(
                    "Probable teleport or scene discontinuity detected after a "
                    "forward command. Return manually to the known spawn before "
                    "resuming; the bot will not teleport itself."
                ),
                recovery_requires_spawn_reset=True,
            )

        strict_heading = self._strict_heading(
            "Heading after forward movement",
            samples=11,
        )
        self._remember_heading(strict_heading)
        self._forward_heading_settle_required = False
        self._set_pose_reliability(
            heading_known=True,
            note="Post-forward heading confirmed; position awaits visual validation.",
        )
        heading_change = signed_angle_delta(strict_heading.angle_deg, start_heading)

        assessment = self._assess_forward_motion(
            motion,
            held_seconds=timing.held_seconds,
            heading_index=travel_heading_index,
        )
        if assessment.outcome is AdaptiveForwardOutcome.UNCERTAIN:
            (
                after,
                fast_heading,
                motion,
                assessment,
                retry_debug_path,
            ) = self._revalidate_forward_observation(
                step=step,
                before=before,
                after=after,
                timing=timing,
                initial_motion=motion,
                initial_assessment=assessment,
                heading_index=travel_heading_index,
            )
            motion_debug_path = retry_debug_path or motion_debug_path

        if assessment.outcome is AdaptiveForwardOutcome.BLOCKED:
            motion_debug_path = motion_debug_path or self._save_motion_debug(
                step, before, after, motion
            )
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

        heading_drift_mode = classify_forward_heading_drift(
            heading_change,
            nominal_limit=self.config.maximum_forward_heading_drift_degrees,
            recoverable_limit=(
                self.config.maximum_recoverable_forward_heading_drift_degrees
            ),
        )
        if (
            assessment.outcome is AdaptiveForwardOutcome.MOVED
            and heading_drift_mode == "unsafe"
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
                distance_cells=assessment.distance_cells,
                integration=None,
                pose_known=False,
                motion_debug_path=motion_debug_path,
                recovery_reason=(
                    "Forward travel changed heading by "
                    f"{heading_change:+.1f}°, beyond the recoverable curved-step "
                    "limit. Return manually to the known spawn before continuing."
                ),
                recovery_requires_spawn_reset=True,
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
                recovery_reason=(
                    "Forward travel remained visually uncertain after fresh-frame "
                    "revalidation, so the new position was not guessed. "
                    f"Reason: {assessment.reason}. Return manually to the known "
                    "spawn, then resume the persistent map."
                ),
                recovery_requires_spawn_reset=True,
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
                recovery_reason=(
                    "Adaptive forward distance failed its safety gate. Return "
                    "manually to the known spawn before resuming."
                ),
                recovery_requires_spawn_reset=True,
                motion_debug_path=motion_debug_path,
            )

        self._blocked_observations.pop(target_cell, None)
        midpoint_heading = (start_heading + 0.5 * heading_change) % 360.0
        integration_confidence = assessment.confidence
        if heading_drift_mode == "recoverable":
            integration_confidence *= 0.75
            self.status_callback(
                "Forward step included moderate heading drift "
                f"({heading_change:+.1f}°); integrating along the midpoint heading "
                "with reduced confidence instead of discarding the confirmed move."
            )
        self.grid.set_heading_degrees(midpoint_heading)
        integration = self.grid.integrate_forward(
            distance_cells,
            confidence=integration_confidence,
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
                recovery_reason=(
                    "Adaptive motion could not be integrated safely "
                    f"({integration.reason}). Return manually to the known spawn "
                    "before resuming."
                ),
                recovery_requires_spawn_reset=True,
            )

        self.motion_model.observe_forward(motion.directional_flow)
        self.run_motion_baseline.observe(
            travel_heading_index,
            motion.directional_flow,
        )
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

    def _assess_forward_motion(
        self,
        motion: MotionEstimate,
        *,
        held_seconds: float,
        heading_index: int,
    ) -> ForwardAssessment:
        contact = self.run_motion_baseline.assess_contact(
            heading_index,
            motion.directional_flow,
        )
        if contact.likely_contact:
            observed = max(0.0, float(motion.directional_flow.magnitude_px))
            return ForwardAssessment(
                outcome=AdaptiveForwardOutcome.BLOCKED,
                reliable=True,
                distance_cells=0.0,
                confidence=contact.confidence,
                expected_flow_px=contact.baseline_flow_px,
                observed_flow_px=observed,
                flow_ratio=contact.flow_ratio,
                reason=contact.reason or "partial obstacle contact likely",
            )
        return self.motion_model.assess_forward(
            motion.directional_flow,
            change_score=motion.change_score,
            held_seconds=held_seconds,
        )

    def _revalidate_forward_observation(
        self,
        *,
        step: int,
        before: FrameSample,
        after: FrameSample,
        timing: KeyPressTiming,
        initial_motion: MotionEstimate,
        initial_assessment: ForwardAssessment,
        heading_index: int,
    ) -> tuple[
        FrameSample,
        HeadingReading | None,
        MotionEstimate,
        ForwardAssessment,
        str | None,
    ]:
        """Retry vision only; never issue another movement command."""
        latest_after = after
        latest_motion = initial_motion
        latest_assessment = initial_assessment
        latest_fast = self.heading_detector.read_fast(after.frame)
        debug_path: str | None = None

        for attempt in range(1, self.config.forward_revalidation_attempts + 1):
            self.status_callback(
                "Forward evidence was uncertain; rechecking the completed step "
                f"with a fresh frame ({attempt}/"
                f"{self.config.forward_revalidation_attempts}) without moving."
            )
            if self.cancellation.wait(0.12):
                self.cancellation.raise_if_cancelled()
            candidate = self._wait_for_frame_sample(
                after_identity=latest_after.identity,
                generation=before.generation,
                not_before=monotonic(),
            )
            candidate_motion = self.tracker.compare(before.frame, candidate.frame)
            candidate_assessment = self._assess_forward_motion(
                candidate_motion,
                held_seconds=timing.held_seconds,
                heading_index=heading_index,
            )
            debug_path = self._save_motion_debug(
                step,
                before,
                candidate,
                candidate_motion,
                suffix=f"recheck_{attempt}",
            )
            latest_after = candidate
            latest_fast = self.heading_detector.read_fast(candidate.frame)
            if candidate_motion.teleport_likely:
                continue
            latest_motion = candidate_motion
            latest_assessment = candidate_assessment
            if candidate_assessment.outcome is not AdaptiveForwardOutcome.UNCERTAIN:
                self.status_callback(
                    "Fresh-frame revalidation resolved the forward step as "
                    f"{candidate_assessment.outcome.value}."
                )
                break

        return (
            latest_after,
            latest_fast,
            latest_motion,
            latest_assessment,
            debug_path,
        )

    def _save_motion_debug(
        self,
        step: int,
        before: FrameSample,
        after: FrameSample,
        motion: MotionEstimate,
        *,
        suffix: str | None = None,
    ) -> str:
        path = self.tracker.save_diagnostics(
            self.output_dir / "motion_debug",
            prefix=(
                f"step_{step:04d}_forward"
                + (f"_{suffix}" if suffix else "")
            ),
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

        if blocked_recorded and target_value != FREE:
            self.status_callback(
                "Obstacle confirmed and marked at "
                f"({target_cell[0]}, {target_cell[1]}); replanning around it."
            )

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

    def _recover_heading_after_turn(
        self,
        direction: TurnDirection,
        context: str,
        samples: int,
    ) -> HeadingReading:
        """Move the arrow out of an ambiguous orientation without translating."""
        last_error: HeadingAcquisitionError | None = None
        for attempt in range(1, self.config.heading_search_attempts + 1):
            self.cancellation.raise_if_cancelled()
            duration = min(
                0.10,
                self.config.heading_search_pulse_seconds * (1.0 + 0.25 * (attempt - 1)),
            )
            self.status_callback(
                f"{context}: heading search {attempt}/"
                f"{self.config.heading_search_attempts}, {direction.value} "
                f"{duration * 1000.0:.0f} ms."
            )
            if direction is TurnDirection.LEFT:
                self.controller.turn_left(duration)
            else:
                self.controller.turn_right(duration)
            if self.cancellation.wait(self.config.turn_settle_seconds + 0.10):
                self.cancellation.raise_if_cancelled()
            try:
                reading = self._strict_heading(
                    f"{context}: heading search {attempt}",
                    samples=max(9, int(samples)),
                )
            except HeadingAcquisitionError as error:
                last_error = error
                continue
            self.grid.metadata.heading_recovery_successes += 1
            self.status_callback(
                f"{context}: heading reacquired at {reading.angle_deg:.1f}° "
                "after a bounded search pulse."
            )
            return reading

        if last_error is not None:
            raise last_error
        raise HeadingAcquisitionError(
            f"{context}: heading search had no permitted attempts"
        )

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
        raise HeadingAcquisitionError(
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
        shadow: ShadowDecision,
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
                "map_name": self.map_profile.name,
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
                "rl_shadow_enabled": shadow.enabled,
                "rl_shadow_action": shadow.action,
                "rl_shadow_agrees": (
                    shadow.enabled and shadow.action == decision.action
                ),
                "rl_shadow_status": shadow.status,
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
                "recovery_reason": result.recovery_reason or "",
                "recovery_requires_spawn_reset": (
                    result.recovery_requires_spawn_reset
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
        shadow: ShadowDecision,
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
        shadow_text = (
            f" rl_shadow={shadow.action}"
            if shadow.enabled and shadow.action
            else ""
        )
        self.status_callback(
            f"map step={step} pose=({self.grid.continuous_pose.x:.2f},"
            f"{self.grid.continuous_pose.y:.2f}) "
            f"heading={self.grid.continuous_pose.heading_deg:.1f}° "
            f"action={decision.action} motion={motion_text} camera={camera_text} "
            f"strict_heading={strict_text} pose_known={result.pose_known} "
            f"pang={pang.visible}{shadow_text}; {self.motion_model.summary()}"
        )

    def _publish_map_preview(self, force: bool = False) -> None:
        if self.frame_callback is None:
            return
        now = monotonic()
        if not force and now - self._last_map_publish_at < self._map_publish_interval:
            return
        self.frame_callback(self.grid.render_overview())
        self._last_map_publish_at = now


# Keep the public name expected by RuntimeController and existing imports.
Mapper = AdaptiveMapper
