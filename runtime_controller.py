from __future__ import annotations

import traceback
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import cast

from assets.Assets import MobInfo
from capture_service import CaptureService, FrameSource
from libs.WindowCapture import WindowCapture
from mapper import Mapper
from position import (
    NativeDiagnosticProgress,
    NativeDiagnosticReport,
    NativePointerSnapshotError,
    PointerRecoveryHints,
    run_native_diagnostic,
)
from preview_service import PreviewService
from recording_sink import RecordingOwnership, RecordingSink, _RuntimeMetadata
from runtime_bus import FarmingSessionSnapshot, RuntimeBus
from worker_manager import (
    CancellationToken,
    WorkerKind,
    WorkerManager,
    WorkerSnapshot,
)


class _RecoveryLog:
    """Best-effort persistent text log for pointer/profile recovery."""

    def __init__(self, label: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        directory = (
            Path(__file__).resolve().parent
            / "training_logs"
            / "native_recovery"
        )
        self.path: Path | None = directory / f"{label}-{timestamp}.log"
        self._lock = Lock()
        try:
            directory.mkdir(parents=True, exist_ok=True)
            self.write(f"Recovery log started: {label}")
        except OSError:
            self.path = None

    def write(self, message: str) -> None:
        if self.path is None:
            return
        line = (
            datetime.now(timezone.utc).isoformat()
            + " "
            + str(message).replace("\r", " ").replace("\n", " ")
            + "\n"
        )
        try:
            with self._lock:
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(line)
                    stream.flush()
        except OSError:
            # Recovery must not fail because a diagnostic log cannot be written.
            pass


class RuntimeController:
    """Single owner for capture and control-worker lifecycle."""

    # The standalone recovery tester allows a full 20-minute scan. Keep the
    # production recovery bounds aligned so a valid cross-machine scan is not
    # cut off merely because it runs through the GUI or RL startup path.
    MAX_NATIVE_DIAGNOSTIC_TIMEOUT_SECONDS: float = 1200.0
    POINTER_RECOVERY_TIMEOUT_SECONDS: float = 1200.0
    STARTUP_POINTER_RECOVERY_TIMEOUT_SECONDS: float = 1200.0
    PROFILE_VALIDATION_TIMEOUT_SECONDS: float = 120.0

    def __init__(self, bot, bus: RuntimeBus) -> None:
        self.bot = bot
        self.bus = bus
        self.workers = WorkerManager(bus)
        self.capture = CaptureService(
            self.workers,
            bus,
            lambda handle: cast(FrameSource, WindowCapture(handle)),
            preview_enabled=lambda: False,
        )
        self.preview = PreviewService(
            self.workers,
            bus,
            self.capture,
            bot.build_preview,
            cancellable_preview_builder=getattr(
                bot,
                "build_preview_cancellable",
                None,
            ),
        )
        self._next_control_session_id = 1
        self._control_session_id: int | None = None
        self._next_diagnostic_session_id = 1
        self._diagnostic_session_id: int | None = None
        self._diagnostic_recovery_requested = False
        self._shutdown_lock = Lock()
        self._shutdown_requested = False
        self._shutdown_finalized = False
        self._shutdown_results = {kind: True for kind in WorkerKind}
        self._shutdown_timed_out: tuple[WorkerKind, ...] = ()
        self.recording: RecordingSink | None = None
        self._attached_hwnd: int | None = None
        self._attached_title: str | None = None

    @property
    def capture_active(self) -> bool:
        return self.workers.is_active(WorkerKind.CAPTURE)

    @property
    def control_active(self) -> bool:
        return self.workers.is_active(WorkerKind.CONTROL)

    def control_snapshot(self) -> WorkerSnapshot | None:
        return self.workers.snapshot(WorkerKind.CONTROL)

    @property
    def diagnostic_active(self) -> bool:
        return self.workers.is_active(WorkerKind.DIAGNOSTIC)

    def diagnostic_snapshot(self) -> WorkerSnapshot | None:
        return self.workers.snapshot(WorkerKind.DIAGNOSTIC)

    @property
    def control_session_id(self) -> int | None:
        return self._control_session_id

    @property
    def diagnostic_session_id(self) -> int | None:
        return self._diagnostic_session_id

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_requested

    @property
    def shutdown_finalized(self) -> bool:
        return self._shutdown_finalized

    @property
    def shutdown_timed_out(self) -> tuple[WorkerKind, ...]:
        return self._shutdown_timed_out

    def attach(self, window_handle: int, title: str | None = None) -> int:
        if self.control_active:
            raise RuntimeError(
                "Cannot reattach while a control task is active. Stop it first."
            )
        if self.diagnostic_active:
            raise RuntimeError(
                "Cannot reattach while native diagnostics are active. Stop them first."
            )
        if not self.preview.stop(3.0):
            raise RuntimeError("Previous preview worker did not stop.")
        self.bot.release_input()
        generation = self.capture.attach(window_handle)
        self._attached_hwnd = window_handle
        self._attached_title = title
        try:
            self.bot.prepare_window(window_handle, self.bus, self.capture)
            self.preview.start()
        except Exception:
            self.preview.stop(3.0)
            self.capture.stop(5.0)
            raise
        return generation

    def start_rl(self, mode: str) -> None:
        def run(token: CancellationToken):
            from farming.trainer import (
                dry_run_native_farming,
                run_native_farming_agent,
                train_native_farming,
                validate_native_farming_data,
            )

            report = self._reporter("rl_status", "rl")
            # Operational-feedback recording is mandatory, not optional
            # (docs/PROJECT_GOALS.md section 6): bot live farming/
            # training => recording active, always. If a USER already
            # started one (rule D), reuse it -- never a second writer.
            # If none is active, auto-start one; if the shared sink
            # cannot start, farming/training must not start either
            # (fail closed, not silently unrecorded).
            recording_started_here = False
            if mode in ("train", "agent"):
                if self.recording is not None and self.recording.is_running:
                    report("Reusing the already-active recording session.")
                else:
                    try:
                        self.start_recording(started_by="RUNTIME_AUTO")
                        recording_started_here = True
                        report("Operational-feedback recording started.")
                    except Exception as error:  # noqa: BLE001 - reported, not swallowed.
                        report(
                            f"Recording could not start; {mode} will not begin. "
                            f"{type(error).__name__}: {error}"
                        )
                        return {"started": False, "reason": "recording_unavailable"}
            try:
                if not self._prepare_native_pointer_startup(
                    token,
                    report,
                    verbose=(mode != "train"),
                ):
                    return {"started": False, "reason": "native_pointer_unavailable"}
                if mode == "train":
                    return train_native_farming(
                        self.bot,
                        status_callback=report,
                        cancellation=token,
                        session_stats_callback=self._farming_stats_reporter(
                            "Training"
                        ),
                    )
                if mode == "agent":
                    run_native_farming_agent(
                        self.bot,
                        status_callback=report,
                        cancellation=token,
                        session_stats_callback=self._farming_stats_reporter(
                            "Agent"
                        ),
                        on_runtime_event=(
                            self.recording.add_runtime_event
                            if self.recording is not None
                            else None
                        ),
                    )
                    return None
                if mode == "dry-run":
                    dry_run_native_farming(
                        self.bot,
                        status_callback=report,
                        cancellation=token,
                    )
                    return None
                if mode == "validate-data":
                    return validate_native_farming_data(
                        self.bot,
                        status_callback=report,
                        cancellation=token,
                    )
                raise ValueError(f"Unknown RL mode: {mode}")
            finally:
                self.bot.stop()
                # Rule E: only finalize a recording THIS call started.
                # A user-started recording keeps running until the user
                # presses Stop Recording, even if farming/training ends.
                if recording_started_here:
                    self.stop_recording()

        self._start_control(f"rl-{mode}", run)

    def start_recording(self, *, started_by: str = "USER") -> RecordingSink:
        """One recording session, as a passive sink over the bot's own
        already-attached native reader -- never a second scanner
        (docs/architecture/RECORDING_TELEMETRY_AND_ARCHIVES.md section
        1a). Raises if one is already active, or if not attached yet."""
        if self.recording is not None and self.recording.is_running:
            raise RuntimeError("A recording is already active.")
        service = getattr(self.bot, "native_process_service", None)
        position_provider = getattr(self.bot, "position_provider", None)
        monster_provider = getattr(self.bot, "monster_provider", None)
        if service is None or position_provider is None or monster_provider is None:
            raise RuntimeError("Attach to the FlyFF window first.")
        attach_policy = getattr(service, "attach_policy", None)
        metadata = _RuntimeMetadata(
            attach_policy_name=getattr(attach_policy, "name", None),
            presence_validation_source=getattr(
                service, "presence_validation_source", None
            ),
        )
        self.recording = RecordingSink(
            native_process_service=service,
            position_provider=position_provider,
            monster_provider=monster_provider,
            ownership=RecordingOwnership(started_by=started_by),
            character_name=self._attached_title or "player",
            metadata=metadata,
        )
        return self.recording

    def stop_recording(self) -> Path | None:
        if self.recording is None:
            return None
        output_zip = self.recording.stop()
        self.recording = None
        return output_zip

    def _prepare_native_pointer_startup(
        self,
        token: CancellationToken,
        report: Callable[[str], None],
        *,
        verbose: bool = True,
    ) -> bool:
        """Resolve expected stale/null native state inside the control worker."""

        service = getattr(self.bot, "native_process_service", None)
        if service is None:
            return True
        recovery_log = _RecoveryLog("startup-recovery")
        recovery_log.write(
            f"mode={'verbose' if verbose else 'training'}"
        )
        # Fast-restore diagnostics (docs/PROJECT_GOALS.md-adjacent live
        # acceptance finding: a user-run relaunch against the same,
        # still-open FlyFF client fell back to full discovery instead of
        # the expected RecoveredNativeProfile fast restore). These are
        # transition/event logs, not per-frame -- state-change points
        # only, so the evidence stays readable.
        attach_policy = getattr(service, "attach_policy", None)
        profile_path = getattr(service, "recovery_profile_path", None)
        profile_exists = profile_path is not None and Path(profile_path).is_file()
        recovery_log.write(
            f"attach_policy={getattr(attach_policy, 'name', 'unknown')} "
            f"recovered_profile_path={profile_path} "
            f"profile_exists={profile_exists}"
        )
        try:
            service.read_pointer_snapshot()
            recovery_log.write(
                "Existing in-process native reader validated "
                "(presence_validation_source="
                f"{getattr(service, 'presence_validation_source', 'unproven')})."
            )
            return True
        except NativePointerSnapshotError:
            recovery_log.write(
                "Existing in-process native reader was unavailable or stale."
            )

        profile_hints = self._pointer_recovery_hints(
            cancellation=token,
            health_attempts=1,
        )
        profile_deadline = monotonic() + self.PROFILE_VALIDATION_TIMEOUT_SECONDS

        def profile_status(message: str) -> None:
            recovery_log.write(message)
            normalized = message.casefold()
            if (
                verbose
                or normalized.startswith("checking the last known")
                or "validated" in normalized
                or "full recovery" in normalized
            ):
                report(message)
            self.bus.heartbeat("control")

        restore_profile = getattr(service, "try_restore_persisted_profile", None)
        if callable(restore_profile):
            recovery_log.write(
                "profile_load_attempted=True" if profile_exists
                else "profile_load_attempted=True (no file on disk)"
            )
            try:
                restore_ok = restore_profile(
                    hints=profile_hints,
                    cancellation=token,
                    deadline=profile_deadline,
                    status_callback=profile_status,
                )
                recovery_log.write(
                    f"restore_validation_result={restore_ok} "
                    f"restore_mode={getattr(service, 'last_profile_restore_mode', 'unknown')} "
                    f"rejection_reason={getattr(service, 'last_profile_restore_error', None)}"
                )
                if restore_ok:
                    snapshot = service.read_pointer_snapshot()
                    presence_source = getattr(
                        service, "presence_validation_source", "unproven"
                    )
                    recovery_log.write(
                        f"presence_validation_source={presence_source} "
                        f"presence_validated={presence_source != 'unproven'}"
                    )
                    report(
                        "Native startup preflight restored the last known "
                        "validated profile: "
                        f"player=0x{snapshot.player_base:X}."
                    )
                    return True
            except Exception as error:  # noqa: BLE001 - optional fast path.
                recovery_log.write(
                    f"restore_validation_result=False "
                    f"rejection_reason={type(error).__name__}: {error}"
                )
                report(
                    "Last known native profile could not be used; full recovery "
                    f"will run. {type(error).__name__}: {error}"
                )
        else:
            recovery_log.write("profile_load_attempted=False (service has no restore method)")

        recovery_log.write(
            f"fallback_reason=restore_did_not_apply full_discovery_started=True"
        )
        report("Native pointers unavailable; running full startup recovery.")
        recovery_log.write("Native pointers unavailable; running full startup recovery.")
        if recovery_log.path is not None:
            report(f"Native recovery log: {recovery_log.path}")
        timeout = self.STARTUP_POINTER_RECOVERY_TIMEOUT_SECONDS
        deadline = monotonic() + timeout

        def publish(progress) -> None:
            message = str(progress.message)
            phase = str(getattr(progress, "phase", ""))
            recovery_log.write(f"[{phase or 'recovery'}] {message}")
            if verbose or phase in {
                "success",
                "cache_hit",
                "cancelled",
                "deadline",
                "error",
            }:
                report(message)
            elif (
                message.startswith("Dynamically recovered the authoritative")
                or message.startswith("Validated target scan read")
                or message.startswith("Authoritative actor scan")
                or message.startswith("Authoritative actor candidate")
                or message.startswith("Reusing the last validated authoritative")
                or message.startswith("Found one selected player")
                or message.startswith("Native player pointer recovered")
            ):
                report(message)
            self.bus.heartbeat("control")

        try:
            result = service.recover_pointers(
                persist=False,
                cancellation=token,
                deadline=deadline,
                timeout_seconds=timeout,
                status_callback=publish,
                hints=self._pointer_recovery_hints(
                    cancellation=token,
                    health_attempts=8,
                ),
            )
        except Exception as error:  # noqa: BLE001 - expected startup boundary.
            recovery_log.write(
                "Startup recovery failed: "
                f"{type(error).__name__}: {error}"
            )
            report(
                "Native pointer startup recovery could not complete: "
                f"{type(error).__name__}: {error}. No input was activated."
            )
            return False
        if not result.succeeded or not result.applied:
            recovery_log.write(
                "Startup recovery did not apply: "
                f"outcome={result.outcome.value}."
            )
            report(
                "Native pointers remain unavailable after bounded recovery "
                f"({result.outcome.value}). No input was activated."
            )
            return False
        try:
            snapshot = service.read_pointer_snapshot()
        except NativePointerSnapshotError as error:
            recovery_log.write(
                "Startup verification failed: " + str(error)
            )
            report(
                "Recovered native pointers did not pass startup verification: "
                f"{error}. No input was activated."
            )
            return False
        if snapshot.mode == "independent":
            recovery_log.write(
                "Startup recovery completed in independent mode: "
                f"player=0x{snapshot.player_base:X}."
            )
            report(
                "Native pointer startup preflight ready in independent mode: "
                f"player=0x{snapshot.player_base:X}; world pointer not required."
            )
        else:
            report(
                "Native pointer startup preflight ready: "
                f"player=0x{snapshot.player_base:X}, world=0x{snapshot.world_base:X}."
            )
        return True

    def start_mapper(
        self,
        map_name: str,
        *,
        rl_shadow_enabled: bool = False,
    ) -> None:
        def run(token: CancellationToken):
            mapper = Mapper(
                self.bot,
                status_callback=self._reporter(
                    "mapper_status",
                    "mapper",
                ),
                frame_callback=lambda frame: self.bus.publish_latest(
                    "map_frame", frame
                ),
                cancellation=token,
                map_name=map_name,
                recovery_callback=lambda selected_map, reason, can_retry, needs_spawn: (
                    self.bus.request_mapper_recovery(
                        map_name=selected_map,
                        reason=reason,
                        can_retry_in_place=can_retry,
                        requires_spawn_reset=needs_spawn,
                        cancellation_event=token.event,
                    )
                ),
                rl_shadow_enabled=rl_shadow_enabled,
            )
            return mapper.run()

        self._start_control("mapper", run)

    def start_manual_mapper(self, map_name: str) -> None:
        """Track user-controlled native movement without sending input."""

        def run(token: CancellationToken):
            from mapper.ManualDriveMapper import ManualDriveMapper

            mapper = ManualDriveMapper(
                self.bot,
                status_callback=self._reporter(
                    "mapper_status",
                    "manual-mapper",
                ),
                frame_callback=lambda frame: self.bus.publish_latest(
                    "map_frame", frame
                ),
                cancellation=token,
                map_name=map_name,
            )
            return mapper.run()

        self._start_control("manual-mapper", run)

    def publish_map_preview(self, map_name: str) -> bool:
        """Render the current map with today's configured local radius.

        Loading the persisted preview PNG can preserve an obsolete local-map
        radius forever. Rendering from the saved occupancy grid keeps idle GUI
        previews consistent with live mapping and the native actor overlay.
        """
        from mapper.CoordinateMapper import load_mapper_config
        from mapper.MapCatalog import MapCatalog
        from mapper.OccupancyGrid import OccupancyGrid

        catalog = MapCatalog()
        profile = catalog.get(map_name)
        directory = catalog.map_directory(profile.name)
        grid, warning = OccupancyGrid.load(directory)
        if warning is not None:
            return False
        if not (directory / "map.json").is_file():
            return False
        radius = load_mapper_config().local_map_radius_cells
        image = grid.render_dashboard(local_radius_cells=radius)
        self.bus.publish_latest("map_frame", image)
        return True

    def apply_manual_map_edits(self, map_name: str, edits):
        """Persist user-authored occupancy cells while control is stopped."""
        if self.control_active:
            raise RuntimeError(
                "Stop mapping or RL control before editing map cells. "
                "This prevents the mapper checkpoint from overwriting the edit."
            )

        from mapper.CoordinateMapper import load_mapper_config
        from mapper.ManualMapEditor import apply_manual_edits
        from mapper.MapCatalog import MapCatalog
        from mapper.OccupancyGrid import OccupancyGrid

        catalog = MapCatalog()
        profile = catalog.get(map_name)
        directory = catalog.map_directory(profile.name)
        if not (directory / "map.json").is_file():
            raise RuntimeError(
                f"'{profile.name}' does not have a saved occupancy map yet."
            )
        grid, warning = OccupancyGrid.load(directory)
        if warning is not None:
            raise RuntimeError(warning)
        summary = apply_manual_edits(grid, dict(edits))
        radius = load_mapper_config().local_map_radius_cells
        grid.save(directory, preview_local_radius_cells=radius)
        self.bus.publish_latest(
            "map_frame",
            grid.render_dashboard(local_radius_cells=radius),
        )
        return summary

    def start_calibration(self, *, visual_confirmation: bool) -> None:
        # Legacy rollback path. Ordinary mapping no longer imports or requires
        # the calibration stack.
        from mapper import RotationCalibrator

        def run(token: CancellationToken):
            confirmation: Callable[..., bool | None] | None = None
            if visual_confirmation:
                confirmation = lambda *args, **kwargs: (
                    self.bus.request_heading_confirmation(
                        *args,
                        **kwargs,
                        cancellation_event=token.event,
                    )
                )
            calibrator = RotationCalibrator(
                self.bot,
                status_callback=self._reporter(
                    "mapper_status",
                    "calibration",
                ),
                visual_confirmation_callback=confirmation,
                cancellation=token,
            )
            return calibrator.run(manual=True)

        self._start_control("calibration", run)

    def start_native_diagnostic(
        self,
        *,
        recover: bool = False,
        timeout: float = 1.0,
        player_current_hp: int | None = None,
        player_max_hp: int | None = None,
    ) -> int:
        """Start bounded native health/recovery work without blocking the caller."""

        bounded_timeout = float(timeout)
        if (
            not isfinite(bounded_timeout)
            or bounded_timeout <= 0.0
            or bounded_timeout > self.MAX_NATIVE_DIAGNOSTIC_TIMEOUT_SECONDS
        ):
            raise ValueError(
                "timeout must be finite, positive, and no greater than "
                f"{self.MAX_NATIVE_DIAGNOSTIC_TIMEOUT_SECONDS:.0f} seconds"
            )
        if recover and self.control_active:
            raise RuntimeError(
                "Stop the active control task before recovering native pointers."
            )

        session_id = self._next_diagnostic_session_id
        self._next_diagnostic_session_id += 1
        previous_session_id = self._diagnostic_session_id
        previous_recovery_requested = self._diagnostic_recovery_requested
        self._diagnostic_session_id = session_id
        self._diagnostic_recovery_requested = bool(recover)
        def run(token: CancellationToken) -> NativeDiagnosticReport:
            deadline = monotonic() + bounded_timeout
            recovery_log = (
                _RecoveryLog("manual-recovery") if recover else None
            )
            if recovery_log is not None and recovery_log.path is not None:
                self.bus.log(
                    f"Native recovery log: {recovery_log.path}",
                    "msg_blue",
                )

            def publish(progress: NativeDiagnosticProgress) -> None:
                if recovery_log is not None:
                    recovery_log.write(
                        f"[{getattr(progress, 'phase', 'recovery')}] "
                        f"{progress.message}"
                    )
                self.bus.publish_status(
                    "native_diagnostic_status",
                    progress.message,
                    session_id=session_id,
                )
                self.bus.log(progress.message, "msg_blue")
                self.bus.heartbeat("diagnostic")

            recovery_hints = self._pointer_recovery_hints(
                player_current_hp=player_current_hp,
                player_max_hp=player_max_hp,
                cancellation=token,
                health_attempts=8,
            )
            if recover:
                if (
                    recovery_hints.player_current_hp is None
                    or recovery_hints.player_max_hp is None
                ):
                    self.bus.log(
                        "Player status OCR could not read current/max HP after "
                        "several fresh frames; continuing with the conservative "
                        "spawn/species structural fallback.",
                        "msg_yellow",
                    )
                else:
                    self.bus.log(
                        "Player status OCR read HP "
                        f"{recovery_hints.player_current_hp}/"
                        f"{recovery_hints.player_max_hp}.",
                        "msg_blue",
                    )

            report = run_native_diagnostic(
                self.bot,
                recover=bool(recover),
                persist=bool(recover),
                cancellation=token,
                deadline=deadline,
                timeout_seconds=bounded_timeout,
                status_callback=publish,
                recovery_hints=recovery_hints,
            )
            snapshot = report.after
            facts = snapshot.runtime
            pointer = (
                "unavailable"
                if snapshot.player_base is None or snapshot.world_base is None
                else f"player=0x{snapshot.player_base:X}, world=0x{snapshot.world_base:X}, "
                f"generation={snapshot.pointer_generation}"
            )
            coordinate = (
                "unavailable"
                if facts.map_coordinate_cell is None
                else f"({facts.map_coordinate_cell[0]:.2f}, "
                f"{facts.map_coordinate_cell[1]:.2f})"
            )
            self.bus.log(
                "Native diagnostic summary: "
                f"health={snapshot.status.value}; {pointer}; "
                f"pid={snapshot.process_id}; "
                f"module={snapshot.module_name or 'unknown'}; "
                f"module_path={snapshot.module_path or 'unknown'}; "
                f"module_base={None if snapshot.module_base is None else f'0x{snapshot.module_base:X}'}; "
                f"module_size={None if snapshot.module_size is None else f'0x{snapshot.module_size:X}'}; "
                f"pointer_width={snapshot.pointer_width_bytes}; "
                f"configured_player_offset={None if snapshot.configured_player_pointer_offset is None else f'0x{snapshot.configured_player_pointer_offset:X}'}; "
                f"configured_world_offset={None if snapshot.configured_world_pointer_offset is None else f'0x{snapshot.configured_world_pointer_offset:X}'}; "
                f"player_chain={tuple(hex(value) for value in snapshot.player_pointer_chain_offsets)}; "
                f"world_chain={tuple(hex(value) for value in snapshot.world_pointer_chain_offsets)}; "
                f"world_field={None if snapshot.world_field_offset is None else f'0x{snapshot.world_field_offset:X}'}; "
                f"world_vtable_offset={None if snapshot.world_vtable_offset is None else f'0x{snapshot.world_vtable_offset:X}'}; "
                f"world_vtable_field={None if snapshot.world_vtable_field_offset is None else f'0x{snapshot.world_vtable_field_offset:X}'}; "
                f"world_identity_kind={snapshot.world_identity_kind}; "
                f"self_field={None if snapshot.self_pointer_offset is None else f'0x{snapshot.self_pointer_offset:X}'}; "
                f"species_field={None if snapshot.species_offset is None else f'0x{snapshot.species_offset:X}'}; "
                f"active_field={None if snapshot.active_species_offset is None else f'0x{snapshot.active_species_offset:X}'}; "
                f"hp_field={None if snapshot.hp_offset is None else f'0x{snapshot.hp_offset:X}'}; "
                f"hp_candidates={tuple(hex(value) for value in snapshot.hp_candidate_offsets)}; "
                f"hp_validated={snapshot.hp_offset_validated}; "
                f"hp_transition_support={tuple((hex(offset), count) for offset, count in snapshot.hp_transition_support)}; "
                f"xyz_fields={tuple(None if value is None else hex(value) for value in (snapshot.x_offset, snapshot.y_offset, snapshot.z_offset))}; "
                f"map={facts.selected_map_name or 'unselected'}; "
                f"map_cell={coordinate}; "
                f"cached_actor_slots={snapshot.providers.cached_actor_slots}; "
                f"ocr_enabled={facts.ocr_enabled}; "
                f"ocr_anchor_cached={facts.ocr_anchor_cached}; "
                f"focused={facts.target_focused}; "
                f"coordinate_error={facts.coordinate_error or 'none'}.",
                "msg_blue",
            )
            self.bus.publish_status(
                "native_diagnostic_status",
                f"Native diagnostic finished: {report.outcome.value}.",
                session_id=session_id,
            )
            return report

        try:
            self.workers.start(
                name=(
                    "native-pointer-recovery"
                    if recover
                    else "native-health"
                ),
                kind=WorkerKind.DIAGNOSTIC,
                target=run,
                session_id=session_id,
            )
        except Exception:
            self._diagnostic_session_id = previous_session_id
            self._diagnostic_recovery_requested = previous_recovery_requested
            raise
        return session_id

    def _pointer_recovery_hints(
        self,
        *,
        player_current_hp: int | None = None,
        player_max_hp: int | None = None,
        cancellation: CancellationToken | None = None,
        health_attempts: int = 1,
    ) -> PointerRecoveryHints:
        config = getattr(self.bot, "config", {})
        species: set[int] = set()
        monster_hp_by_species: dict[int, int] = {}
        for entry in config.get("selected_mobs", ()):
            if not isinstance(entry, dict):
                continue
            value = entry.get("species_id")
            if isinstance(value, bool):
                continue
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed <= 0:
                continue
            species.add(parsed)
            anchor_value = entry.get("recovery_anchor_hp")
            if isinstance(anchor_value, bool):
                continue
            try:
                anchor_hp = int(anchor_value)
            except (TypeError, ValueError):
                continue
            if anchor_hp > 0:
                monster_hp_by_species[parsed] = anchor_hp

        # Existing saved selections predate recovery_anchor_hp. Resolve trusted
        # anchors from the current mob registry by species ID so users do not
        # need to delete/reselect mobs or manually enter HP values.
        try:
            registered_mobs = MobInfo.get_all_mobs().values()
        except Exception:  # noqa: BLE001 - registry hints remain optional.
            registered_mobs = ()
        for entry in registered_mobs:
            if not isinstance(entry, dict):
                continue
            try:
                registered_species = int(entry.get("species_id"))
                anchor_hp = int(entry.get("recovery_anchor_hp"))
            except (TypeError, ValueError):
                continue
            if registered_species in species and anchor_hp > 0:
                monster_hp_by_species[registered_species] = anchor_hp

        current_hp = player_current_hp
        maximum_hp = player_max_hp
        if current_hp is None and maximum_hp is None:
            read_player_health = getattr(self.bot, "read_player_health", None)
            if callable(read_player_health):
                for attempt in range(max(1, int(health_attempts))):
                    if cancellation is not None and cancellation.cancelled:
                        break
                    try:
                        health = read_player_health()
                    except Exception:  # noqa: BLE001 - OCR failure is non-fatal.
                        health = None
                    if isinstance(health, tuple) and len(health) == 2:
                        try:
                            current_hp = int(health[0])
                            maximum_hp = int(health[1])
                        except (TypeError, ValueError):
                            current_hp = None
                            maximum_hp = None
                    if (
                        current_hp is not None
                        and maximum_hp is not None
                        and current_hp > 0
                        and maximum_hp > 0
                        and current_hp <= maximum_hp
                    ):
                        break
                    current_hp = None
                    maximum_hp = None
                    if attempt + 1 < max(1, int(health_attempts)):
                        if cancellation is not None:
                            if cancellation.wait(0.10):
                                break
        if (
            current_hp is None
            or maximum_hp is None
            or current_hp <= 0
            or maximum_hp <= 0
            or current_hp > maximum_hp
        ):
            current_hp = None
            maximum_hp = None

        overlay = getattr(self.bot, "_native_map_overlay", None)
        frame = getattr(overlay, "coordinate_frame", None)
        if frame is None:
            map_name = str(config.get("selected_map_name") or "").strip()
            if map_name:
                try:
                    from mapper.CoordinateFrame import CoordinateFrame
                    from mapper.MapCatalog import MapCatalog

                    catalog = MapCatalog()
                    frame_path = (
                        catalog.map_directory(map_name) / "coordinate_frame.json"
                    )
                    if frame_path.is_file():
                        frame = CoordinateFrame.load(frame_path)
                except Exception:  # noqa: BLE001 - missing map hint stays optional.
                    frame = None
        spawn_x = getattr(frame, "origin_native_x", None)
        spawn_z = getattr(frame, "origin_native_z", None)
        return PointerRecoveryHints(
            known_species_ids=tuple(sorted(species)),
            player_spawn_x=None if spawn_x is None else float(spawn_x),
            player_spawn_z=None if spawn_z is None else float(spawn_z),
            player_current_hp=current_hp,
            player_max_hp=maximum_hp,
            monster_hp_by_species=tuple(sorted(monster_hp_by_species.items())),
            require_verified_monster_hp=True,
        )

    def stop_native_diagnostic(self) -> bool:
        """Request cancellation of the current managed native diagnostic."""

        return self.workers.stop(WorkerKind.DIAGNOSTIC)

    def _start_control(
        self,
        name: str,
        target: Callable[[CancellationToken], object],
    ) -> None:
        if self._diagnostic_recovery_requested and self.diagnostic_active:
            raise RuntimeError(
                "Wait for native pointer recovery to finish before starting "
                "a control task."
            )
        if not self.capture_active:
            raise RuntimeError("Attach the Flyff window first.")
        session_id = self._next_control_session_id
        self._next_control_session_id += 1
        previous_session_id = self._control_session_id
        self._control_session_id = session_id
        try:
            self.workers.start(
                name=name,
                kind=WorkerKind.CONTROL,
                target=target,
                stop_hook=self.bot.stop_movement,
                session_id=session_id,
            )
        except Exception:
            self._control_session_id = previous_session_id
            raise

    def stop_control(self) -> bool:
        # Cancel first so a release failure cannot prevent the worker from
        # observing Stop. The registered hook emits unconditional mapper
        # movement KEYUP messages immediately.
        stopping = self.workers.stop(WorkerKind.CONTROL)
        try:
            self.bot.stop()
        except Exception:  # noqa: BLE001 - keep the GUI stop path alive.
            self.bus.log(
                "Bot input cleanup reported an error after cancellation.\n"
                f"{traceback.format_exc()}",
                "msg_red",
            )
        return stopping

    def shutdown(self, timeout: float = 8.0) -> dict[WorkerKind, bool]:
        timeout = max(0.0, float(timeout))
        deadline = monotonic() + timeout
        if not self._shutdown_lock.acquire(timeout=timeout):
            results = {kind: not self.workers.is_active(kind) for kind in WorkerKind}
            self._shutdown_timed_out = tuple(
                kind for kind, stopped in results.items() if not stopped
            )
            return results

        try:
            self._shutdown_requested = True
            if self._shutdown_finalized:
                return dict(self._shutdown_results)

            if self.recording is not None:
                try:
                    self.stop_recording()
                except Exception:  # noqa: BLE001 - shutdown must not hang on this.
                    pass

            remaining = max(0.0, deadline - monotonic())
            results = self.workers.shutdown(remaining)
            self._shutdown_results = dict(results)
            self._shutdown_timed_out = tuple(
                kind for kind, stopped in results.items() if not stopped
            )
            if self._shutdown_timed_out:
                names = ", ".join(kind.value for kind in self._shutdown_timed_out)
                message = (
                    "Shutdown timed out while waiting for: "
                    f"{names}. Runtime resources remain open so live workers "
                    "are not invalidated."
                )
                self.bus.publish_status("runtime_status", message)
                self.bus.log(message, "msg_red")
                return dict(results)

            try:
                self.bot.stop()
            except Exception:  # noqa: BLE001 - final release must still run.
                self.bus.log(
                    "Bot input cleanup reported an error during shutdown.\n"
                    f"{traceback.format_exc()}",
                    "msg_red",
                )
            try:
                self.bot.release_input()
            except Exception:  # noqa: BLE001 - workers are stopped; finish closure.
                self.bus.log(
                    "Final input release reported an error during shutdown.\n"
                    f"{traceback.format_exc()}",
                    "msg_red",
                )
            finally:
                self.bus.close()
                self._shutdown_finalized = True
            return dict(results)
        finally:
            self._shutdown_lock.release()

    def _farming_stats_reporter(self, mode: str):
        session_id = self._control_session_id

        def report(payload: Mapping[str, object]) -> None:
            action_values = payload.get("action_reward_deltas", ())
            action_deltas: list[tuple[str, float]] = []
            if isinstance(action_values, (tuple, list)):
                for item in action_values:
                    if (
                        isinstance(item, (tuple, list))
                        and len(item) == 2
                        and isinstance(item[0], str)
                        and isinstance(item[1], (int, float))
                    ):
                        action_deltas.append((item[0], float(item[1])))
            snapshot = FarmingSessionSnapshot(
                session_id=session_id,
                mode=str(payload.get("mode", mode)),
                started_at_monotonic=float(
                    payload.get("started_at_monotonic", monotonic())
                ),
                elapsed_seconds=max(
                    0.0, float(payload.get("elapsed_seconds", 0.0))
                ),
                total_steps=max(0, int(payload.get("total_steps", 0))),
                session_steps=max(0, int(payload.get("session_steps", 0))),
                reward=float(payload.get("reward", 0.0)),
                reward_delta=float(payload.get("reward_delta", 0.0)),
                kills=max(0, int(payload.get("kills", 0))),
                kills_per_hour=max(
                    0.0, float(payload.get("kills_per_hour", 0.0))
                ),
                penya_earned=max(0, int(payload.get("penya_earned", 0))),
                penya_per_hour=max(
                    0.0, float(payload.get("penya_per_hour", 0.0))
                ),
                perin_earned=max(
                    0.0, float(payload.get("perin_earned", 0.0))
                ),
                action_reward_deltas=tuple(action_deltas),
            )
            self.bus.publish_latest("farming_session_stats", snapshot)
            self.bus.heartbeat("rl")

        return report

    def _reporter(self, status_key: str, worker: str):
        session_id = self._control_session_id

        def report(message: str) -> None:
            rendered = str(message)
            self.bus.publish_status(
                status_key,
                rendered,
                session_id=session_id,
            )
            self.bus.log(rendered, "msg_blue")
            if (
                status_key == "rl_status"
                and "reason=external_teleport" in rendered
            ):
                pulse_completed = "teleport_pulse=completed" in rendered
                pulse_text = (
                    "The bot sent the configured 300 ms forward recovery pulse"
                    if pulse_completed
                    else "The bot attempted the forward recovery pulse, but it failed"
                )
                self.bus.alert(
                    "Unexpected teleport detected",
                    "The character teleported while outside the mapped teleport "
                    f"area. {pulse_text}, released all movement keys, and stopped. "
                    "Any active training session was saved safely before this "
                    "alert. Review the session diagnostics for the recorded "
                    "coordinates and pulse result.",
                    session_id=session_id,
                )
            self.bus.heartbeat(worker)

        return report
