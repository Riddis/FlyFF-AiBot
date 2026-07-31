from __future__ import annotations

import traceback
from collections.abc import Callable
from math import isfinite
from threading import Lock
from time import monotonic
from typing import cast

import cv2 as cv
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
from runtime_bus import RuntimeBus
from worker_manager import (
    CancellationToken,
    WorkerKind,
    WorkerManager,
    WorkerSnapshot,
)


class RuntimeController:
    """Single owner for capture and control-worker lifecycle."""

    MAX_NATIVE_DIAGNOSTIC_TIMEOUT_SECONDS: float = 30.0
    STARTUP_POINTER_RECOVERY_TIMEOUT_SECONDS: float = 10.0

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

    def attach(self, window_handle: int) -> int:
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
            )

            report = self._reporter("rl_status", "rl")
            try:
                if not self._prepare_native_pointer_startup(token, report):
                    return {"started": False, "reason": "native_pointer_unavailable"}
                if mode == "train":
                    return train_native_farming(
                        self.bot,
                        status_callback=report,
                        cancellation=token,
                    )
                if mode == "agent":
                    run_native_farming_agent(
                        self.bot,
                        status_callback=report,
                        cancellation=token,
                    )
                    return None
                if mode == "dry-run":
                    dry_run_native_farming(
                        self.bot,
                        status_callback=report,
                        cancellation=token,
                    )
                    return None
                raise ValueError(f"Unknown RL mode: {mode}")
            finally:
                self.bot.stop()

        self._start_control(f"rl-{mode}", run)

    def _prepare_native_pointer_startup(
        self,
        token: CancellationToken,
        report: Callable[[str], None],
    ) -> bool:
        """Resolve expected stale/null native state inside the control worker."""

        service = getattr(self.bot, "native_process_service", None)
        if service is None:
            return True
        try:
            service.read_pointer_snapshot()
            return True
        except NativePointerSnapshotError:
            report("Native pointers unavailable; running bounded startup recovery.")

        timeout = self.STARTUP_POINTER_RECOVERY_TIMEOUT_SECONDS
        deadline = monotonic() + timeout

        def publish(progress) -> None:
            report(progress.message)
            self.bus.heartbeat("control")

        try:
            result = service.recover_pointers(
                persist=False,
                cancellation=token,
                deadline=deadline,
                timeout_seconds=timeout,
                status_callback=publish,
                hints=self._pointer_recovery_hints(),
            )
        except Exception as error:  # noqa: BLE001 - expected startup boundary.
            report(
                "Native pointer startup recovery could not complete: "
                f"{type(error).__name__}: {error}. No input was activated."
            )
            return False
        if not result.succeeded or not result.applied:
            report(
                "Native pointers remain unavailable after bounded recovery "
                f"({result.outcome.value}). No input was activated."
            )
            return False
        try:
            snapshot = service.read_pointer_snapshot()
        except NativePointerSnapshotError as error:
            report(
                "Recovered native pointers did not pass startup verification: "
                f"{error}. No input was activated."
            )
            return False
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
        from mapper.MapCatalog import MapCatalog

        preview_path = MapCatalog().preview_path(map_name)
        image = cv.imread(str(preview_path), cv.IMREAD_COLOR)
        if image is None:
            return False
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
        recovery_hints = self._pointer_recovery_hints(
            player_current_hp=player_current_hp,
            player_max_hp=player_max_hp,
        )

        def run(token: CancellationToken) -> NativeDiagnosticReport:
            deadline = monotonic() + bounded_timeout

            def publish(progress: NativeDiagnosticProgress) -> None:
                self.bus.publish_status(
                    "native_diagnostic_status",
                    progress.message,
                    session_id=session_id,
                )
                self.bus.log(progress.message, "msg_blue")
                self.bus.heartbeat("diagnostic")

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
                f"self_field={None if snapshot.self_pointer_offset is None else f'0x{snapshot.self_pointer_offset:X}'}; "
                f"species_field={None if snapshot.species_offset is None else f'0x{snapshot.species_offset:X}'}; "
                f"active_field={None if snapshot.active_species_offset is None else f'0x{snapshot.active_species_offset:X}'}; "
                f"hp_field={None if snapshot.hp_offset is None else f'0x{snapshot.hp_offset:X}'}; "
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
    ) -> PointerRecoveryHints:
        config = getattr(self.bot, "config", {})
        species: set[int] = set()
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
            if parsed > 0:
                species.add(parsed)

        if player_current_hp is not None:
            config["pointer_recovery_current_hp"] = int(player_current_hp)
        if player_max_hp is not None:
            config["pointer_recovery_max_hp"] = int(player_max_hp)
        current_hp = config.get("pointer_recovery_current_hp")
        maximum_hp = config.get("pointer_recovery_max_hp")
        try:
            current_hp = None if current_hp in (None, "") else int(current_hp)
            maximum_hp = None if maximum_hp in (None, "") else int(maximum_hp)
        except (TypeError, ValueError):
            current_hp = None
            maximum_hp = None

        overlay = getattr(self.bot, "_native_map_overlay", None)
        frame = getattr(overlay, "coordinate_frame", None)
        spawn_x = getattr(frame, "origin_native_x", None)
        spawn_z = getattr(frame, "origin_native_z", None)
        return PointerRecoveryHints(
            known_species_ids=tuple(sorted(species)),
            player_spawn_x=None if spawn_x is None else float(spawn_x),
            player_spawn_z=None if spawn_z is None else float(spawn_z),
            player_current_hp=current_hp,
            player_max_hp=maximum_hp,
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

    def _reporter(self, status_key: str, worker: str):
        session_id = self._control_session_id

        def report(message: str) -> None:
            self.bus.publish_status(
                status_key,
                str(message),
                session_id=session_id,
            )
            self.bus.log(str(message), "msg_blue")
            self.bus.heartbeat(worker)

        return report
