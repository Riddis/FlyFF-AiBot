from __future__ import annotations

import traceback
from collections.abc import Callable
from threading import Lock
from time import monotonic
from typing import cast

import cv2 as cv
from capture_service import CaptureService, FrameSource
from libs.WindowCapture import WindowCapture
from mapper import Mapper
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
        )
        self._next_control_session_id = 1
        self._control_session_id: int | None = None
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
    def control_session_id(self) -> int | None:
        return self._control_session_id

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
            from native_farming import (
                dry_run_native_farming,
                run_native_farming_agent,
                train_native_farming,
            )

            report = self._reporter("rl_status", "rl")
            try:
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

    def _start_control(
        self,
        name: str,
        target: Callable[[CancellationToken], object],
    ) -> None:
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
