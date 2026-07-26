from __future__ import annotations

import traceback

import cv2 as cv
from collections.abc import Callable
from typing import cast

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

    @property
    def capture_active(self) -> bool:
        return self.workers.is_active(WorkerKind.CAPTURE)

    @property
    def control_active(self) -> bool:
        return self.workers.is_active(WorkerKind.CONTROL)

    def control_snapshot(self) -> WorkerSnapshot | None:
        return self.workers.snapshot(WorkerKind.CONTROL)

    def attach(self, window_handle: int) -> int:
        if self.control_active:
            raise RuntimeError(
                "Cannot reattach while a control task is active. Stop it first."
            )
        self.bot.release_input()
        if not self.preview.stop(3.0):
            raise RuntimeError("Previous preview worker did not stop.")
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
            from train import run_trained_agent, train_agent

            report = self._reporter("rl_status", "rl")
            self.bot.start()
            try:
                if mode == "train":
                    return train_agent(
                        self.bot,
                        status_callback=report,
                        cancellation=token,
                    )
                if mode == "agent":
                    run_trained_agent(
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

    def publish_map_preview(self, map_name: str) -> bool:
        from mapper.MapCatalog import MapCatalog

        preview_path = MapCatalog().preview_path(map_name)
        image = cv.imread(str(preview_path), cv.IMREAD_COLOR)
        if image is None:
            return False
        self.bus.publish_latest("map_frame", image)
        return True

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
        self.workers.start(
            name=name,
            kind=WorkerKind.CONTROL,
            target=target,
            stop_hook=self.bot.stop_movement,
        )

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
        try:
            self.bot.stop()
        except Exception:  # noqa: BLE001 - shutdown must continue.
            self.bus.log(
                "Bot input cleanup reported an error during shutdown.\n"
                f"{traceback.format_exc()}",
                "msg_red",
            )
        results = self.workers.shutdown(timeout)
        try:
            self.bot.release_input()
        except Exception:  # noqa: BLE001 - close the runtime bus regardless.
            self.bus.log(
                "Final input release reported an error during shutdown.\n"
                f"{traceback.format_exc()}",
                "msg_red",
            )
        finally:
            self.bus.close()
        return results

    def _reporter(self, status_key: str, worker: str):
        def report(message: str) -> None:
            self.bus.publish_latest(status_key, str(message))
            self.bus.log(str(message), "msg_blue")
            self.bus.heartbeat(worker)

        return report
