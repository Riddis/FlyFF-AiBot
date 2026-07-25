from __future__ import annotations

from collections.abc import Callable
from typing import cast

from capture_service import CaptureService, FrameSource
from libs.WindowCapture import WindowCapture
from mapper import Mapper, RotationCalibrator
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

    def start_mapper(self) -> None:
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
            )
            return mapper.run()

        self._start_control("mapper", run)

    def start_calibration(self, *, visual_confirmation: bool) -> None:
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
        self.bot.stop()
        return self.workers.stop(WorkerKind.CONTROL)

    def shutdown(self, timeout: float = 8.0) -> dict[WorkerKind, bool]:
        self.bot.stop()
        results = self.workers.shutdown(timeout)
        self.bot.release_input()
        self.bus.close()
        return results

    def _reporter(self, status_key: str, worker: str):
        def report(message: str) -> None:
            self.bus.publish_latest(status_key, str(message))
            self.bus.log(str(message), "msg_blue")
            self.bus.heartbeat(worker)

        return report
