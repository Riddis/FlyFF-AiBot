from __future__ import annotations

from time import monotonic

from capture_service import CaptureService
from runtime_bus import RuntimeBus
from worker_manager import CancellationToken, WorkerKind, WorkerManager


class PreviewService:
    """Build droppable Bot Vision frames without blocking capture or Tk."""

    def __init__(
        self,
        manager: WorkerManager,
        bus: RuntimeBus,
        capture: CaptureService,
        preview_builder,
        *,
        preview_fps: float = 10.0,
    ) -> None:
        self._manager = manager
        self._bus = bus
        self._capture = capture
        self._preview_builder = preview_builder
        self._preview_interval = 1.0 / max(1.0, preview_fps)

    @property
    def active(self) -> bool:
        return self._manager.is_active(WorkerKind.PREVIEW)

    def start(self) -> None:
        self._manager.start(
            name="preview",
            kind=WorkerKind.PREVIEW,
            target=self._run,
        )

    def stop(self, timeout: float = 3.0) -> bool:
        return self._manager.stop_and_join(WorkerKind.PREVIEW, timeout)

    def _run(self, token: CancellationToken) -> None:
        while not token.cancelled:
            started_at = monotonic()
            color, _gray = self._capture.snapshot()
            if color is not None:
                preview = self._preview_builder(color)
                self._bus.publish_latest("debug_frame", preview)
                self._bus.heartbeat("preview")

            remaining = self._preview_interval - (monotonic() - started_at)
            if token.wait(max(0.0, remaining)):
                break
