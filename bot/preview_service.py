from __future__ import annotations

from time import monotonic

from runtime.capture_service import CaptureService
from runtime.runtime_bus import RuntimeBus
from runtime.worker_manager import CancellationToken, WorkerKind, WorkerManager


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
        cancellable_preview_builder=None,
    ) -> None:
        self._manager = manager
        self._bus = bus
        self._capture = capture
        self._preview_builder = preview_builder
        self._cancellable_preview_builder = cancellable_preview_builder
        self._preview_interval = 1.0 / max(1.0, preview_fps)
        self._session_id: int | None = None

    @property
    def active(self) -> bool:
        return self._manager.is_active(WorkerKind.PREVIEW)

    @property
    def session_id(self) -> int | None:
        return self._session_id

    def start(self) -> None:
        session_id = self._capture.generation
        self._session_id = session_id
        self._bus.publish_status(
            "preview_status",
            "starting",
            session_id=session_id,
        )
        self._manager.start(
            name="preview",
            kind=WorkerKind.PREVIEW,
            target=lambda token: self._run(session_id, token),
            session_id=session_id,
        )

    def stop(self, timeout: float = 3.0) -> bool:
        return self._manager.stop_and_join(WorkerKind.PREVIEW, timeout)

    def _run(self, session_id: int, token: CancellationToken) -> None:
        live = False
        try:
            while not token.cancelled:
                started_at = monotonic()
                color, _gray = self._capture.snapshot()
                if color is not None:
                    if self._cancellable_preview_builder is None:
                        preview = self._preview_builder(color)
                    else:
                        preview = self._cancellable_preview_builder(color, token)
                    if token.cancelled:
                        break
                    self._bus.publish_latest("debug_frame", preview)
                    self._bus.heartbeat("preview")
                    if not live:
                        self._bus.publish_status(
                            "preview_status",
                            "live",
                            session_id=session_id,
                        )
                        live = True

                remaining = self._preview_interval - (monotonic() - started_at)
                if token.wait(max(0.0, remaining)):
                    break
        except Exception:
            self._bus.publish_status(
                "preview_status",
                "failed",
                session_id=session_id,
            )
            raise
        finally:
            if token.cancelled:
                self._bus.publish_status(
                    "preview_status",
                    "stopped",
                    session_id=session_id,
                )
