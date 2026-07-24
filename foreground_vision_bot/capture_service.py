from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from time import monotonic
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from runtime_bus import RuntimeBus
from worker_manager import CancellationToken, WorkerKind, WorkerManager

Frame = NDArray[np.uint8]


class FrameSource(Protocol):
    def get_frame(self) -> tuple[Frame, Frame]: ...

    def close(self) -> None: ...


FrameSourceFactory = Callable[[int], FrameSource]
PreviewBuilder = Callable[[Frame], Frame]


class CaptureService:
    """Owns one generation-safe frame source at a time."""

    def __init__(
        self,
        manager: WorkerManager,
        bus: RuntimeBus,
        source_factory: FrameSourceFactory,
        *,
        preview_builder: PreviewBuilder | None = None,
        preview_fps: float = 12.0,
    ) -> None:
        self._manager = manager
        self._bus = bus
        self._source_factory = source_factory
        self._preview_builder = preview_builder
        self._preview_interval = 1.0 / max(1.0, preview_fps)
        self._lock = Lock()
        self._source: FrameSource | None = None
        self._color_frame: Frame | None = None
        self._gray_frame: Frame | None = None
        self._generation = 0

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def active(self) -> bool:
        return self._manager.is_active(WorkerKind.CAPTURE)

    def attach(self, window_handle: int, *, join_timeout: float = 5.0) -> int:
        if self.active and not self.stop(join_timeout):
            raise RuntimeError(
                "Previous capture generation did not stop; reattach rejected."
            )

        source = self._source_factory(window_handle)
        with self._lock:
            self._generation += 1
            generation = self._generation
            self._source = source
            self._color_frame = None
            self._gray_frame = None

        try:
            self._manager.start(
                name=f"capture-{generation}",
                kind=WorkerKind.CAPTURE,
                target=lambda token: self._run(generation, source, token),
                stop_hook=source.close,
            )
        except Exception:
            source.close()
            with self._lock:
                if self._source is source:
                    self._source = None
            raise
        return generation

    def stop(self, timeout: float = 5.0) -> bool:
        stopped = self._manager.stop_and_join(WorkerKind.CAPTURE, timeout)
        if stopped:
            with self._lock:
                self._source = None
                self._color_frame = None
                self._gray_frame = None
        return stopped

    def snapshot(self) -> tuple[Frame | None, Frame | None]:
        with self._lock:
            color = None if self._color_frame is None else self._color_frame.copy()
            gray = None if self._gray_frame is None else self._gray_frame.copy()
        return color, gray

    def _run(
        self,
        generation: int,
        source: FrameSource,
        token: CancellationToken,
    ) -> None:
        last_preview_at = 0.0
        previous_at = monotonic()

        try:
            while not token.cancelled:
                color, gray = source.get_frame()
                now = monotonic()
                with self._lock:
                    if generation != self._generation:
                        raise RuntimeError(
                            "Stale capture generation attempted to publish."
                        )
                    self._color_frame = color
                    self._gray_frame = gray

                elapsed = max(now - previous_at, 1e-6)
                previous_at = now
                self._bus.publish_latest("video_fps", round(1.0 / elapsed))
                self._bus.heartbeat("capture")

                if now - last_preview_at >= self._preview_interval:
                    preview = color.copy()
                    if self._preview_builder is not None:
                        preview = self._preview_builder(preview)
                    self._bus.publish_latest("debug_frame", preview)
                    last_preview_at = now
        finally:
            source.close()
