from __future__ import annotations

import traceback
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from runtime_bus import RuntimeBus
from worker_manager import CancellationToken, WorkerKind, WorkerManager

Frame = NDArray[np.uint8]


@dataclass(frozen=True)
class FrameSample:
    """One copied capture frame with monotonic freshness metadata."""

    frame: Frame
    generation: int
    sequence: int
    captured_at: float

    @property
    def identity(self) -> tuple[int, int]:
        """Uniquely identify a frame across capture reattachments."""
        return self.generation, self.sequence


class FrameSource(Protocol):
    def get_frame(self) -> tuple[Frame, Frame]: ...

    def close(self) -> None: ...


FrameSourceFactory = Callable[[int], FrameSource]
PreviewBuilder = Callable[[Frame], Frame]
PreviewEnabled = Callable[[], bool]


class CaptureLostError(RuntimeError):
    """The attached source exceeded the bounded transient-retry allowance."""


class CaptureService:
    """Owns one generation-safe frame source at a time."""

    def __init__(
        self,
        manager: WorkerManager,
        bus: RuntimeBus,
        source_factory: FrameSourceFactory,
        *,
        preview_builder: PreviewBuilder | None = None,
        preview_enabled: PreviewEnabled | None = None,
        preview_fps: float = 12.0,
        retry_delay: float = 0.25,
        error_log_interval: float = 15.0,
        maximum_consecutive_failures: int = 20,
    ) -> None:
        if maximum_consecutive_failures < 1:
            raise ValueError("maximum_consecutive_failures must be at least one")
        self._manager = manager
        self._bus = bus
        self._source_factory = source_factory
        self._preview_builder = preview_builder
        self._preview_enabled = preview_enabled or (lambda: True)
        self._preview_interval = 1.0 / max(1.0, preview_fps)
        self._retry_delay = max(0.0, retry_delay)
        self._error_log_interval = max(0.0, error_log_interval)
        self._maximum_consecutive_failures = int(maximum_consecutive_failures)
        self._lock = Lock()
        self._source: FrameSource | None = None
        self._color_frame: Frame | None = None
        self._gray_frame: Frame | None = None
        self._generation = 0
        self._frame_sequence = 0
        self._captured_at: float | None = None

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
            self._frame_sequence = 0
            self._captured_at = None

        try:
            self._bus.publish_status(
                "capture_status",
                "starting",
                session_id=generation,
            )
            self._manager.start(
                name=f"capture-{generation}",
                kind=WorkerKind.CAPTURE,
                target=lambda token: self._run(generation, source, token),
                stop_hook=source.close,
                session_id=generation,
            )
        except Exception:
            source.close()
            with self._lock:
                if self._source is source:
                    self._source = None
            self._bus.publish_status(
                "capture_status",
                "lost",
                session_id=generation,
            )
            raise
        return generation

    def stop(self, timeout: float = 5.0) -> bool:
        stopped = self._manager.stop_and_join(WorkerKind.CAPTURE, timeout)
        if stopped:
            with self._lock:
                self._source = None
                self._color_frame = None
                self._gray_frame = None
                self._frame_sequence = 0
                self._captured_at = None
        return stopped

    def snapshot(self) -> tuple[Frame | None, Frame | None]:
        """Return copied color/gray frames using the original public API."""
        with self._lock:
            color = None if self._color_frame is None else self._color_frame.copy()
            gray = None if self._gray_frame is None else self._gray_frame.copy()
        return color, gray

    def sample(self, *, grayscale: bool = True) -> FrameSample | None:
        """
        Return one copied frame with capture generation, sequence and timestamp.

        ``captured_at`` uses ``time.monotonic()`` and records when the capture
        worker finished acquiring the frame. Consumers can therefore reject
        duplicate or old frames without comparing image contents.
        """
        with self._lock:
            source = self._gray_frame if grayscale else self._color_frame
            captured_at = self._captured_at
            if source is None or captured_at is None or self._frame_sequence <= 0:
                return None
            return FrameSample(
                frame=source.copy(),
                generation=self._generation,
                sequence=self._frame_sequence,
                captured_at=captured_at,
            )

    def _run(
        self,
        generation: int,
        source: FrameSource,
        token: CancellationToken,
    ) -> None:
        last_preview_at = 0.0
        last_fps_at = 0.0
        previous_at = monotonic()
        frame_times: deque[float] = deque(maxlen=30)
        last_error_log_at: float | None = None
        consecutive_failures = 0
        capture_was_live = False

        try:
            while not token.cancelled:
                try:
                    color, gray = source.get_frame()
                except Exception as error:
                    if token.cancelled:
                        break

                    now = monotonic()
                    consecutive_failures += 1
                    if consecutive_failures == 1:
                        self._bus.publish_status(
                            "capture_status",
                            "degraded",
                            session_id=generation,
                        )
                    if (
                        last_error_log_at is None
                        or now - last_error_log_at >= self._error_log_interval
                    ):
                        self._bus.log(
                            "Game capture failed; retrying while the window "
                            f"remains attached.\n{traceback.format_exc()}",
                            "msg_red",
                        )
                        last_error_log_at = now

                    if consecutive_failures >= self._maximum_consecutive_failures:
                        self._clear_generation(generation, source)
                        self._bus.publish_status(
                            "capture_status",
                            "lost",
                            session_id=generation,
                        )
                        raise CaptureLostError(
                            f"Capture generation {generation} was lost after "
                            f"{consecutive_failures} consecutive failures"
                        ) from error

                    if token.wait(self._retry_delay):
                        break
                    previous_at = monotonic()
                    frame_times.clear()
                    continue

                now = monotonic()
                recovered = consecutive_failures > 0
                consecutive_failures = 0
                with self._lock:
                    if generation != self._generation:
                        raise RuntimeError(
                            "Stale capture generation attempted to publish."
                        )
                    self._color_frame = color
                    self._gray_frame = gray
                    self._frame_sequence += 1
                    self._captured_at = now
                if not capture_was_live or recovered:
                    self._bus.publish_status(
                        "capture_status",
                        "live",
                        session_id=generation,
                    )
                    capture_was_live = True

                elapsed = max(now - previous_at, 1e-6)
                previous_at = now
                frame_times.append(elapsed)
                if now - last_fps_at >= 0.5:
                    average = sum(frame_times) / len(frame_times)
                    self._bus.publish_latest(
                        "video_fps",
                        round(1.0 / average),
                    )
                    last_fps_at = now
                self._bus.heartbeat("capture")

                if (
                    self._preview_enabled()
                    and now - last_preview_at >= self._preview_interval
                ):
                    preview = color.copy()
                    if self._preview_builder is not None:
                        preview = self._preview_builder(preview)
                    self._bus.publish_latest("debug_frame", preview)
                    last_preview_at = now
        finally:
            try:
                source.close()
            finally:
                self._clear_generation(generation, source)
                if token.cancelled:
                    self._bus.publish_status(
                        "capture_status",
                        "stopped",
                        session_id=generation,
                    )

    def _clear_generation(
        self,
        generation: int,
        source: FrameSource,
    ) -> None:
        with self._lock:
            if generation != self._generation or self._source is not source:
                return
            self._source = None
            self._color_frame = None
            self._gray_frame = None
            self._frame_sequence = 0
            self._captured_at = None
