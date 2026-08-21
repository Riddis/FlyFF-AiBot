from __future__ import annotations

from threading import Event
from time import monotonic

import numpy as np
from bot.preview_service import PreviewService
from runtime_bus import RuntimeBus
from worker_manager import WorkerManager


class FakeCapture:
    generation = 1

    def snapshot(self):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        return frame, None


def test_preview_service_publishes_built_frames_and_stops() -> None:
    bus = RuntimeBus()
    manager = WorkerManager(bus)
    built = Event()

    def build(frame):
        frame[0, 0] = (0, 255, 0)
        built.set()
        return frame

    service = PreviewService(
        manager,
        bus,
        FakeCapture(),
        build,
        preview_fps=30.0,
    )
    service.start()

    deadline = monotonic() + 1.0
    while not built.is_set() and monotonic() < deadline:
        built.wait(0.01)

    version, frame = bus.read_latest("debug_frame")
    assert version > 0
    assert frame is not None
    assert frame[0, 0].tolist() == [0, 255, 0]
    assert service.stop(1.0)


def test_preview_service_passes_cancellation_into_expensive_builder() -> None:
    bus = RuntimeBus()
    manager = WorkerManager(bus)
    entered = Event()

    def build(_frame):
        raise AssertionError("cancellable builder should be preferred")

    def build_cancellable(frame, token):
        entered.set()
        while not token.cancelled:
            token.wait(0.01)
        return frame

    service = PreviewService(
        manager,
        bus,
        FakeCapture(),
        build,
        preview_fps=30.0,
        cancellable_preview_builder=build_cancellable,
    )
    service.start()

    assert entered.wait(1.0)
    assert service.stop(0.5)
