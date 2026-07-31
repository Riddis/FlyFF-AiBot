from __future__ import annotations

from threading import Event
from time import monotonic

import numpy as np
from preview_service import PreviewService
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
