from __future__ import annotations

from threading import Event
from time import monotonic

import numpy as np
from capture_service import CaptureService
from runtime_bus import RuntimeBus
from worker_manager import WorkerKind, WorkerManager


class FakeSource:
    def __init__(
        self,
        *,
        block_after_first: bool = False,
        failures_before_success: int = 0,
    ) -> None:
        self.closed = Event()
        self.frames = 0
        self.block_after_first = block_after_first
        self.failures_before_success = failures_before_success

    def get_frame(self):
        self.frames += 1
        if self.frames <= self.failures_before_success:
            raise RuntimeError("transient capture failure")
        if self.block_after_first and self.frames > 1:
            self.closed.wait(5.0)
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        gray = np.zeros((4, 4), dtype=np.uint8)
        return frame, gray

    def close(self) -> None:
        self.closed.set()


def _wait_until(predicate, timeout: float = 1.0) -> bool:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if predicate():
            return True
        Event().wait(0.005)
    return False


def test_capture_publishes_snapshots_and_stops_cleanly() -> None:
    source = FakeSource()
    manager = WorkerManager(RuntimeBus())
    service = CaptureService(manager, RuntimeBus(), lambda _handle: source)

    assert service.attach(123) == 1
    assert _wait_until(lambda: source.frames > 0)
    color, gray = service.snapshot()
    assert color is not None
    assert gray is not None
    assert service.stop(1.0)
    assert source.closed.is_set()


def test_reattach_waits_for_previous_generation_to_exit() -> None:
    first = FakeSource(block_after_first=True)
    second = FakeSource()
    sources = iter((first, second))
    bus = RuntimeBus()
    manager = WorkerManager(bus)
    service = CaptureService(manager, bus, lambda _handle: next(sources))

    service.attach(1)
    assert _wait_until(lambda: first.frames > 1)
    assert service.attach(2, join_timeout=1.0) == 2
    assert first.closed.is_set()
    assert manager.is_active(WorkerKind.CAPTURE)
    assert service.stop(1.0)


def test_transient_capture_failure_is_retried_without_detaching() -> None:
    source = FakeSource(failures_before_success=1)
    bus = RuntimeBus()
    manager = WorkerManager(bus)
    service = CaptureService(
        manager,
        bus,
        lambda _handle: source,
        retry_delay=0.0,
    )

    service.attach(123)

    assert _wait_until(lambda: service.snapshot()[0] is not None)
    assert service.active
    logs = bus.drain_logs()
    assert len(logs) == 1
    assert logs[0][0] == "msg_red"
    assert "transient capture failure" in logs[0][1]
    assert service.stop(1.0)
