from __future__ import annotations

from threading import Event
from time import monotonic

import numpy as np
from capture_service import CaptureService
from runtime_bus import RuntimeBus, RuntimeStatus
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


class OneFrameThenFailSource:
    def __init__(self) -> None:
        self.closed = Event()
        self.frames = 0

    def get_frame(self):
        self.frames += 1
        if self.frames > 1:
            raise RuntimeError("window no longer exists")
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

    first_sample = service.sample()
    assert first_sample is not None
    assert first_sample.generation == 1
    assert first_sample.sequence > 0
    assert first_sample.captured_at <= monotonic()
    first_sample.frame.fill(255)

    assert _wait_until(
        lambda: (
            (sample := service.sample()) is not None
            and sample.sequence > first_sample.sequence
        )
    )
    latest_sample = service.sample()
    assert latest_sample is not None
    assert latest_sample.identity != first_sample.identity
    assert not np.any(latest_sample.frame)
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


def test_repeated_capture_failure_becomes_terminal_and_clears_stale_frame() -> None:
    source = OneFrameThenFailSource()
    bus = RuntimeBus()
    manager = WorkerManager(bus)
    service = CaptureService(
        manager,
        bus,
        lambda _handle: source,
        retry_delay=0.0,
        maximum_consecutive_failures=3,
    )

    assert service.attach(123) == 1
    assert _wait_until(lambda: not service.active)

    color, gray = service.snapshot()
    assert color is None
    assert gray is None
    assert service.sample() is None
    assert source.closed.is_set()

    failures = bus.drain_failures()
    assert len(failures) == 1
    assert failures[0].worker_name == "capture-1"
    assert failures[0].session_id == 1
    assert "lost after 3 consecutive failures" in failures[0].traceback

    _version, status = bus.read_latest("capture_status")
    assert status == RuntimeStatus(message="lost", session_id=1)
