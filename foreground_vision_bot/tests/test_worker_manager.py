from __future__ import annotations

from threading import Event

import pytest
from runtime_bus import RuntimeBus
from worker_manager import WorkerKind, WorkerManager, WorkerState


def test_duplicate_worker_kind_is_rejected() -> None:
    bus = RuntimeBus()
    manager = WorkerManager(bus)
    started = Event()

    def run(token):
        started.set()
        token.wait(5.0)

    manager.start(name="capture-one", kind=WorkerKind.CAPTURE, target=run)
    assert started.wait(1.0)

    with pytest.raises(RuntimeError, match="already active"):
        manager.start(
            name="capture-two",
            kind=WorkerKind.CAPTURE,
            target=run,
        )

    assert manager.stop_and_join(WorkerKind.CAPTURE, 1.0)


def test_stop_hook_and_token_are_both_used() -> None:
    bus = RuntimeBus()
    manager = WorkerManager(bus)
    hook_called = Event()
    observed_cancel = Event()

    def run(token):
        token.wait(5.0)
        if token.cancelled:
            observed_cancel.set()

    manager.start(
        name="mapper",
        kind=WorkerKind.CONTROL,
        target=run,
        stop_hook=hook_called.set,
    )

    assert manager.stop_and_join(WorkerKind.CONTROL, 1.0)
    assert hook_called.is_set()
    assert observed_cancel.is_set()
    snapshot = manager.snapshot(WorkerKind.CONTROL)
    assert snapshot is not None
    assert snapshot.state is WorkerState.COMPLETED


def test_failure_contains_full_traceback() -> None:
    bus = RuntimeBus()
    manager = WorkerManager(bus)

    def fail(_token):
        raise ValueError("worker exploded")

    manager.start(
        name="agent",
        kind=WorkerKind.CONTROL,
        target=fail,
        session_id=41,
    )
    assert manager.join(WorkerKind.CONTROL, 1.0)

    failures = bus.drain_failures()
    assert len(failures) == 1
    assert failures[0].worker_name == "agent"
    assert failures[0].session_id == 41
    assert "ValueError: worker exploded" in failures[0].traceback


def test_repeated_stop_does_not_repeat_the_stop_hook() -> None:
    bus = RuntimeBus()
    manager = WorkerManager(bus)
    started = Event()
    release = Event()
    hook_calls = 0

    def run(_token):
        started.set()
        release.wait(5.0)

    def stop_hook() -> None:
        nonlocal hook_calls
        hook_calls += 1

    manager.start(
        name="stuck",
        kind=WorkerKind.CONTROL,
        target=run,
        stop_hook=stop_hook,
    )
    assert started.wait(1.0)

    assert manager.stop(WorkerKind.CONTROL)
    assert manager.stop(WorkerKind.CONTROL)
    assert hook_calls == 1

    release.set()
    assert manager.join(WorkerKind.CONTROL, 1.0)
