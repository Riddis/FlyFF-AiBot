from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from threading import Event, Lock, Thread
from time import monotonic

from runtime_bus import RuntimeBus, WorkerFailure


class WorkerKind(Enum):
    CAPTURE = "capture"
    CONTROL = "control"


class WorkerState(Enum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"


class CancellationToken:
    """Per-worker cooperative cancellation signal."""

    def __init__(self) -> None:
        self._event = Event()

    @property
    def event(self) -> Event:
        return self._event

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    def wait(self, timeout: float) -> bool:
        return self._event.wait(max(0.0, timeout))

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise WorkerCancelled


class WorkerCancelled(Exception):
    """Internal cooperative-cancellation signal."""


WorkerTarget = Callable[[CancellationToken], object]
StopHook = Callable[[], None]


@dataclass(frozen=True)
class WorkerSnapshot:
    name: str
    kind: WorkerKind
    state: WorkerState
    cancellation_requested: bool
    alive: bool
    started_at: float
    stopped_at: float | None


@dataclass
class _WorkerRecord:
    name: str
    kind: WorkerKind
    token: CancellationToken
    thread: Thread
    state: WorkerState
    started_at: float
    stop_hook: StopHook | None = None
    stopped_at: float | None = None


class WorkerManager:
    """Owns every long-running runtime worker and its lifecycle."""

    def __init__(self, bus: RuntimeBus) -> None:
        self._bus = bus
        self._lock = Lock()
        self._workers: dict[WorkerKind, _WorkerRecord] = {}
        self._shutting_down = False

    def start(
        self,
        *,
        name: str,
        kind: WorkerKind,
        target: WorkerTarget,
        stop_hook: StopHook | None = None,
    ) -> CancellationToken:
        with self._lock:
            if self._shutting_down:
                raise RuntimeError("Worker manager is shutting down.")
            current = self._workers.get(kind)
            if current is not None and current.thread.is_alive():
                raise RuntimeError(
                    f"{kind.value} worker '{current.name}' is already active."
                )

            token = CancellationToken()
            thread = Thread(
                target=self._run_worker,
                args=(kind, name, token, target),
                name=f"flyff-{name}",
                daemon=False,
            )
            record = _WorkerRecord(
                name=name,
                kind=kind,
                token=token,
                thread=thread,
                state=WorkerState.STARTING,
                started_at=monotonic(),
                stop_hook=stop_hook,
            )
            self._workers[kind] = record
            thread.start()
            return token

    def _run_worker(
        self,
        kind: WorkerKind,
        name: str,
        token: CancellationToken,
        target: WorkerTarget,
    ) -> None:
        self._set_state(kind, name, WorkerState.RUNNING)
        try:
            result = target(token)
        except WorkerCancelled:
            self._set_state(kind, name, WorkerState.COMPLETED)
            self._bus.complete(name)
        except Exception:  # noqa: BLE001 - worker boundary must contain failures.
            failure = WorkerFailure(
                worker_name=name,
                lifecycle_state=WorkerState.FAILED.value,
                cancellation_requested=token.cancelled,
                traceback=traceback.format_exc(),
                failed_at=datetime.now(timezone.utc),
            )
            self._set_state(kind, name, WorkerState.FAILED)
            self._bus.fail(failure)
        else:
            self._set_state(kind, name, WorkerState.COMPLETED)
            self._bus.complete(name, result)

    def _set_state(
        self,
        kind: WorkerKind,
        name: str,
        state: WorkerState,
    ) -> None:
        with self._lock:
            record = self._workers.get(kind)
            if record is None or record.name != name:
                return
            record.state = state
            if state in (WorkerState.COMPLETED, WorkerState.FAILED):
                record.stopped_at = monotonic()

    def stop(self, kind: WorkerKind) -> bool:
        with self._lock:
            record = self._workers.get(kind)
            if record is None or not record.thread.is_alive():
                return False
            record.state = WorkerState.STOPPING
            record.token.cancel()
            stop_hook = record.stop_hook

        if stop_hook is not None:
            try:
                stop_hook()
            except Exception:  # noqa: BLE001 - stop hooks must be contained.
                self._bus.fail(
                    WorkerFailure(
                        worker_name=record.name,
                        lifecycle_state=WorkerState.STOPPING.value,
                        cancellation_requested=True,
                        traceback=traceback.format_exc(),
                        failed_at=datetime.now(timezone.utc),
                    )
                )
        return True

    def join(self, kind: WorkerKind, timeout: float) -> bool:
        with self._lock:
            record = self._workers.get(kind)
        if record is None:
            return True
        record.thread.join(max(0.0, timeout))
        return not record.thread.is_alive()

    def stop_and_join(self, kind: WorkerKind, timeout: float) -> bool:
        self.stop(kind)
        return self.join(kind, timeout)

    def snapshot(self, kind: WorkerKind) -> WorkerSnapshot | None:
        with self._lock:
            record = self._workers.get(kind)
            if record is None:
                return None
            return WorkerSnapshot(
                name=record.name,
                kind=record.kind,
                state=record.state,
                cancellation_requested=record.token.cancelled,
                alive=record.thread.is_alive(),
                started_at=record.started_at,
                stopped_at=record.stopped_at,
            )

    def is_active(self, kind: WorkerKind) -> bool:
        snapshot = self.snapshot(kind)
        return snapshot is not None and snapshot.alive

    def shutdown(self, timeout: float = 5.0) -> dict[WorkerKind, bool]:
        with self._lock:
            self._shutting_down = True
        for kind in WorkerKind:
            self.stop(kind)

        deadline = monotonic() + max(0.0, timeout)
        results: dict[WorkerKind, bool] = {}
        for kind in (WorkerKind.CONTROL, WorkerKind.CAPTURE):
            remaining = max(0.0, deadline - monotonic())
            results[kind] = self.join(kind, remaining)
        return results
