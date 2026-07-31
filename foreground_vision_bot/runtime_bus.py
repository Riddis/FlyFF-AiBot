from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Event, Lock
from time import monotonic
from typing import Any


@dataclass
class ConfirmationRequest:
    request_id: int
    frame: Any
    angle_deg: float
    confidence: float
    context: str
    completed: Event
    result: bool | None = None




@dataclass
class MapperRecoveryRequest:
    request_id: int
    map_name: str
    reason: str
    can_retry_in_place: bool
    requires_spawn_reset: bool
    completed: Event
    result: str | None = None


@dataclass(frozen=True)
class TaskCompletion:
    worker_name: str
    result: Any
    completed_at: datetime
    session_id: int | None = None


@dataclass(frozen=True)
class WorkerFailure:
    worker_name: str
    lifecycle_state: str
    cancellation_requested: bool
    traceback: str
    failed_at: datetime
    session_id: int | None = None


@dataclass(frozen=True)
class RuntimeStatus:
    """One user-facing status value scoped to a runtime session."""

    message: str
    session_id: int | None = None


class RuntimeBus:
    """Bounded high-rate state and reliable low-rate runtime delivery."""

    def __init__(self, *, max_logs: int = 1000) -> None:
        if max_logs < 1:
            raise ValueError("max_logs must be at least one")

        self._lock = Lock()
        self._closed = False
        self._latest: dict[str, tuple[int, Any]] = {}
        self._versions: dict[str, int] = {}
        self._logs: deque[tuple[str, str]] = deque(maxlen=max_logs)
        self._dropped_logs = 0

        # Completions, failures, and confirmations are low-rate lifecycle
        # messages. They must never be silently evicted.
        self._completions: deque[TaskCompletion] = deque()
        self._failures: deque[WorkerFailure] = deque()
        self._confirmations: deque[ConfirmationRequest] = deque()
        self._mapper_recoveries: deque[MapperRecoveryRequest] = deque()
        self._next_confirmation_id = 1
        self._next_mapper_recovery_id = 1
        self._heartbeats: dict[str, float] = {}

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pending = list(self._confirmations)
            pending_recoveries = list(self._mapper_recoveries)
            self._confirmations.clear()
            self._mapper_recoveries.clear()

        for request in (*pending, *pending_recoveries):
            request.result = None
            request.completed.set()

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def dropped_logs(self) -> int:
        with self._lock:
            return self._dropped_logs

    def publish_latest(self, key: str, value: Any) -> None:
        with self._lock:
            if self._closed:
                return
            version = self._versions.get(key, 0) + 1
            self._versions[key] = version
            self._latest[key] = (version, value)

    def publish_status(
        self,
        key: str,
        message: str,
        *,
        session_id: int | None = None,
    ) -> None:
        self.publish_latest(
            key,
            RuntimeStatus(
                message=str(message),
                session_id=session_id,
            ),
        )

    def read_latest(
        self,
        key: str,
        last_version: int = 0,
    ) -> tuple[int, Any | None]:
        with self._lock:
            item = self._latest.get(key)
            if item is None or item[0] <= last_version:
                return last_version, None
            return item

    def log(self, message: str, level: str = "msg") -> None:
        with self._lock:
            if self._closed:
                return
            if len(self._logs) == self._logs.maxlen:
                self._dropped_logs += 1
            self._logs.append((level, str(message)))

    def drain_logs(self, maximum: int = 100) -> list[tuple[str, str]]:
        if maximum < 1:
            return []
        items: list[tuple[str, str]] = []
        with self._lock:
            while self._logs and len(items) < maximum:
                items.append(self._logs.popleft())
        return items

    def complete(
        self,
        worker_name: str,
        result: Any = None,
        *,
        session_id: int | None = None,
    ) -> None:
        completion = TaskCompletion(
            worker_name=worker_name,
            result=result,
            completed_at=datetime.now(timezone.utc),
            session_id=session_id,
        )
        with self._lock:
            if not self._closed:
                self._completions.append(completion)

    def drain_completions(self) -> list[TaskCompletion]:
        with self._lock:
            items = list(self._completions)
            self._completions.clear()
        return items

    def fail(self, failure: WorkerFailure) -> None:
        with self._lock:
            if not self._closed:
                self._failures.append(failure)

    def drain_failures(self) -> list[WorkerFailure]:
        with self._lock:
            items = list(self._failures)
            self._failures.clear()
        return items

    def heartbeat(self, worker: str) -> None:
        with self._lock:
            if not self._closed:
                self._heartbeats[worker] = monotonic()

    def heartbeats(self) -> dict[str, float]:
        with self._lock:
            return dict(self._heartbeats)

    def request_heading_confirmation(
        self,
        frame: Any,
        angle_deg: float,
        confidence: float,
        context: str,
        *,
        timeout: float = 300.0,
        cancellation_event: Event | None = None,
    ) -> bool | None:
        with self._lock:
            if self._closed:
                return None
            request = ConfirmationRequest(
                request_id=self._next_confirmation_id,
                frame=frame,
                angle_deg=float(angle_deg),
                confidence=float(confidence),
                context=str(context),
                completed=Event(),
            )
            self._next_confirmation_id += 1
            self._confirmations.append(request)

        deadline = monotonic() + max(0.0, timeout)
        while not request.completed.wait(timeout=0.1):
            if cancellation_event is not None and cancellation_event.is_set():
                self.cancel_confirmation(request.request_id)
                return None
            if monotonic() >= deadline:
                self.cancel_confirmation(request.request_id)
                return None
        return request.result

    def pop_confirmation(self) -> ConfirmationRequest | None:
        with self._lock:
            if not self._confirmations:
                return None
            return self._confirmations.popleft()

    def resolve_confirmation(
        self,
        request: ConfirmationRequest,
        result: bool | None,
    ) -> None:
        request.result = result
        request.completed.set()

    def cancel_confirmation(self, request_id: int) -> bool:
        request: ConfirmationRequest | None = None
        with self._lock:
            for candidate in self._confirmations:
                if candidate.request_id == request_id:
                    request = candidate
                    self._confirmations.remove(candidate)
                    break
        if request is None:
            return False
        request.result = None
        request.completed.set()
        return True
    def request_mapper_recovery(
        self,
        *,
        map_name: str,
        reason: str,
        can_retry_in_place: bool,
        requires_spawn_reset: bool,
        timeout: float = 1800.0,
        cancellation_event: Event | None = None,
    ) -> str | None:
        """Pause mapper control until the GUI chooses a safe recovery action."""
        with self._lock:
            if self._closed:
                return None
            request = MapperRecoveryRequest(
                request_id=self._next_mapper_recovery_id,
                map_name=str(map_name),
                reason=str(reason),
                can_retry_in_place=bool(can_retry_in_place),
                requires_spawn_reset=bool(requires_spawn_reset),
                completed=Event(),
            )
            self._next_mapper_recovery_id += 1
            self._mapper_recoveries.append(request)

        deadline = monotonic() + max(0.0, timeout)
        while not request.completed.wait(timeout=0.1):
            if cancellation_event is not None and cancellation_event.is_set():
                self.cancel_mapper_recovery(request.request_id)
                return None
            if monotonic() >= deadline:
                self.cancel_mapper_recovery(request.request_id)
                return None
        return request.result

    def pop_mapper_recovery(self) -> MapperRecoveryRequest | None:
        with self._lock:
            if not self._mapper_recoveries:
                return None
            return self._mapper_recoveries.popleft()

    def resolve_mapper_recovery(
        self,
        request: MapperRecoveryRequest,
        result: str | None,
    ) -> None:
        request.result = result
        request.completed.set()

    def cancel_mapper_recovery(self, request_id: int) -> bool:
        request: MapperRecoveryRequest | None = None
        with self._lock:
            for candidate in self._mapper_recoveries:
                if candidate.request_id == request_id:
                    request = candidate
                    self._mapper_recoveries.remove(candidate)
                    break
        if request is None:
            return False
        request.result = None
        request.completed.set()
        return True
