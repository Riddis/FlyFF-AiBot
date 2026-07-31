from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from time import monotonic, sleep

from .native_world import NativeWorldFrame


@dataclass(frozen=True, slots=True)
class CastCandidate:
    base_address: int
    species_id: int


@dataclass(frozen=True, slots=True)
class CastWindow:
    started_at: float
    candidates: tuple[CastCandidate, ...]


@dataclass(frozen=True, slots=True)
class NativeKillResult:
    confirmed: tuple[CastCandidate, ...]
    polls: int
    successful_reads: int
    failed_reads: int
    cancelled: bool
    elapsed_seconds: float

    @property
    def kill_count(self) -> int:
        return len(self.confirmed)


class NativeKillTracker:
    """Confirm cast-scoped kills from repeated native actor absence."""

    def __init__(
        self,
        *,
        minimum_absence_seconds: float = 0.85,
        result_timeout_seconds: float = 2.0,
        poll_seconds: float = 0.05,
        dedupe_seconds: float = 4.0,
        clock: Callable[[], float] = monotonic,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        if minimum_absence_seconds <= 0.0:
            raise ValueError("minimum_absence_seconds must be positive")
        if result_timeout_seconds < minimum_absence_seconds:
            raise ValueError(
                "result_timeout_seconds cannot be shorter than minimum absence"
            )
        if poll_seconds <= 0.0 or dedupe_seconds <= 0.0:
            raise ValueError("poll_seconds and dedupe_seconds must be positive")
        self.minimum_absence_seconds = float(minimum_absence_seconds)
        self.result_timeout_seconds = float(result_timeout_seconds)
        self.poll_seconds = float(poll_seconds)
        self.dedupe_seconds = float(dedupe_seconds)
        self._clock = clock
        self._sleep = sleeper
        self._recent_confirmations: dict[tuple[int, int], float] = {}

    def begin_cast(
        self,
        frame: NativeWorldFrame,
        *,
        eva_radius_native: float,
    ) -> CastWindow:
        if eva_radius_native <= 0.0:
            raise ValueError("eva_radius_native must be positive")
        candidates = tuple(
            CastCandidate(actor.base_address, actor.species_id)
            for actor in frame.actors
            if actor.distance_native <= float(eva_radius_native)
        )
        return CastWindow(started_at=self._clock(), candidates=candidates)

    @staticmethod
    def _cancelled(cancellation: object | None) -> bool:
        if cancellation is None:
            return False
        cancelled = getattr(cancellation, "cancelled", None)
        if cancelled is not None:
            return bool(cancelled() if callable(cancelled) else cancelled)
        is_set = getattr(cancellation, "is_set", None)
        return bool(is_set()) if callable(is_set) else False

    def _wait(self, cancellation: object | None) -> None:
        wait = None if cancellation is None else getattr(cancellation, "wait", None)
        if callable(wait):
            wait(self.poll_seconds)
        else:
            self._sleep(self.poll_seconds)

    def confirm_cast(
        self,
        window: CastWindow,
        read_frame: Callable[[], NativeWorldFrame],
        *,
        cancellation: object | None = None,
    ) -> NativeKillResult:
        now = self._clock()
        self._recent_confirmations = {
            key: confirmed_at
            for key, confirmed_at in self._recent_confirmations.items()
            if now - confirmed_at < self.dedupe_seconds
        }
        pending = {
            (candidate.base_address, candidate.species_id): candidate
            for candidate in window.candidates
            if (candidate.base_address, candidate.species_id)
            not in self._recent_confirmations
        }
        absence_reads = {key: 0 for key in pending}
        confirmed: list[CastCandidate] = []
        polls = successful = failed = 0
        deadline = window.started_at + self.result_timeout_seconds

        while pending and self._clock() < deadline:
            if self._cancelled(cancellation):
                return NativeKillResult(
                    confirmed=tuple(confirmed),
                    polls=polls,
                    successful_reads=successful,
                    failed_reads=failed,
                    cancelled=True,
                    elapsed_seconds=max(0.0, self._clock() - window.started_at),
                )
            self._wait(cancellation)
            polls += 1
            if self._cancelled(cancellation):
                continue
            try:
                frame = read_frame()
            except Exception:
                failed += 1
                continue
            successful += 1
            present = {(actor.base_address, actor.species_id) for actor in frame.actors}
            if self._clock() - window.started_at < self.minimum_absence_seconds:
                continue
            for key, candidate in tuple(pending.items()):
                if key in present:
                    absence_reads[key] = 0
                    continue
                absence_reads[key] += 1
                if absence_reads[key] < 2:
                    continue
                confirmed.append(candidate)
                self._recent_confirmations[key] = self._clock()
                pending.pop(key)

        return NativeKillResult(
            confirmed=tuple(confirmed),
            polls=polls,
            successful_reads=successful,
            failed_reads=failed,
            cancelled=self._cancelled(cancellation),
            elapsed_seconds=max(0.0, self._clock() - window.started_at),
        )


class OcrDiagnosticOutcome(str, Enum):
    OK = "ok"
    MISS = "miss"
    DECREASE = "decrease"
    OUTLIER = "outlier"


@dataclass(frozen=True, slots=True)
class OcrDiagnostic:
    outcome: OcrDiagnosticOutcome
    value: int | None
    previous: int | None
    delta: int | None


class OcrKillDiagnostics:
    """Track OCR health without producing or modifying native kill reward."""

    def __init__(self, *, maximum_delta: int = 32) -> None:
        if isinstance(maximum_delta, bool) or maximum_delta < 1:
            raise ValueError("maximum_delta must be a positive integer")
        self.maximum_delta = int(maximum_delta)
        self._previous: int | None = None

    def observe(self, value: int | None) -> OcrDiagnostic:
        previous = self._previous
        if value is None:
            return OcrDiagnostic(OcrDiagnosticOutcome.MISS, None, previous, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("OCR kill count must be a non-negative integer or None")
        if previous is None:
            self._previous = value
            return OcrDiagnostic(OcrDiagnosticOutcome.OK, value, None, 0)
        delta = value - previous
        if delta < 0:
            return OcrDiagnostic(
                OcrDiagnosticOutcome.DECREASE,
                value,
                previous,
                delta,
            )
        if delta > self.maximum_delta:
            return OcrDiagnostic(
                OcrDiagnosticOutcome.OUTLIER,
                value,
                previous,
                delta,
            )
        self._previous = value
        return OcrDiagnostic(OcrDiagnosticOutcome.OK, value, previous, delta)
