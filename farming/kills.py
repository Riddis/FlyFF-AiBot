from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from time import monotonic, sleep

from .native_world import NativeWorldFrame


@dataclass(frozen=True, slots=True)
class CastCandidate:
    base_address: int
    species_id: int
    initial_hp: int = 0


@dataclass(frozen=True, slots=True)
class CastWindow:
    started_at: float
    candidates: tuple[CastCandidate, ...]


@dataclass(frozen=True, slots=True)
class CandidateKillDiagnostic:
    base_address: int
    species_id: int
    present_reads: int
    absent_reads: int
    maximum_consecutive_absence: int
    minimum_seen_hp: int | None
    last_seen_hp: int | None
    confirmed: bool
    initial_hp: int | None = None
    zero_hp_reads: int = 0
    maximum_consecutive_zero_hp: int = 0
    hp_decreased: bool = False


@dataclass(frozen=True, slots=True)
class NativeKillResult:
    confirmed: tuple[CastCandidate, ...]
    polls: int
    successful_reads: int
    failed_reads: int
    cancelled: bool
    elapsed_seconds: float
    diagnostics: tuple[CandidateKillDiagnostic, ...] = ()

    @property
    def kill_count(self) -> int:
        return len(self.confirmed)


class NativeKillTracker:
    """Confirm EVA kills from same-slot live-HP transitions.

    The FlyFF client keeps an actor slot allocated after death. The same actor
    base therefore remains readable with live HP set to zero until the slot is
    reused by a respawn. Disappearance is not kill evidence and EVA range is
    deliberately not assumed here. ``begin_cast`` snapshots every selected,
    living actor already admitted by the native world's broad vision radius;
    ``confirm_cast`` then requires the same base/species to reach HP zero.
    """

    def __init__(
        self,
        *,
        zero_hp_confirmation_reads: int = 2,
        result_timeout_seconds: float = 2.0,
        poll_seconds: float = 0.05,
        # Backward-compatible constructor inputs from the old absence tracker.
        minimum_absence_seconds: float | None = None,
        dedupe_seconds: float | None = None,
        clock: Callable[[], float] = monotonic,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        if (
            isinstance(zero_hp_confirmation_reads, bool)
            or not isinstance(zero_hp_confirmation_reads, int)
            or zero_hp_confirmation_reads < 1
        ):
            raise ValueError("zero_hp_confirmation_reads must be a positive integer")
        if result_timeout_seconds <= 0.0 or poll_seconds <= 0.0:
            raise ValueError("result_timeout_seconds and poll_seconds must be positive")
        if minimum_absence_seconds is not None and minimum_absence_seconds <= 0.0:
            raise ValueError("minimum_absence_seconds must be positive when provided")
        if dedupe_seconds is not None and dedupe_seconds <= 0.0:
            raise ValueError("dedupe_seconds must be positive when provided")
        self.zero_hp_confirmation_reads = int(zero_hp_confirmation_reads)
        self.result_timeout_seconds = float(result_timeout_seconds)
        self.poll_seconds = float(poll_seconds)
        self._clock = clock
        self._sleep = sleeper

    def begin_cast(self, frame: NativeWorldFrame) -> CastWindow:
        # ``frame.actors`` already contains only selected, living monsters inside
        # the configured broad native vision radius. Do not apply an assumed EVA
        # radius: the policy must learn which distances actually produce kills.
        candidates = tuple(
            CastCandidate(
                base_address=int(actor.base_address),
                species_id=int(actor.species_id),
                initial_hp=int(actor.hp),
            )
            for actor in frame.actors
            if int(actor.hp) > 0
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

    @staticmethod
    def _tracked_actor_map(frame: NativeWorldFrame) -> dict[tuple[int, int], object]:
        tracked = frame.tracked_actors if frame.tracked_actors else frame.actors
        return {
            (int(actor.base_address), int(actor.species_id)): actor
            for actor in tracked
        }

    def confirm_cast(
        self,
        window: CastWindow,
        read_frame: Callable[[], NativeWorldFrame] | None = None,
        *,
        read_actor_hp_states: (
            Callable[
                [tuple[tuple[int, int], ...]],
                Mapping[tuple[int, int], int],
            ]
            | None
        ) = None,
        cancellation: object | None = None,
    ) -> NativeKillResult:
        candidate_by_key = {
            (candidate.base_address, candidate.species_id): candidate
            for candidate in window.candidates
            if candidate.initial_hp > 0
        }
        pending = dict(candidate_by_key)
        diagnostic_state = {
            key: {
                "present_reads": 0,
                "absent_reads": 0,
                "maximum_consecutive_absence": 0,
                "consecutive_absence": 0,
                "minimum_seen_hp": candidate.initial_hp,
                "last_seen_hp": candidate.initial_hp,
                "zero_hp_reads": 0,
                "maximum_consecutive_zero_hp": 0,
                "consecutive_zero_hp": 0,
                "hp_decreased": False,
            }
            for key, candidate in pending.items()
        }
        confirmed: list[CastCandidate] = []

        def diagnostics() -> tuple[CandidateKillDiagnostic, ...]:
            confirmed_keys = {
                (candidate.base_address, candidate.species_id)
                for candidate in confirmed
            }
            result: list[CandidateKillDiagnostic] = []
            for key, state in diagnostic_state.items():
                result.append(
                    CandidateKillDiagnostic(
                        base_address=key[0],
                        species_id=key[1],
                        present_reads=int(state["present_reads"]),
                        absent_reads=int(state["absent_reads"]),
                        maximum_consecutive_absence=int(
                            state["maximum_consecutive_absence"]
                        ),
                        minimum_seen_hp=(
                            None
                            if state["minimum_seen_hp"] is None
                            else int(state["minimum_seen_hp"])
                        ),
                        last_seen_hp=(
                            None
                            if state["last_seen_hp"] is None
                            else int(state["last_seen_hp"])
                        ),
                        confirmed=key in confirmed_keys,
                        initial_hp=int(candidate_by_key[key].initial_hp),
                        zero_hp_reads=int(state["zero_hp_reads"]),
                        maximum_consecutive_zero_hp=int(
                            state["maximum_consecutive_zero_hp"]
                        ),
                        hp_decreased=bool(state["hp_decreased"]),
                    )
                )
            return tuple(result)

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
                    diagnostics=diagnostics(),
                )
            self._wait(cancellation)
            polls += 1
            if self._cancelled(cancellation):
                continue
            try:
                if read_actor_hp_states is not None:
                    hp_by_key = {
                        (int(base), int(species)): int(hp)
                        for (base, species), hp in read_actor_hp_states(
                            tuple(pending)
                        ).items()
                    }
                    tracked = None
                else:
                    if read_frame is None:
                        raise RuntimeError(
                            "read_frame or read_actor_hp_states is required"
                        )
                    frame = read_frame()
                    tracked = self._tracked_actor_map(frame)
                    hp_by_key = {}
            except Exception:
                failed += 1
                continue
            successful += 1
            for key, candidate in tuple(pending.items()):
                state = diagnostic_state[key]
                if read_actor_hp_states is not None:
                    hp = hp_by_key.get(key)
                    actor = None
                else:
                    assert tracked is not None
                    actor = tracked.get(key)
                    hp = None if actor is None else int(getattr(actor, "hp"))
                if hp is None:
                    state["absent_reads"] = int(state["absent_reads"]) + 1
                    state["consecutive_absence"] = int(
                        state["consecutive_absence"]
                    ) + 1
                    state["maximum_consecutive_absence"] = max(
                        int(state["maximum_consecutive_absence"]),
                        int(state["consecutive_absence"]),
                    )
                    state["consecutive_zero_hp"] = 0
                    # Missing slots are diagnostic only. This client does not
                    # remove the actor object on death, so absence cannot reward.
                    continue

                state["present_reads"] = int(state["present_reads"]) + 1
                state["consecutive_absence"] = 0
                hp = int(hp)
                state["last_seen_hp"] = hp
                minimum = state["minimum_seen_hp"]
                state["minimum_seen_hp"] = hp if minimum is None else min(int(minimum), hp)
                if hp < candidate.initial_hp:
                    state["hp_decreased"] = True
                if hp <= 0:
                    state["zero_hp_reads"] = int(state["zero_hp_reads"]) + 1
                    state["consecutive_zero_hp"] = int(
                        state["consecutive_zero_hp"]
                    ) + 1
                    state["maximum_consecutive_zero_hp"] = max(
                        int(state["maximum_consecutive_zero_hp"]),
                        int(state["consecutive_zero_hp"]),
                    )
                    if (
                        int(state["consecutive_zero_hp"])
                        >= self.zero_hp_confirmation_reads
                    ):
                        confirmed.append(candidate)
                        pending.pop(key)
                else:
                    state["consecutive_zero_hp"] = 0

        return NativeKillResult(
            confirmed=tuple(confirmed),
            polls=polls,
            successful_reads=successful,
            failed_reads=failed,
            cancelled=self._cancelled(cancellation),
            elapsed_seconds=max(0.0, self._clock() - window.started_at),
            diagnostics=diagnostics(),
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
