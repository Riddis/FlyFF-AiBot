from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import Iterable, Protocol

from position.IndependentNativeReader import IndependentActorSlotRead


class ProcessMemoryReader(Protocol):
    def read(self, address: int, size: int) -> bytes: ...


@dataclass(slots=True)
class _OffsetEvidence:
    offset: int
    live_matches: int = 0
    live_samples: int = 0
    death_matches: int = 0
    death_samples: int = 0
    inactive_matches: int = 0
    inactive_samples: int = 0
    dormant_matches: int = 0
    dormant_samples: int = 0
    reappearance_matches: int = 0
    reappearance_samples: int = 0
    species: set[int] = field(default_factory=set)
    bases: set[int] = field(default_factory=set)

    @property
    def live_ratio(self) -> float:
        return 0.0 if self.live_samples <= 0 else self.live_matches / self.live_samples

    @property
    def death_ratio(self) -> float:
        return 0.0 if self.death_samples <= 0 else self.death_matches / self.death_samples

    @property
    def inactive_ratio(self) -> float:
        return 0.0 if self.inactive_samples <= 0 else self.inactive_matches / self.inactive_samples

    @property
    def dormant_ratio(self) -> float:
        return 0.0 if self.dormant_samples <= 0 else self.dormant_matches / self.dormant_samples

    @property
    def reappearance_ratio(self) -> float:
        return (
            0.0
            if self.reappearance_samples <= 0
            else self.reappearance_matches / self.reappearance_samples
        )

    @property
    def validated(self) -> bool:
        """Return True only for strong instantiated/dormant transition evidence.

        Zero-HP samples are deliberately not required to clear. A killed actor can
        remain instantiated for a while, so death is reported separately and is
        never treated as the negative state for this field.
        """

        # Reappearance is useful diagnostics but not a hard gate. Actor bases
        # are reusable pool slots, so a later appearance at the same address is
        # not guaranteed to represent the same monster lifecycle. Strong live
        # matching plus repeated far/dormant clears across many bases is the
        # authoritative longitudinal signal.
        inactive_ok = (
            self.inactive_samples < 4 or self.inactive_ratio <= 0.10
        )
        species_ok = len(self.species) >= 2 or self.live_samples >= 64
        return bool(
            self.live_samples >= 32
            and self.live_ratio >= 0.90
            and self.dormant_samples >= 8
            and self.dormant_ratio <= 0.10
            and inactive_ok
            and species_ok
            and len(self.bases) >= 16
        )

    def score(self) -> float:
        dormant_clear = 1.0 - self.dormant_ratio if self.dormant_samples else 0.0
        release_clear = 1.0 - self.inactive_ratio if self.inactive_samples else 0.0
        reappearance = (
            self.reappearance_ratio if self.reappearance_samples else self.live_ratio
        )
        sample_weight = min(1.0, self.live_samples / 32.0)
        transition_weight = min(1.0, (self.dormant_samples + self.inactive_samples) / 12.0)
        species_bonus = min(0.10, 0.05 * max(0, len(self.species) - 1))
        return (
            0.40 * self.live_ratio
            + 0.30 * dormant_clear
            + 0.10 * release_clear
            + 0.15 * reappearance
            + 0.03 * sample_weight
            + 0.02 * transition_weight
            + species_bonus
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "offset": self.offset,
            "offset_hex": f"0x{self.offset:X}",
            "validated": self.validated,
            "score": round(self.score(), 6),
            "live_matches": self.live_matches,
            "live_samples": self.live_samples,
            "live_match_ratio": round(self.live_ratio, 6),
            "zero_hp_matches": self.death_matches,
            "zero_hp_samples": self.death_samples,
            "zero_hp_match_ratio": round(self.death_ratio, 6),
            "inactive_matches": self.inactive_matches,
            "inactive_samples": self.inactive_samples,
            "inactive_match_ratio": round(self.inactive_ratio, 6),
            "dormant_matches": self.dormant_matches,
            "dormant_samples": self.dormant_samples,
            "dormant_match_ratio": round(self.dormant_ratio, 6),
            "reappearance_matches": self.reappearance_matches,
            "reappearance_samples": self.reappearance_samples,
            "reappearance_match_ratio": round(self.reappearance_ratio, 6),
            "species": sorted(self.species),
            "unique_bases": len(self.bases),
        }


@dataclass(frozen=True, slots=True)
class _LiveBaseline:
    species: int
    captured_ms: int
    candidate_offsets: tuple[int, ...]


@dataclass(slots=True)
class _StateHistory:
    signature: tuple[int, int, int, int, int]
    last_changed_ms: int
    last_near_ms: int | None = None
    last_dormant_sample_ms: int = -10_000_000


class ActiveFieldProfiler:
    """Search for the old instantiated/loaded duplicate-species field.

    The pre-maintenance client used a 32-bit field which equalled the ordinary
    species value while the actor was instantiated, then cleared or changed when
    the reusable actor slot became dormant/reallocated. The offset moved or became
    unusable after maintenance, so this profiler discovers it from transitions
    instead of trusting the historical +0x1DBC value.

    The profiler never hides an actor itself. Recorder 1.11 may pass a strongly
    validated candidate to the native reader, which independently checks the
    offset against the current process before enabling hot/cold polling.
    """

    def __init__(
        self,
        memory: ProcessMemoryReader,
        *,
        actor_stride: int | None,
        object_span: int,
        excluded_offsets: Iterable[int],
        maximum_candidates: int = 256,
    ) -> None:
        self._memory = memory
        stride = 0 if actor_stride is None else int(actor_stride)
        requested = max(4, int(object_span))
        # Never cross into the next actor. This explicitly prevents the old
        # false +0x217C = +0x174 + 0x2008 cross-slot species alias.
        if stride > 0:
            requested = min(requested, stride)
        self.span = min(max(4, requested), 0x4000)
        self.span -= self.span % 4
        self.excluded_offsets = {
            int(value) for value in excluded_offsets if 0 <= int(value) < self.span
        }
        self.maximum_candidates = max(16, int(maximum_candidates))
        self._evidence: dict[int, _OffsetEvidence] = {}
        self._live_baselines: dict[int, _LiveBaseline] = {}
        self._state_history: dict[int, _StateHistory] = {}
        self._round_robin_index = 0
        self._dormant_round_robin_index = 0
        self.read_failures = 0
        self.live_captures = 0
        self.transition_captures = 0
        self._historical_offsets = (0x1DBC, 0x217C)
        self._seen_transition_events: set[tuple[str, int, int, int]] = set()

    @staticmethod
    def _words(data: bytes) -> tuple[int, ...]:
        count = len(data) // 4
        if count <= 0:
            return ()
        return struct.unpack(f"<{count}i", data[: count * 4])

    def _read_words(self, base: int) -> tuple[int, ...] | None:
        try:
            data = self._memory.read(int(base), self.span)
        except Exception:
            self.read_failures += 1
            return None
        if len(data) != self.span:
            self.read_failures += 1
            return None
        return self._words(data)

    def _eligible_offsets(self) -> range:
        return range(0, self.span, 4)

    def _observe_words(
        self,
        *,
        base: int,
        expected_species: int,
        words: tuple[int, ...],
        label: str,
        candidate_offsets: Iterable[int] | None = None,
    ) -> tuple[int, ...]:
        if expected_species <= 0:
            return ()
        if candidate_offsets is None:
            offsets = self._eligible_offsets()
        else:
            offsets = tuple(int(value) for value in candidate_offsets)

        matched: list[int] = []
        for offset in offsets:
            if offset in self.excluded_offsets or offset < 0 or offset >= self.span:
                continue
            index = offset // 4
            if index >= len(words):
                continue
            value = int(words[index])
            evidence = self._evidence.get(offset)
            if evidence is None:
                # During live scans only create evidence for fields which at least
                # once equal the actor species. Transition scans are limited to a
                # prior live candidate set, so they never create noise candidates.
                if label != "live" or value != expected_species:
                    continue
                evidence = _OffsetEvidence(offset=offset)
                self._evidence[offset] = evidence
            evidence.species.add(int(expected_species))
            evidence.bases.add(int(base))
            if label == "live":
                evidence.live_samples += 1
                evidence.live_matches += int(value == expected_species)
            elif label == "death":
                evidence.death_samples += 1
                evidence.death_matches += int(value == expected_species)
            elif label == "inactive":
                evidence.inactive_samples += 1
                evidence.inactive_matches += int(value == expected_species)
            elif label == "dormant":
                evidence.dormant_samples += 1
                evidence.dormant_matches += int(value == expected_species)
            elif label == "reappearance":
                evidence.reappearance_samples += 1
                evidence.reappearance_matches += int(value == expected_species)
            else:
                raise ValueError(f"Unsupported active-field label: {label}")
            if value == expected_species:
                matched.append(offset)

        if len(self._evidence) > self.maximum_candidates:
            # Retain candidates with the strongest live support. This is a safety
            # valve for fields containing many coincidental small integers.
            ranked = sorted(
                self._evidence.values(),
                key=lambda item: (
                    item.validated,
                    item.score(),
                    item.live_matches,
                    item.live_samples,
                    -item.offset,
                ),
                reverse=True,
            )[: self.maximum_candidates]
            keep = {item.offset for item in ranked}
            self._evidence = {
                offset: item for offset, item in self._evidence.items() if offset in keep
            }
            matched = [offset for offset in matched if offset in keep]
        return tuple(sorted(set(matched)))

    def sample_live_states(
        self,
        states: Iterable[IndependentActorSlotRead],
        *,
        elapsed_ms: int,
        maximum_samples: int = 8,
        maximum_distance_native: float | None = None,
    ) -> int:
        state_list = list(states)
        for state in state_list:
            if int(state.species) <= 0 or int(state.hp) < 0:
                continue
            signature = (
                int(state.species),
                int(state.hp),
                int(round(float(state.x) * 20.0)),
                int(round(float(state.y) * 20.0)),
                int(round(float(state.z) * 20.0)),
            )
            history = self._state_history.get(int(state.base))
            if history is None:
                history = _StateHistory(signature=signature, last_changed_ms=int(elapsed_ms))
                self._state_history[int(state.base)] = history
            elif history.signature != signature:
                history.signature = signature
                history.last_changed_ms = int(elapsed_ms)
            if (
                maximum_distance_native is not None
                and float(state.distance_native) <= float(maximum_distance_native)
            ):
                history.last_near_ms = int(elapsed_ms)

        living = [
            state
            for state in state_list
            if int(state.species) > 0
            and int(state.hp) > 0
            and math.isfinite(float(state.x))
            and math.isfinite(float(state.z))
            and (
                maximum_distance_native is None
                or float(state.distance_native) <= float(maximum_distance_native)
            )
        ]
        if not living:
            return 0
        living.sort(key=lambda state: int(state.base))
        start = self._round_robin_index % len(living)
        selected = [
            living[(start + index) % len(living)]
            for index in range(min(max(1, int(maximum_samples)), len(living)))
        ]
        self._round_robin_index = (start + len(selected)) % len(living)
        captured = 0
        for state in selected:
            words = self._read_words(int(state.base))
            if words is None:
                continue
            matched = self._observe_words(
                base=int(state.base),
                expected_species=int(state.species),
                words=words,
                label="live",
            )
            self._live_baselines[int(state.base)] = _LiveBaseline(
                species=int(state.species),
                captured_ms=int(elapsed_ms),
                candidate_offsets=matched,
            )
            captured += 1
        self.live_captures += captured
        return captured

    def sample_dormant_states(
        self,
        states: Iterable[IndependentActorSlotRead],
        *,
        elapsed_ms: int,
        minimum_distance_native: float,
        stable_milliseconds: int,
        after_near_milliseconds: int,
        maximum_samples: int = 8,
        repeat_milliseconds: int = 2000,
    ) -> int:
        candidates: list[IndependentActorSlotRead] = []
        for state in states:
            if int(state.species) <= 0 or int(state.hp) <= 0:
                continue
            if float(state.distance_native) < float(minimum_distance_native):
                continue
            history = self._state_history.get(int(state.base))
            if history is None or history.last_near_ms is None:
                continue
            if int(elapsed_ms) - history.last_changed_ms < int(stable_milliseconds):
                continue
            if int(elapsed_ms) - history.last_near_ms < int(after_near_milliseconds):
                continue
            if int(elapsed_ms) - history.last_dormant_sample_ms < int(repeat_milliseconds):
                continue
            candidates.append(state)
        if not candidates:
            return 0
        candidates.sort(key=lambda state: int(state.base))
        start = self._dormant_round_robin_index % len(candidates)
        selected = [
            candidates[(start + index) % len(candidates)]
            for index in range(min(max(1, int(maximum_samples)), len(candidates)))
        ]
        self._dormant_round_robin_index = (start + len(selected)) % len(candidates)
        captured = 0
        for state in selected:
            words = self._read_words(int(state.base))
            if words is None:
                continue
            self._observe_words(
                base=int(state.base),
                expected_species=int(state.species),
                words=words,
                label="dormant",
                candidate_offsets=tuple(self._evidence),
            )
            history = self._state_history.get(int(state.base))
            if history is not None:
                history.last_dormant_sample_ms = int(elapsed_ms)
            captured += 1
        self.transition_captures += captured
        return captured

    def observe_event(self, event: list[object], *, elapsed_ms: int) -> None:
        if not event:
            return
        kind = str(event[0])
        if kind not in {
            "death",
            "target_disappearance",
            "target_appearance",
            "respawn_candidate",
            "reuse",
        }:
            return
        try:
            base = int(event[2])
        except Exception:
            return

        baseline = self._live_baselines.get(base)
        if kind == "death":
            expected_species = int(event[3])
            label = "death"
        elif kind == "target_disappearance":
            expected_species = int(event[3])
            label = "inactive"
        elif kind == "reuse":
            expected_species = int(event[3])
            label = "inactive"
        else:
            expected_species = int(event[3])
            label = "reappearance"

        dedupe_key = (label, base, expected_species, int(elapsed_ms))
        if dedupe_key in self._seen_transition_events:
            return
        self._seen_transition_events.add(dedupe_key)
        words = self._read_words(base)
        if words is None:
            return
        candidate_offsets = None
        if label in {"death", "inactive"}:
            if baseline is None or baseline.species != expected_species:
                return
            candidate_offsets = baseline.candidate_offsets
        self._observe_words(
            base=base,
            expected_species=expected_species,
            words=words,
            label=label,
            candidate_offsets=candidate_offsets,
        )
        self.transition_captures += 1

        if label == "inactive":
            self._live_baselines.pop(base, None)
        elif label == "reappearance":
            # The normal rotating sampler will refresh the complete live baseline.
            matched = tuple(
                offset
                for offset, evidence in self._evidence.items()
                if offset < self.span
                and offset not in self.excluded_offsets
                and int(words[offset // 4]) == expected_species
            )
            self._live_baselines[base] = _LiveBaseline(
                species=expected_species,
                captured_ms=int(elapsed_ms),
                candidate_offsets=matched,
            )

    def best_validated_candidate(self) -> dict[str, object] | None:
        """Return the strongest session-proven field, or ``None``.

        The returned dictionary is intentionally serialization-friendly so the
        native reader can independently re-check bounds and current live
        matching before enabling optimized sampling.
        """

        validated = [item for item in self._evidence.values() if item.validated]
        if not validated:
            return None
        ranked = sorted(
            validated,
            key=lambda item: (
                item.score(),
                item.live_matches,
                item.dormant_samples,
                item.inactive_samples,
                len(item.bases),
                -item.offset,
            ),
            reverse=True,
        )
        best = ranked[0]
        if len(ranked) > 1 and best.score() - ranked[1].score() < 0.05:
            return None
        return best.to_dict()

    def report(self, *, top_n: int = 32) -> dict[str, object]:
        ranked = sorted(
            self._evidence.values(),
            key=lambda item: (
                item.validated,
                item.score(),
                item.live_matches,
                item.dormant_samples,
                item.inactive_samples,
                item.reappearance_samples,
                -item.offset,
            ),
            reverse=True,
        )
        top = ranked[: max(1, int(top_n))]
        validated = [item for item in top if item.validated]
        historical: dict[str, object] = {}
        for offset in self._historical_offsets:
            item = self._evidence.get(offset)
            historical[f"0x{offset:X}"] = None if item is None else item.to_dict()
        return {
            "schema_version": 1,
            "diagnostics_only": False,
            "promotion_requires_reader_revalidation": True,
            "object_span": self.span,
            "object_span_hex": f"0x{self.span:X}",
            "excluded_offsets": [f"0x{value:X}" for value in sorted(self.excluded_offsets)],
            "live_captures": self.live_captures,
            "transition_captures": self.transition_captures,
            "read_failures": self.read_failures,
            "candidate_count": len(self._evidence),
            "validated_offsets": [f"0x{item.offset:X}" for item in validated],
            "recommended_offset": (
                None if not validated else f"0x{validated[0].offset:X}"
            ),
            "historical_offsets": historical,
            "candidates": [item.to_dict() for item in top],
            "notes": [
                "The historical pre-maintenance field was int32 actor+0x1DBC.",
                "actor+0x217C is excluded by the actor-stride boundary because it is the next slot's species field.",
                "Zero HP is not treated as dormant; a corpse may remain instantiated until the slot is released.",
                "Validation requires strong live matching and repeated far/dormant clears across many actor bases.",
                "Same-slot reappearance is diagnostic only because actor bases are reusable pool identities.",
                "A validated candidate may be promoted only after independent current-process revalidation by the native reader.",
            ],
        }
