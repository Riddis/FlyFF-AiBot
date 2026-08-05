from __future__ import annotations

import math
import struct
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from typing import Protocol

from .NativeTraceTargets import TraceTargetDiscovery
from .Win32ProcessMemory import MemorySearchProgress


class AuthoritativeActorMemory(Protocol):
    def read(self, address: int, size: int) -> bytes: ...

    def find_u32(
        self,
        value: int,
        *,
        maximum_address: int,
        private_only: bool,
        chunk_size: int,
        cancellation: object | None = None,
        deadline: float | None = None,
        progress_callback: Callable[[MemorySearchProgress], None] | None = None,
    ) -> tuple[int, ...]: ...


def _cancellation_requested(cancellation: object | None) -> bool:
    if cancellation is None:
        return False
    cancelled = getattr(cancellation, "cancelled", None)
    if cancelled is not None:
        return bool(cancelled() if callable(cancelled) else cancelled)
    is_set = getattr(cancellation, "is_set", None)
    return bool(is_set()) if callable(is_set) else False


def _raise_if_stopped(
    cancellation: object | None,
    deadline: float | None,
) -> None:
    if _cancellation_requested(cancellation):
        from .Win32ProcessMemory import MemorySearchCancelled

        raise MemorySearchCancelled("Authoritative actor discovery was cancelled")
    if deadline is not None:
        from time import monotonic

        if monotonic() >= float(deadline):
            from .Win32ProcessMemory import MemorySearchDeadline

            raise MemorySearchDeadline(
                "Authoritative actor discovery exceeded its deadline"
            )


@dataclass(frozen=True, slots=True)
class SharedRelationCandidate:
    offset: int
    value: int
    monster_support: int
    monster_samples: int
    player_match: bool
    target_readable: bool
    target_actor_references: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RelationScanEvidence:
    offset: int
    value: int
    references: int
    unique_candidate_bases: int
    valid_actor_bases: int
    exact_anchor_coverage: int
    exact_anchor_total: int
    selected_species_counts: tuple[tuple[int, int], ...]
    self_rejections: int
    relation_rejections: int
    species_rejections: int
    hp_rejections: int
    coordinate_rejections: int
    unreadable_rejections: int
    search_bytes_read: int
    search_regions_read: int
    structural_actor_bases: int = 0

    @property
    def anchor_coverage_ratio(self) -> float:
        if self.exact_anchor_total <= 0:
            return 0.0
        return self.exact_anchor_coverage / self.exact_anchor_total

    @property
    def selected_species_diversity(self) -> int:
        return len(self.selected_species_counts)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["anchor_coverage_ratio"] = self.anchor_coverage_ratio
        payload["selected_species_diversity"] = self.selected_species_diversity
        return payload


@dataclass(frozen=True, slots=True)
class ActiveFieldCandidate:
    offset: int
    living_matches: int
    living_samples: int
    zero_hp_matches: int
    zero_hp_samples: int
    validated: bool

    @property
    def living_match_ratio(self) -> float:
        if self.living_samples <= 0:
            return 0.0
        return self.living_matches / self.living_samples

    @property
    def zero_hp_match_ratio(self) -> float:
        if self.zero_hp_samples <= 0:
            return 0.0
        return self.zero_hp_matches / self.zero_hp_samples

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["living_match_ratio"] = self.living_match_ratio
        payload["zero_hp_match_ratio"] = self.zero_hp_match_ratio
        return payload


HISTORICAL_PRESENCE_OFFSETS = (0x1DCC,)


@dataclass(frozen=True, slots=True)
class PresenceFieldCandidate:
    """Evidence for an instantiated-species field inside one actor slot."""

    offset: int
    selected_matches: int
    selected_samples: int
    zero_hp_matches: int
    zero_hp_samples: int
    dormant_clears: int
    dormant_samples: int
    cross_slot_alias_matches: int
    lifecycle_death_retained: int = 0
    lifecycle_dormant_clears: int = 0
    lifecycle_reappearances: int = 0
    validated: bool = False

    @property
    def selected_match_ratio(self) -> float:
        return self.selected_matches / self.selected_samples if self.selected_samples else 0.0

    @property
    def zero_hp_match_ratio(self) -> float:
        return self.zero_hp_matches / self.zero_hp_samples if self.zero_hp_samples else 0.0

    @property
    def dormant_clear_ratio(self) -> float:
        return self.dormant_clears / self.dormant_samples if self.dormant_samples else 0.0

    @property
    def provisional(self) -> bool:
        return not self.validated

    @property
    def evidence_score(self) -> float:
        zero_ratio = self.zero_hp_match_ratio if self.zero_hp_samples else 0.5
        dormant_ratio = self.dormant_clear_ratio if self.dormant_samples else 0.0
        return (
            4.0 * self.selected_match_ratio
            + 2.0 * zero_ratio
            + 3.0 * dormant_ratio
            + min(self.selected_samples, 32) / 64.0
            - 4.0 * self.cross_slot_alias_matches
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            selected_match_ratio=self.selected_match_ratio,
            zero_hp_match_ratio=self.zero_hp_match_ratio,
            dormant_clear_ratio=self.dormant_clear_ratio,
            provisional=self.provisional,
            evidence_score=self.evidence_score,
        )
        return payload


@dataclass(frozen=True, slots=True)
class AuthoritativeActorDiscovery:
    outcome: str
    message: str
    relation_offset: int | None
    relation_value: int | None
    actor_bases: tuple[int, ...]
    species_counts: tuple[tuple[int, int], ...]
    active_species_offset: int | None
    active_species_validated: bool
    relation_candidates: tuple[SharedRelationCandidate, ...]
    relation_scans: tuple[RelationScanEvidence, ...]
    active_candidates: tuple[ActiveFieldCandidate, ...]
    presence_species_offset: int | None = None
    presence_species_validated: bool = False
    presence_candidates: tuple[PresenceFieldCandidate, ...] = ()

    @property
    def succeeded(self) -> bool:
        return (
            self.outcome == "success"
            and self.relation_offset is not None
            and self.relation_value is not None
            and bool(self.actor_bases)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "outcome": self.outcome,
            "message": self.message,
            "relation_offset": self.relation_offset,
            "relation_value": self.relation_value,
            "actor_bases": list(self.actor_bases),
            "actor_count": len(self.actor_bases),
            "species_counts": [list(item) for item in self.species_counts],
            "active_species_offset": self.active_species_offset,
            "active_species_validated": self.active_species_validated,
            "relation_candidates": [item.to_dict() for item in self.relation_candidates],
            "relation_scans": [item.to_dict() for item in self.relation_scans],
            "active_candidates": [item.to_dict() for item in self.active_candidates],
            "presence_species_offset": self.presence_species_offset,
            "presence_species_validated": self.presence_species_validated,
            "presence_candidates": [item.to_dict() for item in self.presence_candidates],
        }


@dataclass(frozen=True, slots=True)
class AuthoritativeActorRefresh:
    actor_bases: tuple[int, ...]
    species_counts: tuple[tuple[int, int], ...]
    evidence: RelationScanEvidence
    active_species_offset: int | None
    active_species_validated: bool
    active_candidates: tuple[ActiveFieldCandidate, ...]
    presence_species_offset: int | None = None
    presence_species_validated: bool = False
    presence_candidates: tuple[PresenceFieldCandidate, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "actor_bases": list(self.actor_bases),
            "actor_count": len(self.actor_bases),
            "species_counts": [list(item) for item in self.species_counts],
            "evidence": self.evidence.to_dict(),
            "active_species_offset": self.active_species_offset,
            "active_species_validated": self.active_species_validated,
            "active_candidates": [item.to_dict() for item in self.active_candidates],
            "presence_species_offset": self.presence_species_offset,
            "presence_species_validated": self.presence_species_validated,
            "presence_candidates": [item.to_dict() for item in self.presence_candidates],
        }


def _u32(data: bytes, offset: int = 0) -> int:
    return int(struct.unpack_from("<I", data, offset)[0])


def _i32(data: bytes, offset: int = 0) -> int:
    return int(struct.unpack_from("<i", data, offset)[0])


def _f32(data: bytes, offset: int = 0) -> float:
    return float(struct.unpack_from("<f", data, offset)[0])


def _even_sample(values: tuple[object, ...], maximum: int) -> tuple[object, ...]:
    if len(values) <= maximum:
        return values
    if maximum <= 1:
        return (values[0],)
    last = len(values) - 1
    indexes = {round(index * last / (maximum - 1)) for index in range(maximum)}
    return tuple(values[index] for index in sorted(indexes))


def _read_words(
    memory: AuthoritativeActorMemory,
    base: int,
    span: int,
) -> dict[int, int]:
    span = max(4, int(span) - (int(span) % 4))
    try:
        data = memory.read(int(base), span)
        if len(data) == span:
            return {
                offset: _u32(data, offset)
                for offset in range(0, span, 4)
            }
    except Exception:
        pass

    result: dict[int, int] = {}
    for offset in range(0, span, 4):
        try:
            result[offset] = _u32(memory.read(int(base) + offset, 4))
        except Exception:
            continue
    return result


def _pointer_like(value: int, maximum_address: int) -> bool:
    return 0x10000 < int(value) <= int(maximum_address) and int(value) % 4 == 0


def _target_readable(memory: AuthoritativeActorMemory, value: int) -> bool:
    try:
        return len(memory.read(int(value), 4)) == 4
    except Exception:
        return False


def _target_actor_reference_count(
    memory: AuthoritativeActorMemory,
    value: int,
    actor_bases: set[int],
    *,
    span: int = 0x1000,
) -> int:
    try:
        data = memory.read(int(value), int(span))
    except Exception:
        return 0
    count = 0
    for offset in range(0, len(data) - 3, 4):
        if _u32(data, offset) in actor_bases:
            count += 1
    return count


def _shared_relation_candidates(
    memory: AuthoritativeActorMemory,
    discovery: TraceTargetDiscovery,
    *,
    actor_stride: int | None,
    object_span: int,
    maximum_address: int,
    maximum_samples: int = 48,
    cancellation: object | None = None,
    deadline: float | None = None,
) -> tuple[SharedRelationCandidate, ...]:
    if discovery.player is None or not discovery.monsters:
        return ()

    ordered_monsters = tuple(sorted(discovery.monsters, key=lambda item: item.base))
    sampled = _even_sample(ordered_monsters, max(3, int(maximum_samples)))
    scan_span = min(
        max(4, int(object_span)),
        int(actor_stride) if actor_stride is not None and actor_stride > 0 else int(object_span),
    )
    scan_span -= scan_span % 4
    monster_words: list[dict[int, int]] = []
    for item in sampled:
        _raise_if_stopped(cancellation, deadline)
        monster_words.append(_read_words(memory, int(item.base), scan_span))
    _raise_if_stopped(cancellation, deadline)
    player_words = _read_words(memory, int(discovery.player.base), scan_span)

    first = discovery.monsters[0]
    excluded = {
        int(first.species_offset),
        int(first.hp_offset),
        int(first.x_offset),
        int(first.y_offset),
        int(first.z_offset),
        *(int(value) for value in first.self_pointer_offsets),
    }
    known_bases = {int(item.base) for item in sampled}
    known_bases.add(int(discovery.player.base))
    minimum_support = max(3, math.ceil(len(sampled) * 0.80))
    candidates: list[SharedRelationCandidate] = []

    for offset in range(0, scan_span, 4):
        if offset % 0x100 == 0:
            _raise_if_stopped(cancellation, deadline)
        if offset in excluded:
            continue
        values = [words[offset] for words in monster_words if offset in words]
        if len(values) < minimum_support:
            continue
        value, support = Counter(values).most_common(1)[0]
        if support < minimum_support or not _pointer_like(value, maximum_address):
            continue
        player_match = player_words.get(offset) == value
        if not player_match:
            continue
        readable = _target_readable(memory, value)
        if not readable:
            continue
        candidates.append(
            SharedRelationCandidate(
                offset=offset,
                value=value,
                monster_support=support,
                monster_samples=len(sampled),
                player_match=True,
                target_readable=True,
                target_actor_references=_target_actor_reference_count(
                    memory,
                    value,
                    known_bases,
                ),
            )
        )

    species_offset = int(first.species_offset)
    candidates.sort(
        key=lambda item: (
            -item.monster_support,
            -item.target_actor_references,
            abs(item.offset - species_offset),
            item.offset,
        )
    )
    return tuple(candidates)


def _has_self_alias(
    memory: AuthoritativeActorMemory,
    base: int,
    self_offsets: tuple[int, ...],
) -> bool:
    for offset in self_offsets:
        try:
            if _u32(memory.read(int(base) + int(offset), 4)) == int(base):
                return True
        except Exception:
            continue
    return False


def _read_actor_core(
    memory: AuthoritativeActorMemory,
    base: int,
    *,
    species_offset: int,
    hp_offset: int,
    x_offset: int,
    y_offset: int,
    z_offset: int,
) -> tuple[int, int, float, float, float]:
    offsets = (species_offset, hp_offset, x_offset, y_offset, z_offset)
    first = min(offsets)
    last = max(offsets) + 4
    try:
        data = memory.read(int(base) + first, last - first)
        if len(data) != last - first:
            raise OSError("short actor core read")
        return (
            _i32(data, species_offset - first),
            _i32(data, hp_offset - first),
            _f32(data, x_offset - first),
            _f32(data, y_offset - first),
            _f32(data, z_offset - first),
        )
    except Exception:
        return (
            _i32(memory.read(int(base) + species_offset, 4)),
            _i32(memory.read(int(base) + hp_offset, 4)),
            _f32(memory.read(int(base) + x_offset, 4)),
            _f32(memory.read(int(base) + y_offset, 4)),
            _f32(memory.read(int(base) + z_offset, 4)),
        )


def _scan_relation(
    memory: AuthoritativeActorMemory,
    discovery: TraceTargetDiscovery,
    candidate: SharedRelationCandidate,
    *,
    selected_species_ids: set[int],
    maximum_address: int,
    private_memory_only: bool,
    chunk_size: int,
    coordinate_limit: float,
    cancellation: object | None,
    deadline: float | None,
    progress_callback: Callable[[MemorySearchProgress], None] | None = None,
) -> tuple[
    tuple[int, ...],
    tuple[tuple[int, int], ...],
    RelationScanEvidence,
    tuple[int, ...],
]:
    first = discovery.monsters[0]
    self_offsets = tuple(
        dict.fromkeys(
            int(offset)
            for monster in discovery.monsters
            for offset in monster.self_pointer_offsets
        )
    )
    search_kwargs = {
        "maximum_address": int(maximum_address),
        "private_only": bool(private_memory_only),
        "chunk_size": int(chunk_size),
        "cancellation": cancellation,
        "deadline": deadline,
    }
    if progress_callback is None:
        references = memory.find_u32(int(candidate.value), **search_kwargs)
    else:
        try:
            references = memory.find_u32(
                int(candidate.value),
                **search_kwargs,
                progress_callback=progress_callback,
            )
        except TypeError as error:
            if "progress_callback" not in str(error):
                raise
            # Lightweight test doubles and older adapters can omit the optional
            # status callback while preserving the scan's functional contract.
            references = memory.find_u32(int(candidate.value), **search_kwargs)
    diagnostics = getattr(memory, "last_search_diagnostics", None)
    search_bytes = int(getattr(diagnostics, "bytes_read", 0))
    search_regions = int(getattr(diagnostics, "regions_read", 0))

    candidate_bases = {
        int(address) - int(candidate.offset)
        for address in references
        if int(address) >= int(candidate.offset)
    }
    valid: set[int] = set()
    structural: set[int] = set()
    species_counts: Counter[int] = Counter()
    self_rejections = 0
    relation_rejections = 0
    species_rejections = 0
    hp_rejections = 0
    coordinate_rejections = 0
    unreadable_rejections = 0

    for index, base in enumerate(sorted(candidate_bases)):
        if index % 128 == 0:
            _raise_if_stopped(cancellation, deadline)
        if base <= 0x10000 or base > int(maximum_address):
            unreadable_rejections += 1
            continue
        if not _has_self_alias(memory, base, self_offsets):
            self_rejections += 1
            continue
        try:
            if _u32(memory.read(base + candidate.offset, 4)) != candidate.value:
                relation_rejections += 1
                continue
            structural.add(base)
            species, hp, x, y, z = _read_actor_core(
                memory,
                base,
                species_offset=int(first.species_offset),
                hp_offset=int(first.hp_offset),
                x_offset=int(first.x_offset),
                y_offset=int(first.y_offset),
                z_offset=int(first.z_offset),
            )
        except Exception:
            unreadable_rejections += 1
            continue
        if species <= 0 or (
            selected_species_ids and species not in selected_species_ids
        ):
            species_rejections += 1
            continue
        if hp < 0:
            hp_rejections += 1
            continue
        if not all(math.isfinite(value) for value in (x, y, z)) or any(
            abs(value) > float(coordinate_limit) for value in (x, y, z)
        ):
            coordinate_rejections += 1
            continue
        valid.add(base)
        species_counts[species] += 1

    exact = {int(item.base) for item in discovery.monsters}
    evidence = RelationScanEvidence(
        offset=int(candidate.offset),
        value=int(candidate.value),
        references=len(references),
        unique_candidate_bases=len(candidate_bases),
        valid_actor_bases=len(valid),
        exact_anchor_coverage=len(valid & exact),
        exact_anchor_total=len(exact),
        selected_species_counts=tuple(sorted(species_counts.items())),
        self_rejections=self_rejections,
        relation_rejections=relation_rejections,
        species_rejections=species_rejections,
        hp_rejections=hp_rejections,
        coordinate_rejections=coordinate_rejections,
        unreadable_rejections=unreadable_rejections,
        search_bytes_read=search_bytes,
        search_regions_read=search_regions,
        structural_actor_bases=len(structural),
    )
    return (
        tuple(sorted(valid)),
        tuple(sorted(species_counts.items())),
        evidence,
        tuple(sorted(structural)),
    )


def _active_field_candidates(
    memory: AuthoritativeActorMemory,
    discovery: TraceTargetDiscovery,
    actor_bases: tuple[int, ...],
    *,
    actor_stride: int | None,
    relation_offset: int,
    maximum_samples: int = 128,
    cancellation: object | None = None,
    deadline: float | None = None,
) -> tuple[ActiveFieldCandidate, ...]:
    if not actor_bases:
        return ()
    first = discovery.monsters[0]
    span = int(actor_stride) if actor_stride is not None and actor_stride > 0 else 0x2008
    span = min(max(4, span), 0x4000)
    span -= span % 4
    excluded = {
        int(first.species_offset),
        int(first.hp_offset),
        int(first.x_offset),
        int(first.y_offset),
        int(first.z_offset),
        int(relation_offset),
        *(int(value) for value in first.self_pointer_offsets),
    }

    sampled_bases = _even_sample(tuple(actor_bases), max(8, int(maximum_samples)))
    samples: list[tuple[int, int, dict[int, int]]] = []
    for raw_base in sampled_bases:
        _raise_if_stopped(cancellation, deadline)
        base = int(raw_base)
        try:
            species, hp, _x, _y, _z = _read_actor_core(
                memory,
                base,
                species_offset=int(first.species_offset),
                hp_offset=int(first.hp_offset),
                x_offset=int(first.x_offset),
                y_offset=int(first.y_offset),
                z_offset=int(first.z_offset),
            )
        except Exception:
            continue
        if species <= 0 or hp < 0:
            continue
        samples.append((species, hp, _read_words(memory, base, span)))
    if not samples:
        return ()

    candidates: list[ActiveFieldCandidate] = []
    for offset in range(0, span, 4):
        if offset % 0x100 == 0:
            _raise_if_stopped(cancellation, deadline)
        if offset in excluded:
            continue
        living_matches = living_samples = zero_matches = zero_samples = 0
        for species, hp, words in samples:
            if offset not in words:
                continue
            if hp == 0:
                zero_samples += 1
                zero_matches += int(words[offset] == species)
            else:
                living_samples += 1
                living_matches += int(words[offset] == species)
        if living_samples < 3 or living_matches * 5 < living_samples * 4:
            continue
        validated = (
            zero_samples >= 2
            and zero_matches * 5 <= zero_samples
        )
        candidates.append(
            ActiveFieldCandidate(
                offset=offset,
                living_matches=living_matches,
                living_samples=living_samples,
                zero_hp_matches=zero_matches,
                zero_hp_samples=zero_samples,
                validated=validated,
            )
        )

    candidates.sort(
        key=lambda item: (
            0 if item.validated else 1,
            -item.living_match_ratio,
            item.zero_hp_match_ratio,
            -item.living_samples,
            item.offset,
        )
    )
    return tuple(candidates[:32])


def _presence_field_candidates(
    memory: AuthoritativeActorMemory,
    discovery: TraceTargetDiscovery,
    actor_bases: tuple[int, ...],
    structural_bases: tuple[int, ...],
    *,
    selected_species_ids: set[int],
    actor_stride: int | None,
    relation_offset: int,
    preferred_offsets: Iterable[int] = HISTORICAL_PRESENCE_OFFSETS,
    maximum_samples: int = 160,
    cancellation: object | None = None,
    deadline: float | None = None,
) -> tuple[PresenceFieldCandidate, ...]:
    """Rank presence fields using the recovered actor cohort and dormant slots."""

    if not actor_bases or actor_stride is None or int(actor_stride) <= 0:
        return ()
    stride = int(actor_stride)
    span = stride - stride % 4
    first = discovery.monsters[0]
    excluded = {
        int(first.species_offset),
        int(first.hp_offset),
        int(first.x_offset),
        int(first.y_offset),
        int(first.z_offset),
        int(relation_offset),
        *(int(value) for value in first.self_pointer_offsets),
    }
    selected = {int(value) for value in selected_species_ids if int(value) > 0}
    selected.update(int(item.species) for item in discovery.monsters)
    sampled = _even_sample(
        tuple(dict.fromkeys((*actor_bases, *structural_bases))),
        max(16, int(maximum_samples)),
    )
    samples: list[tuple[int, int, int, dict[int, int]]] = []
    species_by_base: dict[int, int] = {}
    for raw_base in sampled:
        _raise_if_stopped(cancellation, deadline)
        base = int(raw_base)
        try:
            species, hp, _x, _y, _z = _read_actor_core(
                memory,
                base,
                species_offset=int(first.species_offset),
                hp_offset=int(first.hp_offset),
                x_offset=int(first.x_offset),
                y_offset=int(first.y_offset),
                z_offset=int(first.z_offset),
            )
        except Exception:
            continue
        species_by_base[base] = species
        if species not in selected and species > 0:
            continue
        samples.append((base, species, hp, _read_words(memory, base, span)))
    if not samples:
        return ()

    preferred = {
        int(offset): index
        for index, offset in enumerate(
            dict.fromkeys(int(value) for value in preferred_offsets if int(value) >= 0)
        )
    }
    candidates: list[PresenceFieldCandidate] = []
    for offset in range(0, span, 4):
        if offset % 0x100 == 0:
            _raise_if_stopped(cancellation, deadline)
        if offset in excluded:
            continue
        selected_matches = selected_samples = 0
        zero_matches = zero_samples = 0
        dormant_clears = dormant_samples = 0
        cross_aliases = 0
        selected_conflicts = 0
        for base, species, hp, words in samples:
            if offset not in words:
                continue
            value = int(words[offset])
            if species in selected and hp >= 0:
                selected_samples += 1
                selected_matches += int(value == species)
                selected_conflicts += int(value in selected and value != species)
                if hp == 0:
                    zero_samples += 1
                    zero_matches += int(value == species)
                neighbour_species = {
                    species_by_base.get(base - stride, 0),
                    species_by_base.get(base + stride, 0),
                }
                if value != species and value > 0 and value in neighbour_species:
                    cross_aliases += 1
            elif species <= 0:
                dormant_samples += 1
                dormant_clears += int(value == 0)
        if selected_samples < 3 or selected_matches * 5 < selected_samples * 4:
            continue
        if selected_conflicts or cross_aliases:
            continue
        if zero_samples and zero_matches * 5 < zero_samples * 4:
            continue
        candidates.append(
            PresenceFieldCandidate(
                offset=offset,
                selected_matches=selected_matches,
                selected_samples=selected_samples,
                zero_hp_matches=zero_matches,
                zero_hp_samples=zero_samples,
                dormant_clears=dormant_clears,
                dormant_samples=dormant_samples,
                cross_slot_alias_matches=cross_aliases,
            )
        )

    candidates.sort(
        key=lambda item: (
            -item.evidence_score,
            0 if item.offset in preferred else 1,
            preferred.get(item.offset, 1_000_000),
            item.offset,
        )
    )
    if candidates:
        best = candidates[0]
        statically_proven = (
            best.dormant_samples >= 2
            and best.dormant_clears * 5 >= best.dormant_samples * 4
            and (best.zero_hp_samples == 0 or best.zero_hp_matches * 5 >= best.zero_hp_samples * 4)
        )
        runner_up = candidates[1] if len(candidates) > 1 else None
        unique = runner_up is None or best.evidence_score > runner_up.evidence_score + 0.05
        if statically_proven and unique:
            candidates[0] = replace(best, validated=True)
    return tuple(candidates[:32])



def _strong_anchor_relation_proven(
    evidence: RelationScanEvidence,
    *,
    exact_total: int,
    anchor_species: set[int],
) -> bool:
    """Return whether one relation is strong enough to stop initial scanning.

    Initial pointer recovery can only see the species that are loaded around the
    player.  Tower starts in the Asterius section, so Dantalian may legitimately
    be absent even when it is selected for later recording.  Requiring every
    selected species here forces scans of unrelated high-cardinality pointer
    values and can make attachment appear to hang.

    A relation is proven by near-complete recovery of the exact validated anchor
    cohort plus material expansion beyond that cohort.  Additional selected
    species are discovered later through the same relation during continuous
    authoritative refreshes.
    """

    recovered_species = {
        int(species)
        for species, count in evidence.selected_species_counts
        if int(count) > 0
    }
    minimum_expansion = max(4, math.ceil(max(1, int(exact_total)) * 0.20))
    return (
        evidence.anchor_coverage_ratio >= 0.90
        and evidence.valid_actor_bases
        >= evidence.exact_anchor_coverage + minimum_expansion
        and bool(anchor_species)
        and anchor_species.issubset(recovered_species)
    )

def discover_authoritative_actors(
    memory: AuthoritativeActorMemory,
    discovery: TraceTargetDiscovery,
    *,
    selected_species_ids: Iterable[int],
    actor_stride: int | None,
    object_span: int,
    maximum_address: int,
    private_memory_only: bool,
    chunk_size: int,
    coordinate_limit: float,
    cancellation: object | None = None,
    deadline: float | None = None,
    maximum_relation_candidates: int = 4,
    preferred_relation_offsets: Iterable[int] = (),
    preferred_presence_offsets: Iterable[int] = HISTORICAL_PRESENCE_OFFSETS,
    status_callback: Callable[[str], None] | None = None,
) -> AuthoritativeActorDiscovery:
    """Recover the old global actor relationship without using stale offsets.

    The validated player and exact full-health monster anchors are treated as
    ground truth. A shared pointer-valued field present at the same offset in
    both classes is inferred, then its value is searched process-wide. Candidate
    objects must pass the already recovered self/species/HP/coordinate layout.
    """

    selected = {int(value) for value in selected_species_ids if int(value) > 0}
    selected.update(int(item.species) for item in discovery.monsters)
    candidates = _shared_relation_candidates(
        memory,
        discovery,
        actor_stride=actor_stride,
        object_span=object_span,
        maximum_address=maximum_address,
        cancellation=cancellation,
        deadline=deadline,
    )
    if not candidates:
        return AuthoritativeActorDiscovery(
            outcome="shared_relation_not_found",
            message=(
                "No pointer-valued field was shared at one offset by the validated "
                "player and monster anchors."
            ),
            relation_offset=None,
            relation_value=None,
            actor_bases=(),
            species_counts=(),
            active_species_offset=None,
            active_species_validated=False,
            relation_candidates=(),
            relation_scans=(),
            active_candidates=(),
        )

    preferred = tuple(
        dict.fromkeys(
            int(value)
            for value in preferred_relation_offsets
            if int(value) >= 0
        )
    )
    if preferred:
        preference_rank = {offset: index for index, offset in enumerate(preferred)}
        candidates = tuple(
            sorted(
                candidates,
                key=lambda item: (
                    0 if item.offset in preference_rank else 1,
                    preference_rank.get(item.offset, 1_000_000),
                    -item.monster_support,
                    -item.target_actor_references,
                    item.offset,
                ),
            )
        )

    scans: list[RelationScanEvidence] = []
    results: list[
        tuple[
            SharedRelationCandidate,
            tuple[int, ...],
            tuple[tuple[int, int], ...],
            RelationScanEvidence,
            tuple[int, ...],
        ]
    ] = []
    exact_total = max(1, len(discovery.monsters))
    minimum_coverage = max(2, math.ceil(exact_total * 0.60))

    selected_candidates = candidates[: max(1, int(maximum_relation_candidates))]
    for candidate_index, candidate in enumerate(selected_candidates, start=1):
        _raise_if_stopped(cancellation, deadline)
        if status_callback is not None:
            status_callback(
                "Authoritative actor scan candidate "
                f"{candidate_index}/{len(selected_candidates)} at "
                f"+0x{candidate.offset:X}: scanning readable private memory."
            )

        def scan_progress(
            progress: MemorySearchProgress,
            *,
            current_index: int = candidate_index,
            candidate_count: int = len(selected_candidates),
        ) -> None:
            if status_callback is None:
                return
            suffix = " complete" if progress.complete else ""
            status_callback(
                "Authoritative actor scan "
                f"{current_index}/{candidate_count}{suffix}: "
                f"{progress.bytes_read / (1024 * 1024):.0f} MiB, "
                f"regions={progress.regions_read}, "
                f"matches={progress.matches}, "
                f"elapsed={progress.elapsed_seconds:.1f}s."
            )

        bases, species_counts, evidence, structural_bases = _scan_relation(
            memory,
            discovery,
            candidate,
            selected_species_ids=selected,
            maximum_address=maximum_address,
            private_memory_only=private_memory_only,
            chunk_size=chunk_size,
            coordinate_limit=coordinate_limit,
            cancellation=cancellation,
            deadline=deadline,
            progress_callback=scan_progress,
        )
        scans.append(evidence)
        if status_callback is not None:
            status_callback(
                "Authoritative actor candidate "
                f"+0x{candidate.offset:X} validated "
                f"{evidence.valid_actor_bases} actors, "
                f"exact anchors={evidence.exact_anchor_coverage}/"
                f"{evidence.exact_anchor_total}, "
                f"species={dict(evidence.selected_species_counts)}."
            )
        if evidence.exact_anchor_coverage < minimum_coverage:
            continue
        results.append((candidate, bases, species_counts, evidence, structural_bases))
        # Initial Tower recovery normally sees Asterius only.  Once one
        # relation recovers the exact validated anchor cohort and materially
        # expands beyond it, accept it immediately.  Dantalian and any other
        # selected species are allowed to load later and are merged by the
        # continuous authoritative refresh path.
        anchor_species = {int(item.species) for item in discovery.monsters}
        if _strong_anchor_relation_proven(
            evidence,
            exact_total=exact_total,
            anchor_species=anchor_species,
        ):
            if status_callback is not None and not selected.issubset(
                {
                    int(species)
                    for species, count in evidence.selected_species_counts
                    if int(count) > 0
                }
            ):
                status_callback(
                    "The authoritative relation is proven from the currently "
                    "loaded anchor species. Additional selected species will be "
                    "added automatically as they load during farming."
                )
            break

    if not results:
        return AuthoritativeActorDiscovery(
            outcome="shared_relation_unvalidated",
            message=(
                "Shared pointer candidates existed, but none recovered enough of "
                "the exact validated monster anchors through one global relation."
            ),
            relation_offset=None,
            relation_value=None,
            actor_bases=(),
            species_counts=(),
            active_species_offset=None,
            active_species_validated=False,
            relation_candidates=candidates,
            relation_scans=tuple(scans),
            active_candidates=(),
        )

    (
        selected_candidate,
        actor_bases,
        species_counts,
        _selected_scan,
        structural_bases,
    ) = max(
        results,
        key=lambda item: (
            item[3].anchor_coverage_ratio,
            item[3].selected_species_diversity,
            item[3].valid_actor_bases,
            item[0].target_actor_references,
            -abs(item[0].offset - discovery.monsters[0].species_offset),
        ),
    )
    active_candidates = _active_field_candidates(
        memory,
        discovery,
        actor_bases,
        actor_stride=actor_stride,
        relation_offset=selected_candidate.offset,
        cancellation=cancellation,
        deadline=deadline,
    )
    active = active_candidates[0] if active_candidates else None
    active_offset = None if active is None else int(active.offset)
    active_validated = bool(active is not None and active.validated)
    presence_candidates = _presence_field_candidates(
        memory,
        discovery,
        actor_bases,
        structural_bases,
        selected_species_ids=selected,
        actor_stride=actor_stride,
        relation_offset=selected_candidate.offset,
        preferred_offsets=preferred_presence_offsets,
        cancellation=cancellation,
        deadline=deadline,
    )
    presence = presence_candidates[0] if presence_candidates else None
    return AuthoritativeActorDiscovery(
        outcome="success",
        message=(
            "Dynamically recovered the authoritative global actor relation at "
            f"+0x{selected_candidate.offset:X}; {len(actor_bases)} selected actors "
            f"were validated across species {dict(species_counts)}."
        ),
        relation_offset=int(selected_candidate.offset),
        relation_value=int(selected_candidate.value),
        actor_bases=actor_bases,
        species_counts=species_counts,
        active_species_offset=active_offset,
        active_species_validated=active_validated,
        relation_candidates=candidates,
        relation_scans=tuple(scans),
        active_candidates=active_candidates,
        presence_species_offset=None if presence is None else int(presence.offset),
        presence_species_validated=bool(presence is not None and presence.validated),
        presence_candidates=presence_candidates,
    )


def refresh_authoritative_actors(
    memory: AuthoritativeActorMemory,
    discovery: TraceTargetDiscovery,
    *,
    relation_offset: int,
    relation_value: int,
    selected_species_ids: Iterable[int],
    actor_stride: int | None,
    maximum_address: int,
    private_memory_only: bool,
    chunk_size: int,
    coordinate_limit: float,
    cancellation: object | None = None,
    deadline: float | None = None,
    status_callback: Callable[[str], None] | None = None,
    preferred_presence_offsets: Iterable[int] = HISTORICAL_PRESENCE_OFFSETS,
) -> AuthoritativeActorRefresh:
    selected = {int(value) for value in selected_species_ids if int(value) > 0}
    selected.update(int(item.species) for item in discovery.monsters)
    candidate = SharedRelationCandidate(
        offset=int(relation_offset),
        value=int(relation_value),
        monster_support=len(discovery.monsters),
        monster_samples=len(discovery.monsters),
        player_match=True,
        target_readable=True,
        target_actor_references=0,
    )
    if status_callback is not None:
        status_callback(
            "Refreshing actors from the saved authoritative relation "
            f"+0x{int(relation_offset):X}."
        )

    def scan_progress(progress: MemorySearchProgress) -> None:
        if status_callback is None:
            return
        suffix = " complete" if progress.complete else ""
        status_callback(
            "Saved authoritative relation scan"
            f"{suffix}: {progress.bytes_read / (1024 * 1024):.0f} MiB, "
            f"regions={progress.regions_read}, matches={progress.matches}, "
            f"elapsed={progress.elapsed_seconds:.1f}s."
        )

    actor_bases, species_counts, evidence, structural_bases = _scan_relation(
        memory,
        discovery,
        candidate,
        selected_species_ids=selected,
        maximum_address=maximum_address,
        private_memory_only=private_memory_only,
        chunk_size=chunk_size,
        coordinate_limit=coordinate_limit,
        cancellation=cancellation,
        deadline=deadline,
        progress_callback=scan_progress,
    )
    active_candidates = _active_field_candidates(
        memory,
        discovery,
        actor_bases,
        actor_stride=actor_stride,
        relation_offset=int(relation_offset),
        cancellation=cancellation,
        deadline=deadline,
    )
    active = active_candidates[0] if active_candidates else None
    presence_candidates = _presence_field_candidates(
        memory,
        discovery,
        actor_bases,
        structural_bases,
        selected_species_ids=selected,
        actor_stride=actor_stride,
        relation_offset=int(relation_offset),
        preferred_offsets=preferred_presence_offsets,
        cancellation=cancellation,
        deadline=deadline,
    )
    presence = presence_candidates[0] if presence_candidates else None
    return AuthoritativeActorRefresh(
        actor_bases=actor_bases,
        species_counts=species_counts,
        evidence=evidence,
        active_species_offset=None if active is None else int(active.offset),
        active_species_validated=bool(active is not None and active.validated),
        active_candidates=active_candidates,
        presence_species_offset=None if presence is None else int(presence.offset),
        presence_species_validated=bool(presence is not None and presence.validated),
        presence_candidates=presence_candidates,
    )
