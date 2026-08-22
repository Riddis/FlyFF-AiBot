from __future__ import annotations

import json
import math
import re
import struct
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep
from typing import Protocol

from .AnchoredPointerDiscovery import AnchoredMonsterObservation
from .PointerScanWorkflow import (
    PointerPath,
    ReadableRegionIndex,
    resolve_pointer_path,
    scan_module_rooted_paths,
)
from .Win32ProcessMemory import ModuleInfo


class AggregateScanMemory(Protocol):
    pid: int

    def read(self, address: int, size: int) -> bytes: ...


@dataclass(frozen=True, slots=True)
class CohortActorState:
    base: int
    species: int
    active_species: int
    hp: int
    x: float
    y: float
    z: float
    self_valid: bool
    readable: bool
    active: bool
    field_values: tuple[tuple[int, int], ...] = ()

    def field_value(self, offset: int) -> int | None:
        for candidate_offset, value in self.field_values:
            if candidate_offset == offset:
                return value
        return None


@dataclass(frozen=True, slots=True)
class AggregateCandidate:
    kind: str
    target_base: int
    actor_field_offset: int | None
    baseline_support: int
    sample_support: tuple[int, ...]
    sample_active_counts: tuple[int, ...]
    average_active_coverage: float
    minimum_active_coverage: float
    changed_actor_support: int
    reference_churn: int
    direct_module_offsets: tuple[int, ...]
    pointer_paths: tuple[PointerPath, ...]
    recommended: bool

    @property
    def shortest_path_depth(self) -> int | None:
        if not self.pointer_paths:
            return None
        return min(path.depth for path in self.pointer_paths)


@dataclass(frozen=True, slots=True)
class AggregateCohortReport:
    schema_version: int
    captured_at_utc: str
    pid: int
    module_name: str
    module_path: str
    module_base: int
    module_size: int
    cohort_size: int
    sample_count: int
    duration_seconds: float
    changed_actor_count: int
    transition_events: int
    reference_scan_bytes: int
    reference_scan_regions: int
    reference_scan_failures: int
    reference_matches: int
    candidates: tuple[AggregateCandidate, ...]

    def to_json(self) -> str:
        payload = asdict(self)
        payload["candidates"] = []
        for item in self.candidates:
            candidate = asdict(item)
            candidate["pointer_paths"] = [
                {
                    "root_module_offset": path.root_module_offset,
                    "field_offsets": list(path.field_offsets),
                    "signature": path.signature,
                    "depth": path.depth,
                }
                for path in item.pointer_paths
            ]
            payload["candidates"].append(candidate)
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True, slots=True)
class _CandidateSeed:
    kind: str
    target_base: int
    actor_field_offset: int | None
    baseline_support: int
    direct_module_offsets: tuple[int, ...]

    @property
    def identity(self) -> tuple[str, int, int | None]:
        return (self.kind, self.target_base, self.actor_field_offset)


@dataclass(frozen=True, slots=True)
class _ReferenceScan:
    references_by_actor: Mapping[int, tuple[int, ...]]
    bytes_read: int
    regions_read: int
    failures: int

    @property
    def matches(self) -> int:
        return sum(len(values) for values in self.references_by_actor.values())


def _u32(data: bytes, offset: int) -> int:
    return int(struct.unpack_from("<I", data, offset)[0])


def _i32(data: bytes, offset: int) -> int:
    return int(struct.unpack_from("<i", data, offset)[0])


def _f32(data: bytes, offset: int) -> float:
    return float(struct.unpack_from("<f", data, offset)[0])


def _read_actor_state(
    memory: AggregateScanMemory,
    observation: AnchoredMonsterObservation,
    *,
    span: int,
    field_offsets: tuple[int, ...],
    coordinate_limit: float,
) -> CohortActorState:
    required = max(
        observation.species_offset,
        observation.active_species_offset,
        observation.hp_offset,
        observation.x_offset,
        observation.y_offset,
        observation.z_offset,
        *observation.self_pointer_offsets,
        *(field_offsets or (0,)),
    ) + 4
    try:
        data = memory.read(observation.base, max(required, int(span)))
        species = _i32(data, observation.species_offset)
        active_species = _i32(data, observation.active_species_offset)
        hp = _i32(data, observation.hp_offset)
        x = _f32(data, observation.x_offset)
        y = _f32(data, observation.y_offset)
        z = _f32(data, observation.z_offset)
        self_valid = any(
            _u32(data, offset) == observation.base
            for offset in observation.self_pointer_offsets
        )
        finite = all(math.isfinite(value) for value in (x, y, z))
        plausible = finite and all(
            abs(value) <= coordinate_limit for value in (x, y, z)
        )
        active = (
            self_valid
            and species == observation.species
            and active_species == species
            and hp > 0
            and plausible
        )
        values = tuple((offset, _u32(data, offset)) for offset in field_offsets)
        return CohortActorState(
            base=observation.base,
            species=species,
            active_species=active_species,
            hp=hp,
            x=x,
            y=y,
            z=z,
            self_valid=self_valid,
            readable=True,
            active=active,
            field_values=values,
        )
    except Exception:
        return CohortActorState(
            base=observation.base,
            species=0,
            active_species=0,
            hp=0,
            x=0.0,
            y=0.0,
            z=0.0,
            self_valid=False,
            readable=False,
            active=False,
        )


def _discover_actor_field_seeds(
    memory: AggregateScanMemory,
    cohort: tuple[AnchoredMonsterObservation, ...],
    module: ModuleInfo,
    readable: ReadableRegionIndex,
    *,
    object_span: int,
    minimum_support: int,
    maximum_candidates: int,
    check: Callable[[], None] | None,
) -> tuple[_CandidateSeed, ...]:
    support: dict[tuple[int, int], set[int]] = defaultdict(set)
    actor_bases = {item.base for item in cohort}
    module_start = int(module.base_address)
    module_stop = module_start + int(module.size)

    for observation in cohort:
        if check is not None:
            check()
        try:
            data = memory.read(observation.base, object_span)
        except Exception:
            continue
        excluded = {
            observation.species_offset,
            observation.active_species_offset,
            observation.hp_offset,
            observation.x_offset,
            observation.y_offset,
            observation.z_offset,
            *observation.self_pointer_offsets,
        }
        for offset in range(0, len(data) - 3, 4):
            if offset in excluded:
                continue
            value = _u32(data, offset)
            if (
                value <= 0x10000
                or value in actor_bases
                or module_start <= value < module_stop
                or not readable.contains(value, 4)
            ):
                continue
            support[(offset, value)].add(observation.base)

    seeds = [
        _CandidateSeed(
            kind="actor_field_target",
            target_base=target,
            actor_field_offset=offset,
            baseline_support=len(bases),
            direct_module_offsets=(),
        )
        for (offset, target), bases in support.items()
        if len(bases) >= minimum_support
    ]
    seeds.sort(
        key=lambda item: (
            -item.baseline_support,
            item.actor_field_offset or 0,
            item.target_base,
        )
    )
    return tuple(seeds[:maximum_candidates])


def scan_aligned_cohort_references(
    memory: AggregateScanMemory,
    regions: tuple[object, ...],
    actor_bases: tuple[int, ...],
    *,
    maximum_scan_bytes: int,
    chunk_size: int = 4 << 20,
    maximum_references_per_actor: int = 8192,
    check: Callable[[], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> _ReferenceScan:
    """Find aligned references to every cohort member in one process scan."""

    if maximum_scan_bytes <= 0:
        raise ValueError("maximum_scan_bytes must be positive")
    if chunk_size < 0x1000:
        raise ValueError("chunk_size must be at least 0x1000")
    unique_bases = tuple(sorted(set(int(value) for value in actor_bases)))
    if not unique_bases:
        return _ReferenceScan({}, 0, 0, 0)

    encoded = {value.to_bytes(4, "little"): value for value in unique_bases}
    pattern = re.compile(b"(?:" + b"|".join(re.escape(key) for key in encoded) + b")")
    references: dict[int, list[int]] = {value: [] for value in unique_bases}
    seen: set[tuple[int, int]] = set()
    bytes_read = 0
    regions_read = 0
    failures = 0
    next_progress = 64 << 20

    for region in sorted(regions, key=lambda item: int(getattr(item, "base_address"))):
        if bytes_read >= maximum_scan_bytes:
            break
        if check is not None:
            check()
        base = int(getattr(region, "base_address"))
        size = min(
            int(getattr(region, "size")),
            maximum_scan_bytes - bytes_read,
        )
        if size <= 0:
            continue
        offset = 0
        carry = b""
        region_had_data = False
        while offset < size:
            if check is not None:
                check()
            amount = min(int(chunk_size), size - offset)
            try:
                data = memory.read(base + offset, amount)
            except Exception:
                failures += 1
                carry = b""
                offset += amount
                continue
            region_had_data = True
            bytes_read += len(data)
            haystack = carry + data
            haystack_base = base + offset - len(carry)
            for match in pattern.finditer(haystack):
                address = haystack_base + match.start()
                if address % 4:
                    continue
                target = encoded.get(match.group(0))
                if target is None:
                    continue
                identity = (target, address)
                bucket = references[target]
                if identity in seen or len(bucket) >= maximum_references_per_actor:
                    continue
                seen.add(identity)
                bucket.append(address)
            carry = haystack[-3:] if len(haystack) >= 3 else haystack
            offset += amount
            if progress is not None and bytes_read >= next_progress:
                progress(bytes_read, sum(len(values) for values in references.values()))
                next_progress += 64 << 20
        if region_had_data:
            regions_read += 1

    return _ReferenceScan(
        references_by_actor={
            actor: tuple(sorted(values))
            for actor, values in references.items()
            if values
        },
        bytes_read=bytes_read,
        regions_read=regions_read,
        failures=failures,
    )


def _discover_module_holder_seeds(
    reference_scan: _ReferenceScan,
    module: ModuleInfo,
    module_refs: Mapping[int, tuple[int, ...]],
    *,
    holder_span: int,
    minimum_support: int,
    maximum_candidates: int,
) -> tuple[_CandidateSeed, ...]:
    flattened = sorted(
        (address, actor)
        for actor, addresses in reference_scan.references_by_actor.items()
        for address in addresses
    )
    if not flattened:
        return ()
    addresses = [item[0] for item in flattened]
    module_base = int(module.base_address)
    actor_bases = set(reference_scan.references_by_actor)
    seeds: list[_CandidateSeed] = []

    for target, slots in module_refs.items():
        target = int(target)
        if target in actor_bases:
            continue
        left = bisect_left(addresses, target)
        right = bisect_left(addresses, target + holder_span)
        if right - left < minimum_support:
            continue
        distinct = {actor for _address, actor in flattened[left:right]}
        if len(distinct) < minimum_support:
            continue
        offsets = tuple(
            sorted(
                slot - module_base
                for slot in slots
                if module_base <= slot < module_base + module.size
            )
        )
        if not offsets:
            continue
        seeds.append(
            _CandidateSeed(
                kind="module_rooted_holder",
                target_base=target,
                actor_field_offset=None,
                baseline_support=len(distinct),
                direct_module_offsets=offsets,
            )
        )

    seeds.sort(
        key=lambda item: (
            -item.baseline_support,
            len(item.direct_module_offsets),
            item.target_base,
        )
    )
    return tuple(seeds[:maximum_candidates])


def _actor_changed(first: CohortActorState, second: CohortActorState) -> bool:
    return (
        first.readable != second.readable
        or first.active != second.active
        or first.species != second.species
        or first.active_species != second.active_species
        or first.hp != second.hp
    )


def _paths_for_target(
    memory: AggregateScanMemory,
    module: ModuleInfo,
    readable: ReadableRegionIndex,
    module_refs: Mapping[int, tuple[int, ...]],
    target: int,
    direct_offsets: tuple[int, ...],
    *,
    maximum_depth: int,
    field_span: int,
    maximum_roots: int,
    maximum_nodes: int,
    check: Callable[[], None] | None,
) -> tuple[PointerPath, ...]:
    direct = tuple(PointerPath(offset, ()) for offset in direct_offsets)
    if direct:
        return direct
    paths = scan_module_rooted_paths(
        memory,
        module,
        readable,
        dict(module_refs),
        target,
        maximum_depth=maximum_depth,
        field_span=field_span,
        maximum_roots=maximum_roots,
        maximum_nodes=maximum_nodes,
        maximum_paths=64,
        check=check,
    )
    verified: list[PointerPath] = []
    for path in paths:
        try:
            if resolve_pointer_path(memory, module.base_address, path) == target:
                verified.append(path)
        except Exception:
            continue
    return tuple(verified)


def scan_aggregate_monster_roots(
    memory: AggregateScanMemory,
    cohort: tuple[AnchoredMonsterObservation, ...],
    module: ModuleInfo,
    regions: tuple[object, ...],
    readable: ReadableRegionIndex,
    module_refs: Mapping[int, tuple[int, ...]],
    *,
    duration_seconds: float = 30.0,
    interval_seconds: float = 1.0,
    object_span: int = 0x4000,
    holder_span: int = 0x4000,
    minimum_support: int = 4,
    maximum_field_candidates: int = 48,
    maximum_holder_candidates: int = 48,
    maximum_scan_bytes: int = 1536 << 20,
    maximum_depth: int = 2,
    path_field_span: int = 0x1000,
    maximum_roots: int = 12000,
    maximum_nodes: int = 100000,
    path_candidate_limit: int = 12,
    coordinate_limit: float = 1_000_000.0,
    check: Callable[[], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
    on_sampling_started: Callable[[int], None] | None = None,
    before_sample: Callable[[int], None] | None = None,
) -> AggregateCohortReport:
    """Rank world/manager roots using the whole monster cohort as one signal.

    Individual monsters are never selected or named. Any HP/active-state churn
    generated by ordinary farming contributes to the aggregate transition score.
    The scan is read-only and never updates runtime configuration.
    """

    if len(cohort) < 2:
        raise ValueError("aggregate scan requires at least two monster actors")
    if duration_seconds < 0.0:
        raise ValueError("duration_seconds cannot be negative")
    if interval_seconds <= 0.0:
        raise ValueError("interval_seconds must be positive")
    if object_span < 4 or object_span % 4:
        raise ValueError("object_span must be a positive multiple of four")
    if holder_span < 4 or holder_span % 4:
        raise ValueError("holder_span must be a positive multiple of four")
    if minimum_support < 2:
        raise ValueError("minimum_support must be at least two")

    field_seeds = _discover_actor_field_seeds(
        memory,
        cohort,
        module,
        readable,
        object_span=object_span,
        minimum_support=minimum_support,
        maximum_candidates=maximum_field_candidates,
        check=check,
    )
    reference_scan = scan_aligned_cohort_references(
        memory,
        regions,
        tuple(item.base for item in cohort),
        maximum_scan_bytes=maximum_scan_bytes,
        check=check,
        progress=progress,
    )
    holder_seeds = _discover_module_holder_seeds(
        reference_scan,
        module,
        module_refs,
        holder_span=holder_span,
        minimum_support=minimum_support,
        maximum_candidates=maximum_holder_candidates,
    )

    # Remove exact duplicate identities while preserving the stronger baseline.
    by_identity: dict[tuple[str, int, int | None], _CandidateSeed] = {}
    for seed in (*field_seeds, *holder_seeds):
        current = by_identity.get(seed.identity)
        if current is None or seed.baseline_support > current.baseline_support:
            by_identity[seed.identity] = seed
    seeds = tuple(by_identity.values())
    field_offsets = tuple(
        sorted(
            {
                int(seed.actor_field_offset)
                for seed in seeds
                if seed.actor_field_offset is not None
            }
        )
    )

    sample_target = max(2, int(math.floor(duration_seconds / interval_seconds)) + 1)
    if on_sampling_started is not None:
        on_sampling_started(sample_target)
    samples: list[dict[int, CohortActorState]] = []
    support_sets: dict[tuple[str, int, int | None], list[set[int]]] = {
        seed.identity: [] for seed in seeds
    }
    start = monotonic()
    for index in range(sample_target):
        if check is not None:
            check()
        if before_sample is not None:
            before_sample(index)
        states = {
            observation.base: _read_actor_state(
                memory,
                observation,
                span=object_span,
                field_offsets=field_offsets,
                coordinate_limit=coordinate_limit,
            )
            for observation in cohort
        }
        active_bases = {base for base, state in states.items() if state.active}
        samples.append(states)
        for seed in seeds:
            if seed.kind == "actor_field_target":
                assert seed.actor_field_offset is not None
                associated = {
                    base
                    for base in active_bases
                    if states[base].field_value(seed.actor_field_offset)
                    == seed.target_base
                }
            else:
                try:
                    data = memory.read(seed.target_base, holder_span)
                    referenced = {
                        _u32(data, offset)
                        for offset in range(0, len(data) - 3, 4)
                    }
                    associated = active_bases & referenced
                except Exception:
                    associated = set()
            support_sets[seed.identity].append(associated)
        if index + 1 >= sample_target:
            break
        remaining = duration_seconds - (monotonic() - start)
        if remaining <= 0.0:
            break
        sleep(min(interval_seconds, remaining))

    changed_actors: set[int] = set()
    transition_events = 0
    changed_by_transition: list[set[int]] = []
    for first, second in zip(samples, samples[1:]):
        changed = {
            base
            for base in first
            if _actor_changed(first[base], second[base])
        }
        changed_by_transition.append(changed)
        changed_actors.update(changed)
        transition_events += len(changed)

    candidates: list[AggregateCandidate] = []
    for seed in seeds:
        associations = support_sets[seed.identity]
        active_counts = tuple(
            sum(1 for state in sample.values() if state.active) for sample in samples
        )
        supports = tuple(len(values) for values in associations)
        coverages = tuple(
            support / active if active > 0 else 0.0
            for support, active in zip(supports, active_counts)
        )
        changed_support: set[int] = set()
        churn = 0
        for index, changed in enumerate(changed_by_transition):
            before = associations[index]
            after = associations[index + 1]
            changed_support.update(changed & (before | after))
            churn += len(before.symmetric_difference(after))
        candidates.append(
            AggregateCandidate(
                kind=seed.kind,
                target_base=seed.target_base,
                actor_field_offset=seed.actor_field_offset,
                baseline_support=seed.baseline_support,
                sample_support=supports,
                sample_active_counts=active_counts,
                average_active_coverage=(
                    sum(coverages) / len(coverages) if coverages else 0.0
                ),
                minimum_active_coverage=min(coverages) if coverages else 0.0,
                changed_actor_support=len(changed_support),
                reference_churn=churn,
                direct_module_offsets=seed.direct_module_offsets,
                pointer_paths=(),
                recommended=False,
            )
        )

    candidates.sort(
        key=lambda item: (
            -item.minimum_active_coverage,
            -item.average_active_coverage,
            -item.changed_actor_support,
            -item.baseline_support,
            item.kind,
            item.target_base,
        )
    )
    path_targets = {
        (item.kind, item.target_base, item.actor_field_offset)
        for item in candidates[: max(0, int(path_candidate_limit))]
    }
    completed: list[AggregateCandidate] = []
    for item in candidates:
        identity = (item.kind, item.target_base, item.actor_field_offset)
        paths: tuple[PointerPath, ...] = ()
        if identity in path_targets:
            paths = _paths_for_target(
                memory,
                module,
                readable,
                module_refs,
                item.target_base,
                item.direct_module_offsets,
                maximum_depth=maximum_depth,
                field_span=path_field_span,
                maximum_roots=maximum_roots,
                maximum_nodes=maximum_nodes,
                check=check,
            )
        recommended = bool(
            paths
            and item.baseline_support >= minimum_support
            and item.average_active_coverage >= 0.50
            and item.minimum_active_coverage >= 0.25
            and (not changed_actors or item.changed_actor_support > 0)
        )
        completed.append(replace(item, pointer_paths=paths, recommended=recommended))

    completed.sort(
        key=lambda item: (
            int(not item.recommended),
            int(not item.pointer_paths),
            -item.minimum_active_coverage,
            -item.average_active_coverage,
            -item.changed_actor_support,
            -item.baseline_support,
            item.shortest_path_depth if item.shortest_path_depth is not None else 99,
            item.target_base,
        )
    )
    return AggregateCohortReport(
        schema_version=1,
        captured_at_utc=datetime.now(UTC).isoformat(),
        pid=int(memory.pid),
        module_name=module.name,
        module_path=module.path,
        module_base=int(module.base_address),
        module_size=int(module.size),
        cohort_size=len(cohort),
        sample_count=len(samples),
        duration_seconds=max(0.0, monotonic() - start),
        changed_actor_count=len(changed_actors),
        transition_events=transition_events,
        reference_scan_bytes=reference_scan.bytes_read,
        reference_scan_regions=reference_scan.regions_read,
        reference_scan_failures=reference_scan.failures,
        reference_matches=reference_scan.matches,
        candidates=tuple(completed),
    )


def save_aggregate_report(
    report: AggregateCohortReport,
    directory: str | Path,
) -> Path:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    path = destination / f"aggregate_monster_scan_{stamp}.json"
    path.write_text(report.to_json(), encoding="utf-8")
    return path
