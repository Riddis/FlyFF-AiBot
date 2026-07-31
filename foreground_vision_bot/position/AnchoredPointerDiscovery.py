from __future__ import annotations

import math
import struct
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from time import sleep
from typing import Protocol

MEM_PRIVATE = 0x20000


class AnchoredDiscoveryMemory(Protocol):
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
    ) -> tuple[int, ...]: ...


@dataclass(frozen=True, slots=True)
class PointerRecoveryHints:
    """Trusted live facts used only by explicit anchored recovery."""

    known_species_ids: tuple[int, ...] = ()
    player_spawn_x: float | None = None
    player_spawn_z: float | None = None
    player_current_hp: int | None = None
    player_max_hp: int | None = None
    spawn_tolerance_native: float = 2.0
    movement_minimum_native: float = 0.5

    def __post_init__(self) -> None:
        if any(value <= 0 for value in self.known_species_ids):
            raise ValueError("known_species_ids must contain positive values")
        if (self.player_spawn_x is None) != (self.player_spawn_z is None):
            raise ValueError("player spawn X and Z must be provided together")
        if self.player_current_hp is not None and self.player_current_hp <= 0:
            raise ValueError("player_current_hp must be positive")
        if self.player_max_hp is not None and self.player_max_hp <= 0:
            raise ValueError("player_max_hp must be positive")
        if (
            self.player_current_hp is not None
            and self.player_max_hp is not None
            and self.player_current_hp > self.player_max_hp
        ):
            raise ValueError("player_current_hp cannot exceed player_max_hp")
        if self.spawn_tolerance_native <= 0.0:
            raise ValueError("spawn_tolerance_native must be positive")
        if self.movement_minimum_native <= 0.0:
            raise ValueError("movement_minimum_native must be positive")

    @property
    def ready(self) -> bool:
        return bool(
            self.known_species_ids
            and self.player_spawn_x is not None
            and self.player_spawn_z is not None
            and self.player_current_hp is not None
            and self.player_max_hp is not None
        )


@dataclass(slots=True)
class AnchoredDiscoveryEvidence:
    anchor_bytes_scanned: int = 0
    anchor_regions_scanned: int = 0
    anchor_read_failures: int = 0
    species_value_matches: int = 0
    spawn_x_matches: int = 0
    monster_candidates: int = 0
    monster_base_hypotheses: int = 0
    monster_layout_species_support: int = 0
    monster_layout_ties: int = 0
    monster_self_field_aliases: int = 0
    monster_species_rejections: int = 0
    monster_active_rejections: int = 0
    monster_hp_rejections: int = 0
    monster_coordinate_rejections: int = 0
    inferred_world_actor_support: int = 0
    inferred_world_species_support: int = 0
    inferred_world_offset: int | None = None
    inferred_world_vtable: int | None = None
    inferred_world_vtable_field_offset: int | None = None
    inferred_world_identity_kind: str | None = None
    inferred_world_readable_pointer_fields: int = 0
    inferred_world_distinct_values: int = 0
    structural_world_hypotheses: int = 0
    world_object_rejections: int = 0
    world_identity_candidates: int = 0
    world_identity_span_rejections: int = 0
    world_identity_vtable_misses: int = 0
    world_identity_unstable_rejections: int = 0
    world_identity_module_pointer_fields: int = 0
    world_identity_readable_pointer_fields: int = 0
    world_identity_distinct_values: int = 0
    world_identity_structural_rejections: int = 0
    world_identity_marker_accepts: int = 0
    world_near_actor_support: int = 0
    world_near_species_support: int = 0
    world_near_module_references: int = 0
    world_near_field_offset: int | None = None
    world_near_target: int | None = None
    world_near_module_pointer_fields: int = 0
    world_near_readable_pointer_fields: int = 0
    world_near_distinct_values: int = 0
    inferred_self_actor_support: int = 0
    inferred_self_offset: int | None = None
    inferred_species_offset: int | None = None
    inferred_active_species_offset: int | None = None
    inferred_hp_offset: int | None = None
    inferred_player_hp_offset: int | None = None
    inferred_player_max_hp_offset: int | None = None
    inferred_x_offset: int | None = None
    inferred_y_offset: int | None = None
    inferred_z_offset: int | None = None
    spawn_structure_candidates: int = 0
    spawn_world_matches: int = 0
    spawn_world_hypothesis_matches: int = 0
    spawn_hp_matches: int = 0
    spawn_player_matches: int = 0
    stable_spawn_candidates: int = 0
    direct_player_slot_candidates: int = 0
    player_chain_candidates: int = 0
    direct_world_slot_candidates: int = 0
    world_chain_candidates: int = 0
    player_reference_matches: int = 0
    player_world_chain_candidates: int = 0
    player_reference_ambiguities: int = 0
    anchor_ambiguities: int = 0
    movement_checks: int = 0
    movement_observed: int = 0


@dataclass(frozen=True, slots=True)
class AnchoredPointerCandidate:
    player_base: int
    world_base: int
    player_pointer_address: int
    player_pointer_chain_offsets: tuple[int, ...]
    world_pointer_address: int
    world_pointer_chain_offsets: tuple[int, ...]
    world_field_offset: int
    world_vtable: int
    world_vtable_field_offset: int
    world_identity_kind: str
    self_pointer_offset: int
    hp_offset: int
    monster_hp_offset: int
    max_hp_offset: int
    species_offset: int
    active_species_offset: int
    x_offset: int
    y_offset: int
    z_offset: int
    baseline_x: float
    baseline_y: float
    baseline_z: float
    baseline_hp: int
    maximum_hp: int
    monster_actor_support: int
    monster_species_support: int


@dataclass(frozen=True, slots=True)
class AnchoredDiscoveryResult:
    outcome: str
    candidate: AnchoredPointerCandidate | None
    evidence: AnchoredDiscoveryEvidence
    message: str


@dataclass(frozen=True, slots=True)
class _MonsterAnchor:
    base: int
    species: int
    data: bytes


@dataclass(frozen=True, slots=True)
class _InferredActorLayout:
    species_offset: int
    active_species_offset: int
    hp_offset: int
    x_offset: int
    y_offset: int
    z_offset: int
    self_offsets: tuple[int, ...]

    @property
    def self_offset(self) -> int:
        return self.self_offsets[0]


@dataclass(frozen=True, slots=True)
class _MonsterHypothesis:
    actor: _MonsterAnchor
    layout: _InferredActorLayout


@dataclass(frozen=True, slots=True)
class _WorldHypothesis:
    score: tuple[int, int, int, int]
    field_offset: int
    world_base: int
    identity_kind: str
    identity_field_offset: int
    identity_value: int
    readable_pointer_fields: int
    distinct_values: int
    actor_support: int
    species_support: int
    module_references: int


@dataclass(frozen=True, slots=True)
class _ModuleReferenceResult:
    slot: int | None = None
    chain_offsets: tuple[int, ...] = ()
    direct_candidates: int = 0
    chained_candidates: int = 0
    reference_matches: int = 0
    preferred_chain_candidates: int = 0
    ambiguous_candidates: int = 0

    @property
    def resolved(self) -> bool:
        return self.slot is not None


def _bounds(region: object) -> tuple[int, int]:
    base = int(getattr(region, "base_address"))
    size = int(getattr(region, "size"))
    return base, base + max(0, size)


def _private_regions(regions: Iterable[object]) -> tuple[object, ...]:
    ordered = sorted(
        (
        region
        for region in regions
        if int(getattr(region, "region_type", MEM_PRIVATE)) == MEM_PRIVATE
        and _bounds(region)[1] > _bounds(region)[0]
        ),
        key=lambda region: _bounds(region)[0],
    )
    # Alternate high and low virtual addresses so a byte cap cannot exclude
    # every late private region merely because earlier regions are large.
    balanced: list[object] = []
    low = 0
    high = len(ordered) - 1
    while low <= high:
        balanced.append(ordered[high])
        high -= 1
        if low <= high:
            balanced.append(ordered[low])
            low += 1
    return tuple(balanced)


def _u32(data: bytes, offset: int) -> int:
    return int(struct.unpack_from("<I", data, offset)[0])


def _i32(data: bytes, offset: int) -> int:
    return int(struct.unpack_from("<i", data, offset)[0])


def _f32(data: bytes, offset: int) -> float:
    return float(struct.unpack_from("<f", data, offset)[0])


def _find_aligned(data: bytes, needle: bytes, absolute_start: int) -> tuple[int, ...]:
    result: list[int] = []
    cursor = 0
    while True:
        found = data.find(needle, cursor)
        if found < 0:
            return tuple(result)
        absolute = absolute_start + found
        if absolute % 4 == 0:
            result.append(absolute)
        cursor = found + 1


def _contains(
    readable_contains: Callable[[int, int], bool],
    address: int,
    size: int,
) -> bool:
    return address > 0 and readable_contains(int(address), int(size))


def _scan_anchor_values(
    memory: AnchoredDiscoveryMemory,
    regions: tuple[object, ...],
    hints: PointerRecoveryHints,
    *,
    chunk_size: int,
    maximum_bytes: int,
    check: Callable[[], None],
    evidence: AnchoredDiscoveryEvidence,
    progress_callback: Callable[[AnchoredDiscoveryEvidence], None] | None,
) -> tuple[dict[int, set[int]], set[int]]:
    species_matches: dict[int, set[int]] = {
        value: set() for value in hints.known_species_ids
    }
    spawn_matches: set[int] = set()
    species_needles = {
        value: struct.pack("<I", int(value)) for value in hints.known_species_ids
    }
    spawn_needle = (
        None
        if hints.player_spawn_x is None
        else struct.pack("<f", float(hints.player_spawn_x))
    )
    budget = max(0x10000, int(maximum_bytes))
    amount_per_read = max(0x1000, int(chunk_size))
    next_progress = 0x4000000

    for region in _private_regions(regions):
        check()
        if evidence.anchor_bytes_scanned >= budget:
            break
        start, stop = _bounds(region)
        evidence.anchor_regions_scanned += 1
        cursor = start
        while cursor < stop and evidence.anchor_bytes_scanned < budget:
            check()
            amount = min(
                amount_per_read + 3,
                stop - cursor,
                budget - evidence.anchor_bytes_scanned,
            )
            if amount < 4:
                break
            try:
                data = memory.read(cursor, amount)
            except Exception:
                evidence.anchor_read_failures += 1
                cursor += amount_per_read
                continue
            evidence.anchor_bytes_scanned += len(data)
            for species, needle in species_needles.items():
                species_matches[species].update(_find_aligned(data, needle, cursor))
            if spawn_needle is not None:
                spawn_matches.update(_find_aligned(data, spawn_needle, cursor))
            if (
                progress_callback is not None
                and evidence.anchor_bytes_scanned >= next_progress
            ):
                evidence.species_value_matches = sum(
                    len(matches) for matches in species_matches.values()
                )
                evidence.spawn_x_matches = len(spawn_matches)
                progress_callback(evidence)
                next_progress += 0x4000000
            cursor += amount_per_read
            sleep(0)

    evidence.species_value_matches = sum(
        len(matches) for matches in species_matches.values()
    )
    evidence.spawn_x_matches = len(spawn_matches)
    return species_matches, spawn_matches


def _monster_anchors(
    memory: AnchoredDiscoveryMemory,
    species_matches: Mapping[int, set[int]],
    *,
    species_offset: int,
    active_species_offset: int,
    hp_offset: int,
    x_offset: int,
    y_offset: int,
    z_offset: int,
    object_span: int,
    coordinate_limit: float,
    readable_contains: Callable[[int, int], bool],
    maximum_candidates: int,
    check: Callable[[], None],
    evidence: AnchoredDiscoveryEvidence,
) -> tuple[tuple[_MonsterAnchor, ...], _InferredActorLayout | None]:
    hypotheses: list[_MonsterHypothesis] = []
    accepted_by_species: dict[int, int] = defaultdict(int)
    seen: set[tuple[int, int, int]] = set()
    coordinate_deltas = (
        x_offset - species_offset,
        y_offset - species_offset,
        z_offset - species_offset,
    )
    hp_delta = hp_offset - species_offset
    active_delta = active_species_offset - species_offset
    page_size = 0x1000
    local_window_size = 0x5000
    maximum_species_offset = 0x1000
    maximum_actor_span = max(0x4000, object_span)
    per_species_limit = max(2, maximum_candidates // max(1, len(species_matches)))

    for expected_species, matches in species_matches.items():
        for address in sorted(matches):
            check()
            if accepted_by_species[expected_species] >= per_species_limit:
                break
            # A species value is an address anchor, not proof that the old
            # species offset still identifies the actor base. Read its local
            # allocation and look for self references that imply both base and
            # current species offset.
            window_start = int(address) & ~(page_size - 1)
            if not _contains(
                readable_contains,
                window_start,
                local_window_size,
            ):
                evidence.monster_species_rejections += 1
                continue
            try:
                window = memory.read(window_start, local_window_size)
            except Exception:
                evidence.monster_species_rejections += 1
                continue

            local_candidates = 0
            for relative in range(0, len(window) - 3, 4):
                if relative % 0x1000 == 0:
                    check()
                base = _u32(window, relative)
                current_species_offset = int(address) - base
                current_self_offset = window_start + relative - base
                identity = (base, current_species_offset, current_self_offset)
                if (
                    base <= 0x10000
                    or current_species_offset < 0x40
                    or current_species_offset > maximum_species_offset
                    or current_self_offset < 0
                    or current_self_offset >= maximum_actor_span
                    or identity in seen
                    or not _contains(
                        readable_contains,
                        base,
                        maximum_actor_span,
                    )
                ):
                    continue
                seen.add(identity)
                local_candidates += 1
                evidence.monster_base_hypotheses += 1

                current_x_offset = current_species_offset + coordinate_deltas[0]
                current_y_offset = current_species_offset + coordinate_deltas[1]
                current_z_offset = current_species_offset + coordinate_deltas[2]
                current_hp_offset = current_species_offset + hp_delta
                if min(
                    current_x_offset,
                    current_y_offset,
                    current_z_offset,
                    current_hp_offset,
                ) < 0:
                    evidence.monster_species_rejections += 1
                    continue
                try:
                    data = memory.read(base, maximum_actor_span)
                    species = _i32(data, current_species_offset)
                    hp = _i32(data, current_hp_offset)
                    coordinates = (
                        _f32(data, current_x_offset),
                        _f32(data, current_y_offset),
                        _f32(data, current_z_offset),
                    )
                except Exception:
                    evidence.monster_species_rejections += 1
                    continue
                if species != expected_species:
                    evidence.monster_species_rejections += 1
                    continue

                active_offsets = tuple(
                    offset
                    for offset in _aligned_value_offsets(data, expected_species)
                    if offset != current_species_offset
                )
                if not active_offsets:
                    evidence.monster_active_rejections += 1
                    continue
                current_active_offset = min(
                    active_offsets,
                    key=lambda offset: abs(
                        offset - (current_species_offset + active_delta)
                    ),
                )
                if hp <= 0:
                    evidence.monster_hp_rejections += 1
                    continue
                if not all(math.isfinite(value) for value in coordinates) or any(
                    abs(value) > coordinate_limit for value in coordinates
                ):
                    evidence.monster_coordinate_rejections += 1
                    continue
                hypotheses.append(
                    _MonsterHypothesis(
                        actor=_MonsterAnchor(base, species, data),
                        layout=_InferredActorLayout(
                            species_offset=current_species_offset,
                            active_species_offset=current_active_offset,
                            hp_offset=current_hp_offset,
                            x_offset=current_x_offset,
                            y_offset=current_y_offset,
                            z_offset=current_z_offset,
                            self_offsets=(current_self_offset,),
                        ),
                    )
                )
                accepted_by_species[expected_species] += 1
            if local_candidates == 0:
                evidence.monster_species_rejections += 1
            sleep(0)

    support: dict[_InferredActorLayout, dict[int, _MonsterAnchor]] = defaultdict(dict)
    for hypothesis in hypotheses:
        support[hypothesis.layout][hypothesis.actor.base] = hypothesis.actor
    if not support:
        evidence.monster_candidates = 0
        return (), None

    # Several fields inside a live actor can legitimately point back to the
    # actor base. They are equivalent self-field aliases, not independent
    # species/HP/transform layouts. Collapse aliases only when they validate
    # the exact same actor cohort; genuinely different field families remain
    # ambiguous and are rejected below.
    family_support: dict[
        tuple[int, int, int, int, int, int],
        list[tuple[_InferredActorLayout, dict[int, _MonsterAnchor]]],
    ] = defaultdict(list)
    for layout, actors in support.items():
        family = (
            layout.species_offset,
            layout.active_species_offset,
            layout.hp_offset,
            layout.x_offset,
            layout.y_offset,
            layout.z_offset,
        )
        family_support[family].append((layout, actors))

    ranked_families: list[
        tuple[
            tuple[int, int],
            tuple[int, int, int, int, int, int],
            tuple[_MonsterAnchor, ...],
            tuple[int, ...],
        ]
    ] = []
    for family, aliases in family_support.items():
        aliases.sort(
            key=lambda item: (
                len(item[1]),
                len({actor.species for actor in item[1].values()}),
                -item[0].self_offset,
            ),
            reverse=True,
        )
        _preferred_layout, preferred_actors = aliases[0]
        preferred_bases = frozenset(preferred_actors)
        equivalent_offsets = tuple(
            sorted(
                layout.self_offset
                for layout, actors in aliases
                if frozenset(actors) == preferred_bases
            )
        )
        selected_actors = tuple(preferred_actors.values())
        score = (
            len(selected_actors),
            len({actor.species for actor in selected_actors}),
        )
        ranked_families.append(
            (score, family, selected_actors, equivalent_offsets)
        )
    ranked_families.sort(
        key=lambda item: (
            item[0],
            -abs(item[1][0] - species_offset),
            -abs(item[1][1] - active_species_offset),
        ),
        reverse=True,
    )
    selected_score, selected_family, selected_actors, self_offsets = (
        ranked_families[0]
    )
    selected_layout = _InferredActorLayout(
        species_offset=selected_family[0],
        active_species_offset=selected_family[1],
        hp_offset=selected_family[2],
        x_offset=selected_family[3],
        y_offset=selected_family[4],
        z_offset=selected_family[5],
        self_offsets=self_offsets,
    )
    required_species = min(2, len(species_matches))
    selected_species_count = selected_score[1]
    evidence.monster_layout_species_support = selected_species_count
    evidence.monster_self_field_aliases = len(self_offsets)
    if len(selected_actors) < 2 or not (
        selected_species_count >= required_species or len(selected_actors) >= 3
    ):
        evidence.monster_candidates = len(selected_actors)
        return selected_actors, None

    tied_layouts = sum(
        1
        for score, _family, _actors, _self_offsets in ranked_families[1:]
        if score == selected_score
    )
    evidence.monster_layout_ties = tied_layouts
    if tied_layouts:
        evidence.anchor_ambiguities += tied_layouts + 1
        evidence.monster_candidates = len(selected_actors)
        return selected_actors, None

    evidence.monster_candidates = len(selected_actors)
    evidence.inferred_species_offset = selected_layout.species_offset
    evidence.inferred_active_species_offset = selected_layout.active_species_offset
    evidence.inferred_hp_offset = selected_layout.hp_offset
    evidence.inferred_x_offset = selected_layout.x_offset
    evidence.inferred_y_offset = selected_layout.y_offset
    evidence.inferred_z_offset = selected_layout.z_offset
    evidence.inferred_self_offset = selected_layout.self_offset
    evidence.inferred_self_actor_support = len(selected_actors)
    return selected_actors, selected_layout


_WORLD_IDENTITY_SPAN = 0x400
_WORLD_VTABLE_PROBE_WORDS = 3
_WORLD_MARKER_MINIMUM_READABLE_POINTERS = 3
_WORLD_MARKER_MINIMUM_DISTINCT_VALUES = 8


def _looks_like_module_vtable(
    memory: AnchoredDiscoveryMemory,
    value: int,
    *,
    module_start: int,
    module_stop: int,
) -> bool:
    try:
        table = memory.read(value, _WORLD_VTABLE_PROBE_WORDS * 4)
    except Exception:
        return False
    module_entries = sum(
        module_start <= _u32(table, offset) < module_stop
        for offset in range(0, len(table), 4)
    )
    return module_entries >= 2


def _infer_world_identity(
    memory: AnchoredDiscoveryMemory,
    value: int,
    *,
    module_start: int,
    module_stop: int,
    readable_contains: Callable[[int, int], bool],
    evidence: AnchoredDiscoveryEvidence,
) -> tuple[str, int, int, int, int] | None:
    evidence.world_identity_candidates += 1
    if not readable_contains(value, _WORLD_IDENTITY_SPAN):
        evidence.world_identity_span_rejections += 1
        return None
    try:
        first = memory.read(value, _WORLD_IDENTITY_SPAN)
        second = memory.read(value, _WORLD_IDENTITY_SPAN)
    except Exception:
        evidence.world_identity_span_rejections += 1
        return None

    module_fields = tuple(
        (offset, _u32(first, offset))
        for offset in range(0, len(first) - 3, 4)
        if module_start <= _u32(first, offset) < module_stop
    )
    evidence.world_identity_module_pointer_fields += len(module_fields)
    stable_module_fields = tuple(
        (offset, candidate)
        for offset, candidate in module_fields
        if _u32(second, offset) == candidate
    )
    readable_pointer_fields = tuple(
        (offset, candidate)
        for offset in range(0, len(first) - 3, 4)
        if (
            (candidate := _u32(first, offset)) > 0x10000
            and not module_start <= candidate < module_stop
            and readable_contains(candidate, 4)
            and _u32(second, offset) == candidate
        )
    )
    distinct_values = len(
        {
            _u32(first, offset)
            for offset in range(0, len(first) - 3, 4)
            if _u32(first, offset) != 0
            and _u32(first, offset) == _u32(second, offset)
        }
    )
    evidence.world_identity_readable_pointer_fields += len(
        readable_pointer_fields
    )
    evidence.world_identity_distinct_values += distinct_values
    vtable_fields = tuple(
        (offset, candidate)
        for offset, candidate in module_fields
        if _looks_like_module_vtable(
            memory,
            candidate,
            module_start=module_start,
            module_stop=module_stop,
        )
    )
    if not vtable_fields:
        evidence.world_identity_vtable_misses += 1
    else:
        stable_fields = tuple(
            (offset, candidate)
            for offset, candidate in vtable_fields
            if _u32(second, offset) == candidate
        )
        if stable_fields:
            offset, candidate = min(
                stable_fields,
                key=lambda item: (item[0] != 0, item[0]),
            )
            return (
                "vtable",
                offset,
                candidate,
                len(readable_pointer_fields),
                distinct_values,
            )
        evidence.world_identity_unstable_rejections += 1

    # Some client managers are non-polymorphic but still carry a stable
    # module-owned identity marker. Admit that shape only when the surrounding
    # object is demonstrably pointer-rich and diverse; a readable scalar page
    # or a lone module literal cannot pass this fallback.
    if (
        stable_module_fields
        and len(readable_pointer_fields)
        >= _WORLD_MARKER_MINIMUM_READABLE_POINTERS
        and distinct_values >= _WORLD_MARKER_MINIMUM_DISTINCT_VALUES
    ):
        evidence.world_identity_marker_accepts += 1
        offset, candidate = min(
            stable_module_fields,
            key=lambda item: (item[0] != 0, item[0]),
        )
        return (
            "module_marker",
            offset,
            candidate,
            len(readable_pointer_fields),
            distinct_values,
        )
    evidence.world_identity_structural_rejections += 1
    return None


def _infer_worlds(
    anchors: tuple[_MonsterAnchor, ...],
    *,
    memory: AnchoredDiscoveryMemory,
    configured_world_offset: int,
    module_start: int,
    module_stop: int,
    slot_refs: Mapping[int, list[int]],
    readable_contains: Callable[[int, int], bool],
    evidence: AnchoredDiscoveryEvidence,
) -> tuple[_WorldHypothesis, ...]:
    support: dict[tuple[int, int], set[int]] = defaultdict(set)
    species_support: dict[tuple[int, int], set[int]] = defaultdict(set)
    for actor in anchors:
        scan_stop = min(len(actor.data), 0x1000)
        for offset in range(0, scan_stop - 3, 4):
            value = _u32(actor.data, offset)
            if (
                value <= 0x10000
                or module_start <= value < module_stop
                or not readable_contains(value, 4)
            ):
                continue
            key = (offset, value)
            support[key].add(actor.base)
            species_support[key].add(actor.species)

    ranked: list[_WorldHypothesis] = []
    best_near_score: tuple[int, int, int, int] | None = None
    for (offset, value), actors in support.items():
        module_refs = len(slot_refs.get(value, ()))
        distinct_species = len(species_support[(offset, value)])
        if len(actors) < 2:
            continue
        if module_refs <= 0 and (len(actors) < 3 or distinct_species < 2):
            continue
        score = (
            len(actors),
            distinct_species,
            min(module_refs, 8),
            -abs(offset - configured_world_offset),
        )
        identity_fields_before = evidence.world_identity_module_pointer_fields
        readable_fields_before = (
            evidence.world_identity_readable_pointer_fields
        )
        distinct_values_before = evidence.world_identity_distinct_values
        identity = _infer_world_identity(
            memory,
            value,
            module_start=module_start,
            module_stop=module_stop,
            readable_contains=readable_contains,
            evidence=evidence,
        )
        observed_identity_fields = (
            evidence.world_identity_module_pointer_fields - identity_fields_before
        )
        observed_readable_fields = (
            evidence.world_identity_readable_pointer_fields
            - readable_fields_before
        )
        observed_distinct_values = (
            evidence.world_identity_distinct_values - distinct_values_before
        )
        if best_near_score is None or score > best_near_score:
            best_near_score = score
            evidence.world_near_actor_support = len(actors)
            evidence.world_near_species_support = distinct_species
            evidence.world_near_module_references = module_refs
            evidence.world_near_field_offset = offset
            evidence.world_near_target = value
            evidence.world_near_module_pointer_fields = observed_identity_fields
            evidence.world_near_readable_pointer_fields = (
                observed_readable_fields
            )
            evidence.world_near_distinct_values = observed_distinct_values
        if identity is None:
            evidence.world_object_rejections += 1
            continue
        (
            identity_kind,
            world_vtable_field_offset,
            world_vtable,
            readable_pointer_fields,
            distinct_values,
        ) = identity
        ranked.append(
            _WorldHypothesis(
                score=score,
                field_offset=offset,
                world_base=value,
                identity_kind=identity_kind,
                identity_field_offset=world_vtable_field_offset,
                identity_value=world_vtable,
                readable_pointer_fields=readable_pointer_fields,
                distinct_values=distinct_values,
                actor_support=len(actors),
                species_support=distinct_species,
                module_references=module_refs,
            )
        )
    ranked.sort(
        key=lambda item: (
            item.score,
            -item.field_offset,
            -item.world_base,
        ),
        reverse=True,
    )
    evidence.structural_world_hypotheses = len(ranked)
    return tuple(ranked)


def _record_selected_world(
    evidence: AnchoredDiscoveryEvidence,
    world: _WorldHypothesis,
) -> None:
    evidence.inferred_world_actor_support = world.actor_support
    evidence.inferred_world_species_support = world.species_support
    evidence.inferred_world_offset = world.field_offset
    evidence.inferred_world_vtable = world.identity_value
    evidence.inferred_world_vtable_field_offset = world.identity_field_offset
    evidence.inferred_world_identity_kind = world.identity_kind
    evidence.inferred_world_readable_pointer_fields = (
        world.readable_pointer_fields
    )
    evidence.inferred_world_distinct_values = world.distinct_values


def _aligned_value_offsets(data: bytes, value: int) -> tuple[int, ...]:
    needle = struct.pack("<i", int(value))
    return tuple(
        absolute
        for absolute in _find_aligned(data, needle, 0)
        if 0 <= absolute <= len(data) - 4
    )


def _resolve_module_reference(
    memory: AnchoredDiscoveryMemory,
    value: int,
    *,
    configured_slot: int,
    slot_refs: Mapping[int, list[int]],
    maximum_address: int,
    chunk_size: int,
    cancellation: object | None,
    deadline: float,
    check: Callable[[], None],
    preferred_roots: tuple[tuple[int, int, tuple[int, ...]], ...] = (),
    maximum_preferred_offset: int = 0,
) -> _ModuleReferenceResult:
    direct = tuple(sorted(set(slot_refs.get(value, ()))))
    if configured_slot in direct:
        return _ModuleReferenceResult(
            slot=configured_slot,
            direct_candidates=len(direct),
        )
    if direct:
        # Every entry is a module-image slot containing this same exact target.
        # They are direct aliases, so select the slot nearest the configured
        # hint deterministically and let movement confirmation prove stability.
        selected = min(direct, key=lambda slot: (abs(slot - configured_slot), slot))
        return _ModuleReferenceResult(
            slot=selected,
            direct_candidates=len(direct),
        )

    try:
        references = memory.find_u32(
            value,
            maximum_address=maximum_address,
            private_only=False,
            chunk_size=chunk_size,
            cancellation=cancellation,
            deadline=deadline,
        )
    except Exception:
        return _ModuleReferenceResult(direct_candidates=len(direct))
    chains: set[tuple[int, int]] = set()
    preferred_chains: set[tuple[int, int]] = set()
    for reference in references[:4096]:
        check()
        for root_value, root_slot, root_chain in preferred_roots:
            offset = int(reference) - int(root_value)
            if (
                not root_chain
                and 0 <= offset <= maximum_preferred_offset
                and offset % 4 == 0
            ):
                preferred_chains.add((root_slot, offset))
        for offset in range(0, 0x104, 4):
            holder = int(reference) - offset
            for slot in slot_refs.get(holder, ()):
                chains.add((slot, offset))
    configured_chains = [item for item in chains if item[0] == configured_slot]
    if len(configured_chains) == 1:
        slot, offset = configured_chains[0]
        return _ModuleReferenceResult(
            slot=slot,
            chain_offsets=(offset,),
            direct_candidates=len(direct),
            chained_candidates=len(chains | preferred_chains),
            reference_matches=len(references),
            preferred_chain_candidates=len(preferred_chains),
        )
    if preferred_chains:
        # Multiple fields under the same confirmed world root that all contain
        # this exact player base are equivalent chain aliases. Select one
        # deterministically; repeated stationary reads and the mandatory
        # movement sample still have to prove it stable before persistence.
        slot, offset = min(preferred_chains, key=lambda item: (item[1], item[0]))
        return _ModuleReferenceResult(
            slot=slot,
            chain_offsets=(offset,),
            direct_candidates=len(direct),
            chained_candidates=len(chains | preferred_chains),
            reference_matches=len(references),
            preferred_chain_candidates=len(preferred_chains),
        )
    if not direct and len(chains) == 1:
        slot, offset = next(iter(chains))
        return _ModuleReferenceResult(
            slot=slot,
            chain_offsets=(offset,),
            chained_candidates=1,
            reference_matches=len(references),
            preferred_chain_candidates=len(preferred_chains),
        )
    ambiguous = len(direct) + len(chains | preferred_chains)
    return _ModuleReferenceResult(
        direct_candidates=len(direct),
        chained_candidates=len(chains | preferred_chains),
        reference_matches=len(references),
        preferred_chain_candidates=len(preferred_chains),
        ambiguous_candidates=ambiguous,
    )


def _resolve_pointer(
    memory: AnchoredDiscoveryMemory,
    slot: int,
    chain_offsets: tuple[int, ...],
) -> int:
    value = int(struct.unpack("<I", memory.read(slot, 4))[0])
    for offset in chain_offsets:
        value = int(struct.unpack("<I", memory.read(value + offset, 4))[0])
    return value


def discover_anchored_pointer_candidate(
    memory: AnchoredDiscoveryMemory,
    *,
    regions: tuple[object, ...],
    slot_refs: Mapping[int, list[int]],
    hints: PointerRecoveryHints,
    module_base: int,
    module_stop: int,
    configured_player_slot: int,
    configured_world_slot: int,
    species_offset: int,
    active_species_offset: int,
    hp_offset: int,
    x_offset: int,
    y_offset: int,
    z_offset: int,
    configured_world_field_offset: int,
    configured_self_offset: int,
    coordinate_limit: float,
    maximum_address: int,
    chunk_size: int,
    cancellation: object | None,
    deadline: float,
    readable_contains: Callable[[int, int], bool],
    check: Callable[[], None],
    stability_samples: int = 3,
    stability_delay_seconds: float = 0.03,
    maximum_scan_bytes: int = 0x60000000,
    maximum_monster_candidates: int = 512,
    progress_callback: Callable[[AnchoredDiscoveryEvidence], None] | None = None,
) -> AnchoredDiscoveryResult:
    evidence = AnchoredDiscoveryEvidence()
    if not hints.ready:
        return AnchoredDiscoveryResult(
            "anchor_hints_required",
            None,
            evidence,
            "Known species, Tower spawn, and exact current/maximum HP are required.",
        )
    assert hints.player_spawn_x is not None
    assert hints.player_spawn_z is not None
    assert hints.player_current_hp is not None
    assert hints.player_max_hp is not None

    species_matches, spawn_matches = _scan_anchor_values(
        memory,
        regions,
        hints,
        chunk_size=chunk_size,
        maximum_bytes=maximum_scan_bytes,
        check=check,
        evidence=evidence,
        progress_callback=progress_callback,
    )
    object_span = max(
        configured_self_offset + 0x200,
        active_species_offset,
        hp_offset,
        z_offset,
        0x4000,
    ) + 4
    anchors, actor_layout = _monster_anchors(
        memory,
        species_matches,
        species_offset=species_offset,
        active_species_offset=active_species_offset,
        hp_offset=hp_offset,
        x_offset=x_offset,
        y_offset=y_offset,
        z_offset=z_offset,
        object_span=object_span,
        coordinate_limit=coordinate_limit,
        readable_contains=readable_contains,
        maximum_candidates=maximum_monster_candidates,
        check=check,
        evidence=evidence,
    )
    if len(anchors) < 2:
        return AnchoredDiscoveryResult(
            "monster_consensus_not_found",
            None,
            evidence,
            "Fewer than two known active monster actors passed structural checks.",
        )
    if actor_layout is None:
        return AnchoredDiscoveryResult(
            "actor_layout_inconclusive",
            None,
            evidence,
            "Known monster hits did not produce one consensus actor layout.",
        )
    world_hypotheses = _infer_worlds(
        anchors,
        memory=memory,
        configured_world_offset=configured_world_field_offset,
        module_start=module_base,
        module_stop=module_stop,
        slot_refs=slot_refs,
        readable_contains=readable_contains,
        evidence=evidence,
    )
    if not world_hypotheses:
        return AnchoredDiscoveryResult(
            "actor_layout_inconclusive",
            None,
            evidence,
            "Monster consensus did not produce a structural world hypothesis.",
        )

    player_candidates: list[
        tuple[
            tuple[int, int, int, float, int, int, int],
            AnchoredPointerCandidate,
            _WorldHypothesis,
        ]
    ] = []
    expected_x = float(hints.player_spawn_x)
    expected_z = float(hints.player_spawn_z)
    current_hp = int(hints.player_current_hp)
    maximum_hp = int(hints.player_max_hp)
    for x_address in sorted(spawn_matches):
        check()
        base = int(x_address) - actor_layout.x_offset
        if not _contains(readable_contains, base, object_span):
            continue
        try:
            data = memory.read(base, object_span)
            x = _f32(data, actor_layout.x_offset)
            y = _f32(data, actor_layout.y_offset)
            z = _f32(data, actor_layout.z_offset)
        except Exception:
            continue
        evidence.spawn_structure_candidates += 1
        if (
            not all(math.isfinite(value) for value in (x, y, z))
            or abs(x - expected_x) > hints.spawn_tolerance_native
            or abs(z - expected_z) > hints.spawn_tolerance_native
            or abs(y) > coordinate_limit
        ):
            continue
        matching_worlds = tuple(
            world
            for world in world_hypotheses
            if _u32(data, world.field_offset) == world.world_base
        )
        if not matching_worlds:
            continue
        evidence.spawn_world_matches += 1
        evidence.spawn_world_hypothesis_matches += len(matching_worlds)
        matching_self_offsets = tuple(
            offset
            for offset in actor_layout.self_offsets
            if _u32(data, offset) == base
        )
        if not matching_self_offsets:
            continue
        selected_self_offset = matching_self_offsets[0]
        hp_offsets = _aligned_value_offsets(data, current_hp)
        max_hp_offsets = _aligned_value_offsets(data, maximum_hp)
        if not hp_offsets or not max_hp_offsets:
            continue
        evidence.spawn_hp_matches += 1
        selected_hp_offset = (
            actor_layout.hp_offset
            if actor_layout.hp_offset in hp_offsets
            else hp_offsets[0]
        )
        if current_hp == maximum_hp:
            # At full health both fields contain the same value. Prefer a
            # distinct second occurrence so a later damage sample can still
            # prove which field remains the maximum-HP field.
            distinct_max_offsets = tuple(
                offset for offset in max_hp_offsets if offset != selected_hp_offset
            )
            selected_max_offset = (
                distinct_max_offsets[0]
                if distinct_max_offsets
                else selected_hp_offset
            )
        else:
            selected_max_offset = max_hp_offsets[0]
        try:
            species = _i32(data, actor_layout.species_offset)
            active = _i32(data, actor_layout.active_species_offset)
        except Exception:
            species = 0
            active = 0
        player_like = not (species > 0 and active == species)
        if not player_like:
            continue
        evidence.spawn_player_matches += 1

        for world in matching_worlds:
            stable = True
            for sample in range(max(3, int(stability_samples))):
                check()
                try:
                    sample_data = memory.read(base, object_span)
                    sample_x = _f32(sample_data, actor_layout.x_offset)
                    sample_z = _f32(sample_data, actor_layout.z_offset)
                    sample_hp = _i32(sample_data, selected_hp_offset)
                    sample_world = _u32(sample_data, world.field_offset)
                    sample_self = _u32(sample_data, selected_self_offset)
                except Exception:
                    stable = False
                    break
                if (
                    abs(sample_x - x) > 0.05
                    or abs(sample_z - z) > 0.05
                    or sample_hp != current_hp
                    or sample_world != world.world_base
                    or sample_self != base
                ):
                    stable = False
                    break
                if sample + 1 < max(3, int(stability_samples)):
                    sleep(max(0.0, float(stability_delay_seconds)))
            if not stable:
                continue
            evidence.stable_spawn_candidates += 1

            world_ref = _resolve_module_reference(
                memory,
                world.world_base,
                configured_slot=configured_world_slot,
                slot_refs=slot_refs,
                maximum_address=maximum_address,
                chunk_size=chunk_size,
                cancellation=cancellation,
                deadline=deadline,
                check=check,
            )
            evidence.direct_world_slot_candidates += world_ref.direct_candidates
            evidence.world_chain_candidates += world_ref.chained_candidates
            evidence.anchor_ambiguities += world_ref.ambiguous_candidates
            if not world_ref.resolved:
                continue
            assert world_ref.slot is not None
            world_slot = world_ref.slot
            world_chain = world_ref.chain_offsets
            player_ref = _resolve_module_reference(
                memory,
                base,
                configured_slot=configured_player_slot,
                slot_refs=slot_refs,
                maximum_address=maximum_address,
                chunk_size=chunk_size,
                cancellation=cancellation,
                deadline=deadline,
                check=check,
                preferred_roots=((world.world_base, world_slot, world_chain),),
                maximum_preferred_offset=0x4000,
            )
            evidence.direct_player_slot_candidates += player_ref.direct_candidates
            evidence.player_chain_candidates += player_ref.chained_candidates
            evidence.player_reference_matches += player_ref.reference_matches
            evidence.player_world_chain_candidates += (
                player_ref.preferred_chain_candidates
            )
            evidence.player_reference_ambiguities += (
                player_ref.ambiguous_candidates
            )
            evidence.anchor_ambiguities += player_ref.ambiguous_candidates
            if not player_ref.resolved:
                continue
            assert player_ref.slot is not None
            player_slot = player_ref.slot
            player_chain = player_ref.chain_offsets
            candidate = AnchoredPointerCandidate(
                player_base=base,
                world_base=world.world_base,
                player_pointer_address=player_slot,
                player_pointer_chain_offsets=player_chain,
                world_pointer_address=world_slot,
                world_pointer_chain_offsets=world_chain,
                world_field_offset=world.field_offset,
                world_vtable=world.identity_value,
                world_vtable_field_offset=world.identity_field_offset,
                world_identity_kind=world.identity_kind,
                self_pointer_offset=selected_self_offset,
                hp_offset=selected_hp_offset,
                monster_hp_offset=actor_layout.hp_offset,
                max_hp_offset=selected_max_offset,
                species_offset=actor_layout.species_offset,
                active_species_offset=actor_layout.active_species_offset,
                x_offset=actor_layout.x_offset,
                y_offset=actor_layout.y_offset,
                z_offset=actor_layout.z_offset,
                baseline_x=x,
                baseline_y=y,
                baseline_z=z,
                baseline_hp=current_hp,
                maximum_hp=maximum_hp,
                monster_actor_support=world.actor_support,
                monster_species_support=world.species_support,
            )
            score = (
                int(not player_chain),
                int(not world_chain),
                int(selected_hp_offset == actor_layout.hp_offset),
                -math.hypot(x - expected_x, z - expected_z),
                world.actor_support,
                world.species_support,
                min(world.module_references, 8),
            )
            player_candidates.append((score, candidate, world))

    if not player_candidates:
        return AnchoredDiscoveryResult(
            "spawn_player_not_found",
            None,
            evidence,
            "No stable spawn/HP player candidate matched any structural world hypothesis.",
        )
    player_candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, candidate, selected_world = player_candidates[0]
    tied = tuple(
        (item, world)
        for score, item, world in player_candidates
        if score == best_score
    )
    distinct_targets = {
        (
            item.player_base,
            item.world_base,
            item.player_pointer_address,
            item.player_pointer_chain_offsets,
            item.world_pointer_address,
            item.world_pointer_chain_offsets,
        )
        for item, _world in tied
    }
    if len(distinct_targets) > 1:
        evidence.anchor_ambiguities += len(distinct_targets)
        return AnchoredDiscoveryResult(
            "anchor_ambiguous",
            None,
            evidence,
            "Multiple equally supported spawn/HP player candidates remain.",
        )
    _record_selected_world(evidence, selected_world)
    evidence.inferred_self_offset = candidate.self_pointer_offset
    evidence.inferred_player_hp_offset = candidate.hp_offset
    evidence.inferred_player_max_hp_offset = candidate.max_hp_offset
    return AnchoredDiscoveryResult(
        "movement_required",
        candidate,
        evidence,
        "Anchored player candidate is stable; controlled movement is required.",
    )


def confirm_anchored_movement(
    memory: AnchoredDiscoveryMemory,
    candidate: AnchoredPointerCandidate,
    hints: PointerRecoveryHints,
    *,
    check: Callable[[], None],
    samples: int = 3,
    delay_seconds: float = 0.05,
) -> AnchoredDiscoveryResult:
    evidence = AnchoredDiscoveryEvidence(movement_checks=1)
    try:
        player = _resolve_pointer(
            memory,
            candidate.player_pointer_address,
            candidate.player_pointer_chain_offsets,
        )
        world = _resolve_pointer(
            memory,
            candidate.world_pointer_address,
            candidate.world_pointer_chain_offsets,
        )
    except Exception:
        return AnchoredDiscoveryResult(
            "movement_candidate_stale",
            None,
            evidence,
            "Pending pointer slot/chain is no longer readable.",
        )
    if player != candidate.player_base or world != candidate.world_base:
        return AnchoredDiscoveryResult(
            "movement_candidate_stale",
            None,
            evidence,
            "Pending pointer slot/chain changed before movement confirmation.",
        )
    try:
        if (
            int(
                struct.unpack(
                    "<I",
                    memory.read(world + candidate.world_vtable_field_offset, 4),
                )[0]
            )
            != candidate.world_vtable
        ):
            raise ValueError("world vtable changed")
    except Exception:
        return AnchoredDiscoveryResult(
            "movement_candidate_stale",
            None,
            evidence,
            "Pending world object identity changed before movement confirmation.",
        )

    readings: list[tuple[float, float, float]] = []
    for index in range(max(3, int(samples))):
        check()
        try:
            x = float(
                struct.unpack("<f", memory.read(player + candidate.x_offset, 4))[0]
            )
            y = float(
                struct.unpack("<f", memory.read(player + candidate.y_offset, 4))[0]
            )
            z = float(
                struct.unpack("<f", memory.read(player + candidate.z_offset, 4))[0]
            )
            hp = int(
                struct.unpack(
                    "<i",
                    memory.read(player + candidate.hp_offset, 4),
                )[0]
            )
            maximum_hp = int(
                struct.unpack(
                    "<i",
                    memory.read(player + candidate.max_hp_offset, 4),
                )[0]
            )
            actor_world = int(
                struct.unpack(
                    "<I",
                    memory.read(player + candidate.world_field_offset, 4),
                )[0]
            )
            actor_self = int(
                struct.unpack(
                    "<I",
                    memory.read(player + candidate.self_pointer_offset, 4),
                )[0]
            )
        except Exception:
            return AnchoredDiscoveryResult(
                "movement_candidate_stale",
                None,
                evidence,
                "Pending player fields became unreadable.",
            )
        if (
            not all(math.isfinite(value) for value in (x, y, z))
            or hp != hints.player_current_hp
            or maximum_hp != hints.player_max_hp
            or actor_world != world
            or actor_self != player
        ):
            return AnchoredDiscoveryResult(
                "movement_candidate_stale",
                None,
                evidence,
                "Pending player failed exact HP/world/self/coordinate validation.",
            )
        readings.append((x, y, z))
        if index + 1 < max(3, int(samples)):
            sleep(max(0.0, float(delay_seconds)))

    final_x, _final_y, final_z = readings[-1]
    movement = math.hypot(
        final_x - candidate.baseline_x,
        final_z - candidate.baseline_z,
    )
    stable = all(
        math.hypot(x - final_x, z - final_z) <= 0.05
        for x, _y, z in readings
    )
    if movement < hints.movement_minimum_native or not stable:
        return AnchoredDiscoveryResult(
            "movement_not_observed",
            candidate,
            evidence,
            (
                f"Observed movement {movement:.3f} native units; move at least "
                f"{hints.movement_minimum_native:.3f} and stop before confirming."
            ),
        )
    evidence.movement_observed = 1
    return AnchoredDiscoveryResult(
        "movement_confirmed",
        candidate,
        evidence,
        f"Observed coherent movement of {movement:.3f} native units.",
    )
