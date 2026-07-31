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
    monster_species_rejections: int = 0
    monster_active_rejections: int = 0
    monster_hp_rejections: int = 0
    monster_coordinate_rejections: int = 0
    inferred_world_actor_support: int = 0
    inferred_world_species_support: int = 0
    inferred_world_offset: int | None = None
    inferred_self_actor_support: int = 0
    inferred_self_offset: int | None = None
    spawn_structure_candidates: int = 0
    spawn_world_matches: int = 0
    spawn_hp_matches: int = 0
    spawn_player_matches: int = 0
    stable_spawn_candidates: int = 0
    direct_player_slot_candidates: int = 0
    player_chain_candidates: int = 0
    direct_world_slot_candidates: int = 0
    world_chain_candidates: int = 0
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
    self_pointer_offset: int
    hp_offset: int
    max_hp_offset: int
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


def _bounds(region: object) -> tuple[int, int]:
    base = int(getattr(region, "base_address"))
    size = int(getattr(region, "size"))
    return base, base + max(0, size)


def _private_regions(regions: Iterable[object]) -> tuple[object, ...]:
    return tuple(
        region
        for region in regions
        if int(getattr(region, "region_type", MEM_PRIVATE)) == MEM_PRIVATE
        and _bounds(region)[1] > _bounds(region)[0]
    )


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
) -> tuple[_MonsterAnchor, ...]:
    anchors: list[_MonsterAnchor] = []
    seen: set[int] = set()
    for expected_species, matches in species_matches.items():
        for address in sorted(matches):
            check()
            if len(seen) >= maximum_candidates:
                break
            base = int(address) - int(species_offset)
            if base in seen or not _contains(readable_contains, base, object_span):
                continue
            seen.add(base)
            try:
                data = memory.read(base, object_span)
                species = _i32(data, species_offset)
                active = _i32(data, active_species_offset)
                hp = _i32(data, hp_offset)
                coordinates = (
                    _f32(data, x_offset),
                    _f32(data, y_offset),
                    _f32(data, z_offset),
                )
            except Exception:
                evidence.monster_species_rejections += 1
                continue
            if species != expected_species:
                evidence.monster_species_rejections += 1
                continue
            if active != species:
                evidence.monster_active_rejections += 1
                continue
            if hp <= 0:
                evidence.monster_hp_rejections += 1
                continue
            if not all(math.isfinite(value) for value in coordinates) or any(
                abs(value) > coordinate_limit for value in coordinates
            ):
                evidence.monster_coordinate_rejections += 1
                continue
            anchors.append(_MonsterAnchor(base, species, data))
    evidence.monster_candidates = len(anchors)
    return tuple(anchors)


def _infer_self_offset(
    anchors: tuple[_MonsterAnchor, ...],
    *,
    evidence: AnchoredDiscoveryEvidence,
) -> int | None:
    support: dict[int, set[int]] = defaultdict(set)
    for actor in anchors:
        needle = struct.pack("<I", actor.base)
        for absolute in _find_aligned(actor.data, needle, actor.base):
            support[absolute - actor.base].add(actor.base)
    if not support:
        return None
    ordered = sorted(
        support.items(),
        key=lambda item: (len(item[1]), -item[0]),
        reverse=True,
    )
    offset, bases = ordered[0]
    minimum = max(2, math.ceil(len(anchors) * 0.6))
    if len(bases) < minimum:
        return None
    evidence.inferred_self_offset = offset
    evidence.inferred_self_actor_support = len(bases)
    return offset


def _infer_world(
    anchors: tuple[_MonsterAnchor, ...],
    *,
    configured_world_offset: int,
    module_start: int,
    module_stop: int,
    slot_refs: Mapping[int, list[int]],
    readable_contains: Callable[[int, int], bool],
    evidence: AnchoredDiscoveryEvidence,
) -> tuple[int, int] | None:
    support: dict[tuple[int, int], set[int]] = defaultdict(set)
    species_support: dict[tuple[int, int], set[int]] = defaultdict(set)
    for actor in anchors:
        scan_stop = min(len(actor.data), 0x500)
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

    ranked: list[tuple[tuple[int, int, int, int], int, int]] = []
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
        ranked.append((score, offset, value))
    ranked.sort(reverse=True)
    if not ranked:
        return None
    best_score, offset, world = ranked[0]
    tied = [item for item in ranked[1:] if item[0] == best_score]
    if tied:
        evidence.anchor_ambiguities += len(tied) + 1
        return None
    evidence.inferred_world_actor_support = best_score[0]
    evidence.inferred_world_species_support = best_score[1]
    evidence.inferred_world_offset = offset
    return offset, world


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
) -> tuple[int, tuple[int, ...], int, int] | None:
    direct = tuple(sorted(set(slot_refs.get(value, ()))))
    if configured_slot in direct:
        return configured_slot, (), len(direct), 0
    if len(direct) == 1:
        return direct[0], (), 1, 0
    if len(direct) > 1:
        return None

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
        return None
    chains: set[tuple[int, int]] = set()
    for reference in references[:4096]:
        check()
        for offset in range(0, 0x104, 4):
            holder = int(reference) - offset
            for slot in slot_refs.get(holder, ()):
                chains.add((slot, offset))
    configured_chains = [item for item in chains if item[0] == configured_slot]
    if len(configured_chains) == 1:
        slot, offset = configured_chains[0]
        return slot, (offset,), 0, len(chains)
    if len(chains) != 1:
        return None
    slot, offset = next(iter(chains))
    return slot, (offset,), 0, 1


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
    maximum_scan_bytes: int = 0x40000000,
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
        0x500,
    ) + 4
    anchors = _monster_anchors(
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
    self_offset = _infer_self_offset(anchors, evidence=evidence)
    inferred_world = _infer_world(
        anchors,
        configured_world_offset=configured_world_field_offset,
        module_start=module_base,
        module_stop=module_stop,
        slot_refs=slot_refs,
        readable_contains=readable_contains,
        evidence=evidence,
    )
    if self_offset is None or inferred_world is None:
        return AnchoredDiscoveryResult(
            "actor_layout_inconclusive",
            None,
            evidence,
            "Monster consensus did not produce unique self/world fields.",
        )
    world_field_offset, world_base = inferred_world

    player_candidates: list[
        tuple[tuple[int, int, int, float], AnchoredPointerCandidate]
    ] = []
    expected_x = float(hints.player_spawn_x)
    expected_z = float(hints.player_spawn_z)
    current_hp = int(hints.player_current_hp)
    maximum_hp = int(hints.player_max_hp)
    for x_address in sorted(spawn_matches):
        check()
        base = int(x_address) - x_offset
        if not _contains(readable_contains, base, object_span):
            continue
        try:
            data = memory.read(base, object_span)
            x = _f32(data, x_offset)
            y = _f32(data, y_offset)
            z = _f32(data, z_offset)
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
        try:
            if _u32(data, world_field_offset) != world_base:
                continue
        except Exception:
            continue
        evidence.spawn_world_matches += 1
        if _u32(data, self_offset) != base:
            continue
        hp_offsets = _aligned_value_offsets(data, current_hp)
        max_hp_offsets = _aligned_value_offsets(data, maximum_hp)
        if not hp_offsets or not max_hp_offsets:
            continue
        evidence.spawn_hp_matches += 1
        selected_hp_offset = hp_offset if hp_offset in hp_offsets else hp_offsets[0]
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
            species = _i32(data, species_offset)
            active = _i32(data, active_species_offset)
        except Exception:
            species = 0
            active = 0
        player_like = not (species > 0 and active == species)
        if not player_like:
            continue
        evidence.spawn_player_matches += 1

        stable = True
        for sample in range(max(3, int(stability_samples))):
            check()
            try:
                sample_data = memory.read(base, object_span)
                sample_x = _f32(sample_data, x_offset)
                sample_z = _f32(sample_data, z_offset)
                sample_hp = _i32(sample_data, selected_hp_offset)
                sample_world = _u32(sample_data, world_field_offset)
                sample_self = _u32(sample_data, self_offset)
            except Exception:
                stable = False
                break
            if (
                abs(sample_x - x) > 0.05
                or abs(sample_z - z) > 0.05
                or sample_hp != current_hp
                or sample_world != world_base
                or sample_self != base
            ):
                stable = False
                break
            if sample + 1 < max(3, int(stability_samples)):
                sleep(max(0.0, float(stability_delay_seconds)))
        if not stable:
            continue
        evidence.stable_spawn_candidates += 1

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
        )
        world_ref = _resolve_module_reference(
            memory,
            world_base,
            configured_slot=configured_world_slot,
            slot_refs=slot_refs,
            maximum_address=maximum_address,
            chunk_size=chunk_size,
            cancellation=cancellation,
            deadline=deadline,
            check=check,
        )
        if player_ref is None or world_ref is None:
            evidence.anchor_ambiguities += 1
            continue
        player_slot, player_chain, direct_player, chained_player = player_ref
        world_slot, world_chain, direct_world, chained_world = world_ref
        evidence.direct_player_slot_candidates += direct_player
        evidence.player_chain_candidates += chained_player
        evidence.direct_world_slot_candidates += direct_world
        evidence.world_chain_candidates += chained_world
        candidate = AnchoredPointerCandidate(
            player_base=base,
            world_base=world_base,
            player_pointer_address=player_slot,
            player_pointer_chain_offsets=player_chain,
            world_pointer_address=world_slot,
            world_pointer_chain_offsets=world_chain,
            world_field_offset=world_field_offset,
            self_pointer_offset=self_offset,
            hp_offset=selected_hp_offset,
            max_hp_offset=selected_max_offset,
            baseline_x=x,
            baseline_y=y,
            baseline_z=z,
            baseline_hp=current_hp,
            maximum_hp=maximum_hp,
            monster_actor_support=evidence.inferred_world_actor_support,
            monster_species_support=evidence.inferred_world_species_support,
        )
        score = (
            int(not player_chain),
            int(not world_chain),
            int(selected_hp_offset == hp_offset),
            -math.hypot(x - expected_x, z - expected_z),
        )
        player_candidates.append((score, candidate))

    if not player_candidates:
        return AnchoredDiscoveryResult(
            "spawn_player_not_found",
            None,
            evidence,
            "No stable spawn/HP player candidate shared the inferred world.",
        )
    player_candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, candidate = player_candidates[0]
    if any(score == best_score for score, _item in player_candidates[1:]):
        evidence.anchor_ambiguities += sum(
            1 for score, _item in player_candidates if score == best_score
        )
        return AnchoredDiscoveryResult(
            "anchor_ambiguous",
            None,
            evidence,
            "Multiple equally supported spawn/HP player candidates remain.",
        )
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
    x_offset: int,
    y_offset: int,
    z_offset: int,
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

    readings: list[tuple[float, float, float]] = []
    for index in range(max(3, int(samples))):
        check()
        try:
            x = float(struct.unpack("<f", memory.read(player + x_offset, 4))[0])
            y = float(struct.unpack("<f", memory.read(player + y_offset, 4))[0])
            z = float(struct.unpack("<f", memory.read(player + z_offset, 4))[0])
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
