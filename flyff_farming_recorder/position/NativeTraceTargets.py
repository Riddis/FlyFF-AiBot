from __future__ import annotations

import math
import struct
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from time import sleep
from typing import Protocol

from .AnchoredPointerDiscovery import (
    AnchoredDiscoveryEvidence,
    AnchoredDiscoveryResult,
    PointerRecoveryHints,
    _aligned_pointer_offsets,
    _aligned_value_offsets,
    _f32,
    _i32,
    _monster_anchors,
    _public_monster_cohort,
    _scan_anchor_values,
    _u32,
)
from .PointerScanWorkflow import ReadableRegionIndex
from .Win32ProcessMemory import ModuleInfo


class TraceTargetMemory(Protocol):
    def read(self, address: int, size: int) -> bytes: ...


@dataclass(frozen=True, slots=True)
class TraceMonsterTarget:
    base: int
    species: int
    hp: int
    x: float
    y: float
    z: float
    self_pointer_offsets: tuple[int, ...]
    species_offset: int
    active_species_offset: int
    hp_offset: int
    x_offset: int
    y_offset: int
    z_offset: int


@dataclass(frozen=True, slots=True)
class TracePlayerTarget:
    base: int
    hp: int
    x: float
    y: float
    z: float
    self_pointer_offsets: tuple[int, ...]
    direct_module_slots: tuple[int, ...]
    hp_offset: int
    species_offset: int
    active_species_offset: int
    x_offset: int
    y_offset: int
    z_offset: int


@dataclass(frozen=True, slots=True)
class TraceTargetEvidence:
    bytes_scanned: int
    regions_scanned: int
    read_failures: int
    species_hits: int
    spawn_x_hits: int
    monster_candidates: int
    player_candidates: int
    monster_hp_rejections: int
    monster_coordinate_rejections: int
    player_hp_rejections: int
    player_coordinate_rejections: int
    observed_hp_values: tuple[tuple[int, int], ...]
    monster_base_hypotheses: int
    monster_layout_ties: int
    monster_self_aliases: int
    player_self_rejections: int
    inferred_species_offset: int | None
    inferred_active_species_offset: int | None
    inferred_monster_hp_offset: int | None
    inferred_x_offset: int | None
    inferred_y_offset: int | None
    inferred_z_offset: int | None
    selected_player_hp_offset: int | None


@dataclass(frozen=True, slots=True)
class TraceTargetDiscovery:
    player: TracePlayerTarget | None
    monsters: tuple[TraceMonsterTarget, ...]
    evidence: TraceTargetEvidence
    outcome: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _module_slots_for_value(
    memory: TraceTargetMemory,
    module: ModuleInfo,
    value: int,
    *,
    chunk_size: int,
) -> tuple[int, ...]:
    needle = struct.pack("<I", int(value))
    result: list[int] = []
    offset = 0
    carry = b""
    while offset < module.size:
        amount = min(chunk_size, module.size - offset)
        try:
            data = memory.read(module.base_address + offset, amount)
        except Exception:
            carry = b""
            offset += amount
            continue
        haystack = carry + data
        haystack_base = module.base_address + offset - len(carry)
        start = 0
        while True:
            found = haystack.find(needle, start)
            if found < 0:
                break
            address = haystack_base + found
            if address % 4 == 0:
                result.append(address)
            start = found + 1
        carry = haystack[-3:] if len(haystack) >= 3 else haystack
        offset += amount
    return tuple(sorted(set(result)))


def _trace_evidence(
    anchored: AnchoredDiscoveryEvidence,
    *,
    player_candidates: int,
    player_hp_rejections: int,
    player_coordinate_rejections: int,
    player_self_rejections: int,
    selected_player_hp_offset: int | None,
) -> TraceTargetEvidence:
    return TraceTargetEvidence(
        bytes_scanned=anchored.anchor_bytes_scanned,
        regions_scanned=anchored.anchor_regions_scanned,
        read_failures=anchored.anchor_read_failures,
        species_hits=anchored.species_value_matches,
        spawn_x_hits=anchored.spawn_x_matches,
        monster_candidates=anchored.monster_candidates,
        player_candidates=player_candidates,
        monster_hp_rejections=anchored.monster_hp_rejections,
        monster_coordinate_rejections=anchored.monster_coordinate_rejections,
        player_hp_rejections=player_hp_rejections,
        player_coordinate_rejections=player_coordinate_rejections,
        # The old tracer sampled one assumed HP field and reported misleading
        # values. Exact HP is now searched across each validated object, so no
        # fixed-offset histogram is emitted.
        observed_hp_values=(),
        monster_base_hypotheses=anchored.monster_base_hypotheses,
        monster_layout_ties=anchored.monster_layout_ties,
        monster_self_aliases=anchored.monster_self_field_aliases,
        player_self_rejections=player_self_rejections,
        inferred_species_offset=anchored.inferred_species_offset,
        inferred_active_species_offset=anchored.inferred_active_species_offset,
        inferred_monster_hp_offset=anchored.inferred_hp_offset,
        inferred_x_offset=anchored.inferred_x_offset,
        inferred_y_offset=anchored.inferred_y_offset,
        inferred_z_offset=anchored.inferred_z_offset,
        selected_player_hp_offset=selected_player_hp_offset,
    )


def trace_discovery_from_anchored(
    result: AnchoredDiscoveryResult,
) -> TraceTargetDiscovery:
    """Convert production anchored evidence into the independent reader contract.

    The anchored recovery scanner already proved the actor layout, a stable
    spawn/HP player, direct module aliases, and a coherent monster cohort.
    This adapter lets production use that proof without repeating the tester's
    scan or requiring a world pointer that the current client may not expose.
    """

    players = result.unlinked_players
    monsters = result.monster_cohort
    selected = players[0] if len(players) == 1 else None
    evidence = _trace_evidence(
        result.evidence,
        player_candidates=len(players),
        player_hp_rejections=0,
        player_coordinate_rejections=0,
        player_self_rejections=0,
        selected_player_hp_offset=(None if selected is None else selected.hp_offset),
    )
    if selected is None:
        return TraceTargetDiscovery(
            None,
            (),
            evidence,
            "player_ambiguous",
            "Independent recovery requires exactly one stable player candidate.",
        )
    if not selected.direct_module_slots:
        return TraceTargetDiscovery(
            None,
            (),
            evidence,
            "player_alias_not_found",
            "The stable player has no direct module alias.",
        )
    if len(monsters) < 2:
        return TraceTargetDiscovery(
            None,
            (),
            evidence,
            "monster_consensus_not_found",
            "Independent recovery requires at least two validated monster anchors.",
        )

    player = TracePlayerTarget(
        base=selected.player_base,
        hp=selected.current_hp,
        x=selected.x,
        y=selected.y,
        z=selected.z,
        self_pointer_offsets=(selected.self_pointer_offset,),
        direct_module_slots=selected.direct_module_slots,
        hp_offset=selected.hp_offset,
        species_offset=selected.species_offset,
        active_species_offset=selected.active_species_offset,
        x_offset=selected.x_offset,
        y_offset=selected.y_offset,
        z_offset=selected.z_offset,
    )
    converted_monsters = tuple(
        TraceMonsterTarget(
            base=item.base,
            species=item.species,
            hp=item.current_hp,
            x=item.x,
            y=item.y,
            z=item.z,
            self_pointer_offsets=item.self_pointer_offsets,
            species_offset=item.species_offset,
            active_species_offset=item.active_species_offset,
            hp_offset=item.hp_offset,
            x_offset=item.x_offset,
            y_offset=item.y_offset,
            z_offset=item.z_offset,
        )
        for item in sorted(
            monsters,
            key=lambda item: (
                math.hypot(item.x - selected.x, item.z - selected.z),
                item.base,
            ),
        )
    )
    return TraceTargetDiscovery(
        player,
        converted_monsters,
        evidence,
        "success",
        (
            f"Reused anchored proof for one player, {len(converted_monsters)} "
            f"monster anchors, and {len(player.direct_module_slots)} direct "
            "module aliases without a world pointer."
        ),
    )


def discover_trace_targets(
    memory: TraceTargetMemory,
    *,
    regions: tuple[object, ...],
    readable: ReadableRegionIndex,
    module: ModuleInfo,
    species_hp: Mapping[int, int],
    spawn_x: float,
    spawn_z: float,
    player_hp: int,
    species_offset: int,
    active_species_offset: int,
    hp_offset: int,
    x_offset: int,
    y_offset: int,
    z_offset: int,
    self_pointer_offset: int,
    coordinate_limit: float,
    spawn_tolerance: float = 2.0,
    object_span: int = 0x4000,
    chunk_size: int = 1 << 20,
    maximum_scan_bytes: int = 0x60000000,
    maximum_monster_candidates: int = 512,
    stability_samples: int = 3,
    stability_delay_seconds: float = 0.03,
    check: Callable[[], None] | None = None,
    progress: Callable[[int, int, Mapping[str, int]], None] | None = None,
) -> TraceTargetDiscovery:
    """Find trace targets with the same dynamic layout proof as recovery.

    Exact player and monster HP values remain hard acceptance requirements.
    Their field offsets are discovered inside structurally validated objects;
    the configured offsets are only starting-layout hints and tie-breakers.
    """

    if not species_hp:
        raise ValueError("at least one exact species/full-HP pair is required")
    if player_hp <= 0:
        raise ValueError("player_hp must be positive")
    if object_span % 4:
        raise ValueError("object_span must be divisible by four")
    object_span = max(
        int(object_span),
        int(self_pointer_offset) + 0x204,
        int(active_species_offset) + 4,
        int(hp_offset) + 4,
        int(z_offset) + 4,
        0x4004,
    )
    check = check or (lambda: None)
    hints = PointerRecoveryHints(
        known_species_ids=tuple(species_hp),
        player_spawn_x=float(spawn_x),
        player_spawn_z=float(spawn_z),
        player_current_hp=int(player_hp),
        player_max_hp=int(player_hp),
        monster_hp_by_species=tuple(species_hp.items()),
        require_exact_monster_hp=True,
        spawn_tolerance_native=float(spawn_tolerance),
    )
    anchored = AnchoredDiscoveryEvidence()

    def anchored_progress(evidence: AnchoredDiscoveryEvidence) -> None:
        if progress is None:
            return
        progress(
            evidence.anchor_bytes_scanned,
            evidence.anchor_regions_scanned,
            {
                "species_hits": evidence.species_value_matches,
                "spawn_x": evidence.spawn_x_matches,
            },
        )

    species_matches, spawn_matches = _scan_anchor_values(
        memory,  # type: ignore[arg-type]
        regions,
        hints,
        chunk_size=chunk_size,
        maximum_bytes=maximum_scan_bytes,
        check=check,
        evidence=anchored,
        progress_callback=anchored_progress,
    )
    anchors, layout = _monster_anchors(
        memory,  # type: ignore[arg-type]
        species_matches,
        species_offset=species_offset,
        active_species_offset=active_species_offset,
        hp_offset=hp_offset,
        x_offset=x_offset,
        y_offset=y_offset,
        z_offset=z_offset,
        object_span=object_span,
        coordinate_limit=coordinate_limit,
        readable_contains=readable.contains,
        maximum_candidates=maximum_monster_candidates,
        check=check,
        evidence=anchored,
        monster_hp_by_species=species_hp,
        require_exact_monster_hp=True,
    )

    if len(anchors) < 2 or layout is None:
        evidence = _trace_evidence(
            anchored,
            player_candidates=0,
            player_hp_rejections=0,
            player_coordinate_rejections=0,
            player_self_rejections=0,
            selected_player_hp_offset=None,
        )
        outcome = (
            "monster_consensus_not_found"
            if len(anchors) < 2
            else "actor_layout_inconclusive"
        )
        message = (
            "Fewer than two exact species/full-HP monsters passed dynamic "
            "structural validation."
            if len(anchors) < 2
            else "Exact monster anchors did not produce one consensus actor layout."
        )
        return TraceTargetDiscovery(None, (), evidence, outcome, message)

    monster_observations = _public_monster_cohort(anchors, layout)
    monsters = tuple(
        TraceMonsterTarget(
            base=item.base,
            species=item.species,
            hp=item.current_hp,
            x=item.x,
            y=item.y,
            z=item.z,
            self_pointer_offsets=item.self_pointer_offsets,
            species_offset=item.species_offset,
            active_species_offset=item.active_species_offset,
            hp_offset=item.hp_offset,
            x_offset=item.x_offset,
            y_offset=item.y_offset,
            z_offset=item.z_offset,
        )
        for item in monster_observations
    )

    player_candidates: list[TracePlayerTarget] = []
    player_hp_rejections = 0
    player_coordinate_rejections = 0
    player_self_rejections = 0
    expected_x = float(spawn_x)
    expected_z = float(spawn_z)
    exact_monster_bases = {int(item.base) for item in anchors}
    for x_address in sorted(spawn_matches):
        check()
        base = int(x_address) - int(layout.x_offset)
        if base <= 0x10000 or not readable.contains(base, object_span):
            continue
        try:
            data = memory.read(base, object_span)
            x = _f32(data, layout.x_offset)
            y = _f32(data, layout.y_offset)
            z = _f32(data, layout.z_offset)
        except Exception:
            continue
        if (
            not all(math.isfinite(value) for value in (x, y, z))
            or abs(x - expected_x) > spawn_tolerance
            or abs(z - expected_z) > spawn_tolerance
            or abs(y) > coordinate_limit
        ):
            player_coordinate_rejections += 1
            continue
        # Player and monster classes do not have to expose self aliases at
        # the same offsets. Search the complete player object for exact base
        # references, then rank known monster/configured locations first.
        discovered_self_offsets = _aligned_pointer_offsets(data, base)
        if not discovered_self_offsets:
            player_self_rejections += 1
            continue
        selected_self_offset = min(
            discovered_self_offsets,
            key=lambda offset: (
                0 if offset in layout.self_offsets else 1,
                abs(offset - int(self_pointer_offset)),
                offset,
            ),
        )
        self_offsets = (
            selected_self_offset,
            *(
                offset
                for offset in discovered_self_offsets
                if offset != selected_self_offset
            ),
        )
        hp_offsets = _aligned_value_offsets(data, int(player_hp))
        if not hp_offsets:
            player_hp_rejections += 1
            continue
        selected_hp_offset = min(
            hp_offsets,
            key=lambda offset: (abs(offset - layout.hp_offset), offset),
        )
        # The instantiated-species field is a monster-layout field. It must
        # never be used as a mandatory player discriminator: the player class
        # may contain unrelated data at the same offset, and a newly recovered
        # monster field can otherwise reject the real player. Exact validated
        # monster anchors are already known, so exclude those bases directly.
        if base in exact_monster_bases:
            continue

        stable = True
        for sample in range(max(3, int(stability_samples))):
            check()
            try:
                current = memory.read(base, object_span)
                if (
                    _u32(current, self_offsets[0]) != base
                    or _i32(current, selected_hp_offset) != int(player_hp)
                    or abs(_f32(current, layout.x_offset) - x) > 0.05
                    or abs(_f32(current, layout.z_offset) - z) > 0.05
                ):
                    stable = False
                    break
            except Exception:
                stable = False
                break
            if sample + 1 < max(3, int(stability_samples)):
                sleep(max(0.0, float(stability_delay_seconds)))
        if not stable:
            continue

        aliases = _module_slots_for_value(
            memory,
            module,
            base,
            chunk_size=chunk_size,
        )
        player_candidates.append(
            TracePlayerTarget(
                base=base,
                hp=int(player_hp),
                x=x,
                y=y,
                z=z,
                self_pointer_offsets=self_offsets,
                direct_module_slots=aliases,
                hp_offset=selected_hp_offset,
                species_offset=layout.species_offset,
                active_species_offset=layout.active_species_offset,
                x_offset=layout.x_offset,
                y_offset=layout.y_offset,
                z_offset=layout.z_offset,
            )
        )

    player_candidates.sort(
        key=lambda item: (
            bool(item.direct_module_slots),
            len(item.direct_module_slots),
            len(item.self_pointer_offsets),
            -math.hypot(item.x - spawn_x, item.z - spawn_z),
        ),
        reverse=True,
    )
    player = player_candidates[0] if player_candidates else None
    evidence = _trace_evidence(
        anchored,
        player_candidates=len(player_candidates),
        player_hp_rejections=player_hp_rejections,
        player_coordinate_rejections=player_coordinate_rejections,
        player_self_rejections=player_self_rejections,
        selected_player_hp_offset=(None if player is None else player.hp_offset),
    )
    if player is None:
        return TraceTargetDiscovery(
            None,
            monsters,
            evidence,
            "player_not_found",
            "No stable spawn/self/exact-HP player object passed dynamic layout validation.",
        )
    return TraceTargetDiscovery(
        player,
        tuple(
            sorted(
                monsters,
                key=lambda item: (
                    math.hypot(item.x - player.x, item.z - player.z),
                    item.base,
                ),
            )
        ),
        evidence,
        "success",
        (
            f"Found one selected player, {len(monsters)} exact full-health "
            f"monster actors, and {len(player.direct_module_slots)} direct "
            "module aliases with dynamically inferred field offsets."
        ),
    )
