from __future__ import annotations

import base64
import json
import math
import os
import struct
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Event, RLock
from time import monotonic, sleep
from typing import Protocol, cast

from .AnchoredPointerDiscovery import (
    AnchoredDiscoveryEvidence,
    AnchoredDiscoveryMemory,
    AnchoredDiscoveryResult,
    AnchoredPointerCandidate,
    PointerRecoveryHints,
    confirm_anchored_movement,
    discover_anchored_pointer_candidate,
)
from .MonsterConfig import (
    DEFAULT_MONSTER_CONFIG_PATH,
    NativeMonsterConfig,
    load_native_monster_config,
)
from .NativeTraceTargets import (
    TraceTargetDiscovery,
    TraceTargetEvidence,
    discover_trace_targets,
    trace_discovery_from_anchored,
)
from .PointerScanWorkflow import ReadableRegionIndex
from .PositionConfig import DEFAULT_POSITION_CONFIG_PATH


class PointerRecoveryMemory(Protocol):
    pid: int

    def read(self, address: int, size: int) -> bytes: ...

    def readable_regions(
        self,
        *,
        maximum_address: int = 0x7FFFFFFF,
        private_only: bool = True,
    ) -> tuple[object, ...]: ...


@dataclass(frozen=True, slots=True)
class PlayerPointerRecovery:
    player_pointer_address: int
    player_pointer_offset: int
    player_base: int
    world_base: int
    world_pointer_address: int | None
    world_pointer_offset: int | None
    configured_player_pointer_offset: int
    configured_world_pointer_offset: int
    search_radius: int
    validated_candidates: int
    strategy: str = "configured_neighborhood"
    player_pointer_chain_offsets: tuple[int, ...] = ()
    world_pointer_chain_offsets: tuple[int, ...] = ()
    world_field_offset: int | None = None
    world_vtable_offset: int | None = None
    world_vtable_field_offset: int | None = None
    world_identity_kind: str | None = None
    self_pointer_offset: int | None = None
    species_offset: int | None = None
    active_species_offset: int | None = None
    hp_offset: int | None = None
    x_offset: int | None = None
    y_offset: int | None = None
    z_offset: int | None = None
    movement_validated: bool = False
    independent_discovery: TraceTargetDiscovery | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    independent_expected_full_hp_by_species: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class PointerRecoveryMetrics:
    """Bounded diagnostics for the latest explicit recovery request."""

    pid: int
    module_base: int
    outcome: str
    elapsed_seconds: float
    region_enumerations: int = 0
    region_count: int = 0
    radii_started: int = 0
    radii_completed: int = 0
    scan_intervals: int = 0
    chunks_read: int = 0
    bytes_scanned: int = 0
    read_failures: int = 0
    aligned_words_examined: int = 0
    pointer_like_words: int = 0
    containment_checks: int = 0
    candidate_targets: int = 0
    candidate_slots: int = 0
    candidates_validated: int = 0
    cache_hit: bool = False
    negative_cache_hit: bool = False
    joined_existing_attempt: bool = False
    cooldown_remaining_seconds: float = 0.0
    deadline_is_cooperative: bool = True
    strategy: str = "unknown"
    module_size: int = 0
    layout_rejections: int = 0
    self_mismatch_rejections: int = 0
    world_null_rejections: int = 0
    coordinate_rejections: int = 0
    hp_rejections: int = 0
    candidate_read_failures: int = 0
    non_player_rejections: int = 0
    missing_world_slot_rejections: int = 0
    unstable_rejections: int = 0
    ambiguous_candidates: int = 0
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
    monster_base_hypotheses: int = 0
    monster_layout_species_support: int = 0
    monster_layout_ties: int = 0
    monster_self_field_aliases: int = 0
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
    player_world_rooted_matches: int = 0
    player_reference_ambiguities: int = 0
    movement_checks: int = 0
    movement_observed: int = 0
    self_mismatch_near_probed: int = 0
    self_mismatch_world_nonzero: int = 0
    self_mismatch_world_module_ref: int = 0
    self_mismatch_coordinate_plausible: int = 0
    self_mismatch_hp_positive: int = 0
    self_mismatch_player_like: int = 0
    self_mismatch_full_near_matches: int = 0


@dataclass(frozen=True, slots=True)
class PointerRecoveryProgress:
    """One status update emitted by an explicit recovery attempt."""

    phase: str
    message: str
    metrics: PointerRecoveryMetrics


PointerRecoveryStatusCallback = Callable[[PointerRecoveryProgress], None]


@dataclass(frozen=True, slots=True)
class _ValidatedCandidate:
    slot_address: int
    target_address: int
    world_base: int
    distance: int
    reference_count: int
    world_pointer_slot: int | None
    pair_matches: bool
    configured_world_matches: bool
    player_like: bool
    world_reference_count: int


@dataclass(slots=True)
class _MetricsBuilder:
    pid: int
    module_base: int
    started_at: float
    region_enumerations: int = 0
    region_count: int = 0
    radii_started: int = 0
    radii_completed: int = 0
    scan_intervals: int = 0
    chunks_read: int = 0
    bytes_scanned: int = 0
    read_failures: int = 0
    aligned_words_examined: int = 0
    pointer_like_words: int = 0
    containment_checks: int = 0
    candidate_targets: int = 0
    candidate_slots: int = 0
    candidates_validated: int = 0
    strategy: str = "unknown"
    module_size: int = 0
    layout_rejections: int = 0
    self_mismatch_rejections: int = 0
    world_null_rejections: int = 0
    coordinate_rejections: int = 0
    hp_rejections: int = 0
    candidate_read_failures: int = 0
    non_player_rejections: int = 0
    missing_world_slot_rejections: int = 0
    unstable_rejections: int = 0
    ambiguous_candidates: int = 0
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
    monster_base_hypotheses: int = 0
    monster_layout_species_support: int = 0
    monster_layout_ties: int = 0
    monster_self_field_aliases: int = 0
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
    player_world_rooted_matches: int = 0
    player_reference_ambiguities: int = 0
    movement_checks: int = 0
    movement_observed: int = 0
    self_mismatch_near_probed: int = 0
    self_mismatch_world_nonzero: int = 0
    self_mismatch_world_module_ref: int = 0
    self_mismatch_coordinate_plausible: int = 0
    self_mismatch_hp_positive: int = 0
    self_mismatch_player_like: int = 0
    self_mismatch_full_near_matches: int = 0

    def freeze(
        self,
        outcome: str,
        *,
        now: float,
        cache_hit: bool = False,
        negative_cache_hit: bool = False,
        joined_existing_attempt: bool = False,
        cooldown_remaining_seconds: float = 0.0,
    ) -> PointerRecoveryMetrics:
        return PointerRecoveryMetrics(
            pid=self.pid,
            module_base=self.module_base,
            outcome=outcome,
            elapsed_seconds=max(0.0, float(now) - self.started_at),
            region_enumerations=self.region_enumerations,
            region_count=self.region_count,
            radii_started=self.radii_started,
            radii_completed=self.radii_completed,
            scan_intervals=self.scan_intervals,
            chunks_read=self.chunks_read,
            bytes_scanned=self.bytes_scanned,
            read_failures=self.read_failures,
            aligned_words_examined=self.aligned_words_examined,
            pointer_like_words=self.pointer_like_words,
            containment_checks=self.containment_checks,
            candidate_targets=self.candidate_targets,
            candidate_slots=self.candidate_slots,
            candidates_validated=self.candidates_validated,
            cache_hit=cache_hit,
            negative_cache_hit=negative_cache_hit,
            joined_existing_attempt=joined_existing_attempt,
            cooldown_remaining_seconds=max(
                0.0,
                float(cooldown_remaining_seconds),
            ),
            strategy=self.strategy,
            module_size=self.module_size,
            layout_rejections=self.layout_rejections,
            self_mismatch_rejections=self.self_mismatch_rejections,
            world_null_rejections=self.world_null_rejections,
            coordinate_rejections=self.coordinate_rejections,
            hp_rejections=self.hp_rejections,
            candidate_read_failures=self.candidate_read_failures,
            non_player_rejections=self.non_player_rejections,
            missing_world_slot_rejections=self.missing_world_slot_rejections,
            unstable_rejections=self.unstable_rejections,
            ambiguous_candidates=self.ambiguous_candidates,
            anchor_bytes_scanned=self.anchor_bytes_scanned,
            anchor_regions_scanned=self.anchor_regions_scanned,
            anchor_read_failures=self.anchor_read_failures,
            species_value_matches=self.species_value_matches,
            spawn_x_matches=self.spawn_x_matches,
            monster_candidates=self.monster_candidates,
            monster_species_rejections=self.monster_species_rejections,
            monster_active_rejections=self.monster_active_rejections,
            monster_hp_rejections=self.monster_hp_rejections,
            monster_coordinate_rejections=self.monster_coordinate_rejections,
            inferred_world_actor_support=self.inferred_world_actor_support,
            inferred_world_species_support=self.inferred_world_species_support,
            inferred_world_offset=self.inferred_world_offset,
            inferred_world_vtable=self.inferred_world_vtable,
            inferred_world_vtable_field_offset=(
                self.inferred_world_vtable_field_offset
            ),
            inferred_world_identity_kind=self.inferred_world_identity_kind,
            inferred_world_readable_pointer_fields=(
                self.inferred_world_readable_pointer_fields
            ),
            inferred_world_distinct_values=self.inferred_world_distinct_values,
            structural_world_hypotheses=self.structural_world_hypotheses,
            world_object_rejections=self.world_object_rejections,
            world_identity_candidates=self.world_identity_candidates,
            world_identity_span_rejections=(
                self.world_identity_span_rejections
            ),
            world_identity_vtable_misses=self.world_identity_vtable_misses,
            world_identity_unstable_rejections=(
                self.world_identity_unstable_rejections
            ),
            world_identity_module_pointer_fields=(
                self.world_identity_module_pointer_fields
            ),
            world_identity_readable_pointer_fields=(
                self.world_identity_readable_pointer_fields
            ),
            world_identity_distinct_values=(
                self.world_identity_distinct_values
            ),
            world_identity_structural_rejections=(
                self.world_identity_structural_rejections
            ),
            world_identity_marker_accepts=self.world_identity_marker_accepts,
            world_near_actor_support=self.world_near_actor_support,
            world_near_species_support=self.world_near_species_support,
            world_near_module_references=self.world_near_module_references,
            world_near_field_offset=self.world_near_field_offset,
            world_near_target=self.world_near_target,
            world_near_module_pointer_fields=(
                self.world_near_module_pointer_fields
            ),
            world_near_readable_pointer_fields=(
                self.world_near_readable_pointer_fields
            ),
            world_near_distinct_values=self.world_near_distinct_values,
            inferred_self_actor_support=self.inferred_self_actor_support,
            inferred_self_offset=self.inferred_self_offset,
            monster_base_hypotheses=self.monster_base_hypotheses,
            monster_layout_species_support=self.monster_layout_species_support,
            monster_layout_ties=self.monster_layout_ties,
            monster_self_field_aliases=self.monster_self_field_aliases,
            inferred_species_offset=self.inferred_species_offset,
            inferred_active_species_offset=self.inferred_active_species_offset,
            inferred_hp_offset=self.inferred_hp_offset,
            inferred_player_hp_offset=self.inferred_player_hp_offset,
            inferred_player_max_hp_offset=self.inferred_player_max_hp_offset,
            inferred_x_offset=self.inferred_x_offset,
            inferred_y_offset=self.inferred_y_offset,
            inferred_z_offset=self.inferred_z_offset,
            spawn_structure_candidates=self.spawn_structure_candidates,
            spawn_world_matches=self.spawn_world_matches,
            spawn_world_hypothesis_matches=(
                self.spawn_world_hypothesis_matches
            ),
            spawn_hp_matches=self.spawn_hp_matches,
            spawn_player_matches=self.spawn_player_matches,
            stable_spawn_candidates=self.stable_spawn_candidates,
            direct_player_slot_candidates=self.direct_player_slot_candidates,
            player_chain_candidates=self.player_chain_candidates,
            direct_world_slot_candidates=self.direct_world_slot_candidates,
            world_chain_candidates=self.world_chain_candidates,
            player_reference_matches=self.player_reference_matches,
            player_world_chain_candidates=self.player_world_chain_candidates,
            player_world_rooted_matches=self.player_world_rooted_matches,
            player_reference_ambiguities=self.player_reference_ambiguities,
            movement_checks=self.movement_checks,
            movement_observed=self.movement_observed,
            self_mismatch_near_probed=self.self_mismatch_near_probed,
            self_mismatch_world_nonzero=self.self_mismatch_world_nonzero,
            self_mismatch_world_module_ref=self.self_mismatch_world_module_ref,
            self_mismatch_coordinate_plausible=(
                self.self_mismatch_coordinate_plausible
            ),
            self_mismatch_hp_positive=self.self_mismatch_hp_positive,
            self_mismatch_player_like=self.self_mismatch_player_like,
            self_mismatch_full_near_matches=(
                self.self_mismatch_full_near_matches
            ),
        )


@dataclass(slots=True)
class _RecoveryFlight:
    completed: Event = field(default_factory=Event)
    result: PlayerPointerRecovery | None = None
    metrics: PointerRecoveryMetrics | None = None
    persist_requested: bool = False


@dataclass(frozen=True, slots=True)
class _RegionIndex:
    """Merged readable intervals with logarithmic containment lookup."""

    starts: tuple[int, ...]
    stops: tuple[int, ...]

    @classmethod
    def build(cls, regions: tuple[object, ...]) -> _RegionIndex:
        bounds = sorted(
            (start, stop)
            for start, stop in (_region_bounds(region) for region in regions)
            if stop > start
        )
        merged: list[list[int]] = []
        for start, stop in bounds:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], stop)
            else:
                merged.append([start, stop])
        return cls(
            starts=tuple(item[0] for item in merged),
            stops=tuple(item[1] for item in merged),
        )

    def contains(self, address: int, size: int = 1) -> bool:
        if not self.starts:
            return False
        start = int(address)
        end = start + max(1, int(size))
        index = bisect_right(self.starts, start) - 1
        return index >= 0 and end <= self.stops[index]

    def intersections(self, start: int, stop: int) -> tuple[tuple[int, int], ...]:
        if stop <= start or not self.starts:
            return ()
        index = max(0, bisect_right(self.starts, int(start)) - 1)
        while index < len(self.starts) and self.stops[index] <= start:
            index += 1
        result: list[tuple[int, int]] = []
        while index < len(self.starts) and self.starts[index] < stop:
            clipped_start = max(int(start), self.starts[index])
            clipped_stop = min(int(stop), self.stops[index])
            if clipped_stop > clipped_start:
                result.append((clipped_start, clipped_stop))
            index += 1
        return tuple(result)


class _AttemptStopped(Exception):
    def __init__(self, outcome: str) -> None:
        super().__init__(outcome)
        self.outcome = outcome


@dataclass(frozen=True, slots=True)
class _AttemptControl:
    cancellation: object | None
    deadline: float
    clock: Callable[[], float]

    def check(self) -> None:
        if _cancellation_requested(self.cancellation):
            raise _AttemptStopped("cancelled")
        if self.clock() >= self.deadline:
            raise _AttemptStopped("deadline")

    def wait(self, seconds: float) -> None:
        wait_until = min(self.deadline, self.clock() + max(0.0, float(seconds)))
        while self.clock() < wait_until:
            self.check()
            sleep(min(0.01, max(0.0, wait_until - self.clock())))
        self.check()


class PointerRecoveryState:
    """Per-attachment cache, cooldown, and single-flight recovery state."""

    def __init__(self) -> None:
        self.lock = RLock()
        self.recovery_cache: dict[
            tuple[int, int],
            PlayerPointerRecovery,
        ] = {}
        self.negative_cache: dict[tuple[int, int], float] = {}
        self.inflight: dict[tuple[int, int], _RecoveryFlight] = {}
        self.last_metrics: dict[tuple[int, int], PointerRecoveryMetrics] = {}
        self.pending_candidates: dict[
            tuple[int, int], AnchoredPointerCandidate
        ] = {}

    def clear(
        self,
        *,
        pid: int | None = None,
        module_base: int | None = None,
    ) -> None:
        with self.lock:
            if pid is None and module_base is None:
                self.recovery_cache.clear()
                self.negative_cache.clear()
                self.last_metrics.clear()
                self.pending_candidates.clear()
                return
            keys = (
                set(self.recovery_cache)
                | set(self.negative_cache)
                | set(self.last_metrics)
                | set(self.pending_candidates)
            )
            for key in keys:
                if pid is not None and key[0] != int(pid):
                    continue
                if module_base is not None and key[1] != int(module_base):
                    continue
                self.recovery_cache.pop(key, None)
                self.negative_cache.pop(key, None)
                self.last_metrics.pop(key, None)
                self.pending_candidates.pop(key, None)

    def metrics_for(
        self,
        pid: int,
        module_base: int,
    ) -> PointerRecoveryMetrics | None:
        with self.lock:
            return self.last_metrics.get((int(pid), int(module_base)))


_PERSIST_LOCK = RLock()
MINIMUM_NEGATIVE_COOLDOWN_SECONDS = 5.0
DEFAULT_RECOVERY_TIMEOUT_SECONDS = 0.5


def _u32(memory: PointerRecoveryMemory, address: int) -> int:
    return int(struct.unpack("<I", memory.read(address, 4))[0])


def _i32(memory: PointerRecoveryMemory, address: int) -> int:
    return int(struct.unpack("<i", memory.read(address, 4))[0])


def _f32(memory: PointerRecoveryMemory, address: int) -> float:
    return float(struct.unpack("<f", memory.read(address, 4))[0])


def _region_bounds(region: object) -> tuple[int, int]:
    start = int(getattr(region, "base_address"))
    size = int(getattr(region, "size"))
    return start, start + max(0, size)


def _contains(
    regions: tuple[object, ...] | _RegionIndex,
    address: int,
    size: int = 1,
) -> bool:
    """Compatibility wrapper; recovery builds one index and reuses it."""

    index = (
        regions
        if isinstance(regions, _RegionIndex)
        else _RegionIndex.build(regions)
    )
    return index.contains(address, size)


def _cancellation_requested(cancellation: object | None) -> bool:
    if cancellation is None:
        return False
    cancelled = getattr(cancellation, "cancelled", None)
    if cancelled is not None:
        return bool(cancelled() if callable(cancelled) else cancelled)
    is_set = getattr(cancellation, "is_set", None)
    if callable(is_set):
        return bool(is_set())
    return False


def _notify(
    callback: PointerRecoveryStatusCallback | None,
    phase: str,
    message: str,
    metrics: PointerRecoveryMetrics,
) -> None:
    if callback is None:
        return
    try:
        callback(
            PointerRecoveryProgress(
                phase=phase,
                message=message,
                metrics=metrics,
            )
        )
    except Exception:
        # Recovery diagnostics must never turn a read-only diagnostic into a
        # runtime failure.
        return


def clear_pointer_recovery_state(
    *,
    state: PointerRecoveryState | None = None,
    pid: int | None = None,
    module_base: int | None = None,
) -> None:
    """Invalidate one explicitly owned recovery state.

    Calls without ``state`` are retained as a compatibility no-op. New code
    owns a :class:`PointerRecoveryState` for the lifetime of one attachment.
    """

    if state is not None:
        state.clear(pid=pid, module_base=module_base)


def get_last_pointer_recovery_metrics(
    pid: int,
    module_base: int,
    *,
    state: PointerRecoveryState | None = None,
) -> PointerRecoveryMetrics | None:
    if state is None:
        return None
    return state.metrics_for(pid, module_base)


def _read_configured_world_hint(
    memory: PointerRecoveryMemory,
    module_base: int,
    config: NativeMonsterConfig,
) -> int | None:
    try:
        value = _u32(memory, module_base + config.world_pointer_offset)
    except Exception:
        return None
    return value if value > 0 else None


_SELF_MISMATCH_NEAR_MATCH_LIMIT = 1024


def _record_self_mismatch_near_match(
    memory: PointerRecoveryMemory,
    *,
    target_address: int,
    config: NativeMonsterConfig,
    slot_refs: dict[int, list[int]],
    metrics: _MetricsBuilder,
) -> None:
    """Continue bounded structural validation after the legacy self check fails.

    These counters are diagnostic evidence only.  They never make a legacy
    candidate acceptable; anchored discovery must independently infer the
    current layout before a pointer can be applied or persisted.
    """

    if metrics.self_mismatch_near_probed >= _SELF_MISMATCH_NEAR_MATCH_LIMIT:
        return

    offsets = (
        config.world_offset,
        config.x_offset,
        config.y_offset,
        config.z_offset,
        config.species_offset,
        config.active_species_offset,
        config.hp_offset,
    )
    first_offset = min(offsets)
    last_offset = max(offsets) + 4
    try:
        sample = memory.read(
            target_address + first_offset,
            last_offset - first_offset,
        )
        world = struct.unpack_from("<I", sample, config.world_offset - first_offset)[0]
        x = struct.unpack_from("<f", sample, config.x_offset - first_offset)[0]
        y = struct.unpack_from("<f", sample, config.y_offset - first_offset)[0]
        z = struct.unpack_from("<f", sample, config.z_offset - first_offset)[0]
        species = struct.unpack_from(
            "<i", sample, config.species_offset - first_offset
        )[0]
        active_species = struct.unpack_from(
            "<i", sample, config.active_species_offset - first_offset
        )[0]
        hp = struct.unpack_from("<i", sample, config.hp_offset - first_offset)[0]
    except Exception:
        metrics.candidate_read_failures += 1
        return

    metrics.self_mismatch_near_probed += 1
    world_nonzero = world > 0
    world_module_ref = bool(slot_refs.get(world)) if world_nonzero else False
    limit = float(config.maximum_absolute_coordinate)
    coordinates_plausible = all(
        math.isfinite(value) and abs(value) <= limit for value in (x, y, z)
    )
    hp_positive = hp > 0
    player_like = not (species > 0 and active_species == species)

    metrics.self_mismatch_world_nonzero += int(world_nonzero)
    metrics.self_mismatch_world_module_ref += int(world_module_ref)
    metrics.self_mismatch_coordinate_plausible += int(coordinates_plausible)
    metrics.self_mismatch_hp_positive += int(hp_positive)
    metrics.self_mismatch_player_like += int(player_like)
    metrics.self_mismatch_full_near_matches += int(
        world_nonzero
        and world_module_ref
        and coordinates_plausible
        and hp_positive
        and player_like
    )


def _validate_candidate(
    memory: PointerRecoveryMemory,
    *,
    slot_address: int,
    target_address: int,
    reference_count: int,
    configured_slot_address: int,
    module_base: int,
    config: NativeMonsterConfig,
    configured_world_hint: int | None,
    slot_refs: dict[int, list[int]],
    metrics: _MetricsBuilder,
) -> _ValidatedCandidate | None:
    try:
        self_pointer = _u32(memory, target_address + config.self_pointer_offset)
    except Exception:
        metrics.candidate_read_failures += 1
        return None
    if self_pointer != target_address:
        metrics.self_mismatch_rejections += 1
        _record_self_mismatch_near_match(
            memory,
            target_address=target_address,
            config=config,
            slot_refs=slot_refs,
            metrics=metrics,
        )
        return None
    try:
        world = _u32(memory, target_address + config.world_offset)
    except Exception:
        metrics.candidate_read_failures += 1
        return None
    if world <= 0:
        metrics.world_null_rejections += 1
        return None
    try:
        x = _f32(memory, target_address + config.x_offset)
        y = _f32(memory, target_address + config.y_offset)
        z = _f32(memory, target_address + config.z_offset)
    except Exception:
        metrics.candidate_read_failures += 1
        return None
    limit = float(config.maximum_absolute_coordinate)
    if not all(math.isfinite(value) for value in (x, y, z)) or any(
        abs(value) > limit for value in (x, y, z)
    ):
        metrics.coordinate_rejections += 1
        return None
    try:
        species = _i32(memory, target_address + config.species_offset)
        active_species = _i32(memory, target_address + config.active_species_offset)
        hp = _i32(memory, target_address + config.hp_offset)
    except Exception:
        metrics.candidate_read_failures += 1
        return None
    if hp <= 0:
        metrics.hp_rejections += 1
        return None

    offset_delta = config.world_pointer_offset - config.player_pointer_offset
    historical_world_slot = slot_address + offset_delta
    configured_world_slot = module_base + config.world_pointer_offset
    discovered_world_slots = [
        slot
        for slot in slot_refs.get(world, ())
        if slot != slot_address
    ]
    # Preserve direct evidence from the historical/configured locations even
    # when an injected backend does not enumerate the world object's region.
    # Independently moved globals are still obtained from the module scan.
    for hinted_slot in (historical_world_slot, configured_world_slot):
        if hinted_slot == slot_address or hinted_slot in discovered_world_slots:
            continue
        try:
            if _u32(memory, hinted_slot) == world:
                discovered_world_slots.append(hinted_slot)
        except Exception:
            pass
    world_slots = tuple(discovered_world_slots)
    world_pointer_slot = None
    if world_slots:
        world_pointer_slot = min(
            world_slots,
            key=lambda slot: (
                int(slot != historical_world_slot),
                int(slot != configured_world_slot),
                abs(slot - configured_world_slot),
            ),
        )
    pair_matches = historical_world_slot in world_slots

    configured_world_matches = (
        configured_world_hint is not None and configured_world_hint == world
    )
    player_like = not (species > 0 and active_species == species)
    return _ValidatedCandidate(
        slot_address=slot_address,
        target_address=target_address,
        world_base=world,
        distance=abs(slot_address - configured_slot_address),
        reference_count=reference_count,
        world_pointer_slot=world_pointer_slot,
        pair_matches=pair_matches,
        configured_world_matches=configured_world_matches,
        player_like=player_like,
        world_reference_count=len(world_slots),
    )


def _stable_candidate(
    memory: PointerRecoveryMemory,
    candidate: _ValidatedCandidate,
    config: NativeMonsterConfig,
    *,
    samples: int = 3,
    delay_seconds: float = 0.03,
    control: _AttemptControl | None = None,
) -> bool:
    # A recovered global may be persisted later from the success cache, so
    # every cached result must meet the same minimum multi-sample standard.
    sample_count = max(3, int(samples))
    for index in range(sample_count):
        if control is not None:
            control.check()
        try:
            if _u32(memory, candidate.slot_address) != candidate.target_address:
                return False
            if (
                _u32(
                    memory,
                    candidate.target_address + config.self_pointer_offset,
                )
                != candidate.target_address
            ):
                return False
            if (
                _u32(memory, candidate.target_address + config.world_offset)
                != candidate.world_base
            ):
                return False
            if candidate.world_pointer_slot is None or (
                _u32(memory, candidate.world_pointer_slot) != candidate.world_base
            ):
                return False
            coordinates = (
                _f32(memory, candidate.target_address + config.x_offset),
                _f32(memory, candidate.target_address + config.y_offset),
                _f32(memory, candidate.target_address + config.z_offset),
            )
            if not all(math.isfinite(value) for value in coordinates):
                return False
        except Exception:
            return False
        if index + 1 < sample_count and delay_seconds > 0.0:
            if control is None:
                sleep(delay_seconds)
            else:
                control.wait(delay_seconds)
    return True


class PointerPersistenceError(RuntimeError):
    """A two-config pointer update could not commit or recover safely."""


@dataclass(frozen=True, slots=True)
class _PersistenceTarget:
    path: Path
    original_existed: bool
    original_bytes: bytes
    replacement_bytes: bytes
    changed: bool
    backup_path: Path
    backup_existed: bool
    replacement_path: Path
    backup_stage_path: Path
    restore_path: Path


_POINTER_TRANSACTION_VERSION = 1


def _canonical_path(path: str | Path) -> Path:
    return Path(path).resolve(strict=False)


def _journal_path(position_config_path: Path) -> Path:
    return position_config_path.with_name(
        f".{position_config_path.name}.pointer_recovery.transaction.json"
    )


def _journal_stage_path(journal_path: Path) -> Path:
    return journal_path.with_name(f"{journal_path.name}.tmp")


def _target_paths(path: Path) -> tuple[Path, Path, Path, Path]:
    backup = path.with_suffix(path.suffix + ".pre_pointer_recovery.bak")
    replacement = path.with_name(
        f".{path.name}.pointer_recovery.replacement.tmp"
    )
    backup_stage = path.with_name(
        f".{path.name}.pointer_recovery.backup.tmp"
    )
    restore = path.with_name(f".{path.name}.pointer_recovery.restore.tmp")
    return backup, replacement, backup_stage, restore


def _sync_directory(directory: Path) -> None:
    """Best-effort metadata flush; opening directories is unsupported on Windows."""

    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _write_durable_file(path: Path, data: bytes) -> None:
    path.unlink(missing_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_replace(source: Path, destination: Path) -> None:
    source.replace(destination)
    _sync_directory(destination.parent)


def _remove_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _sync_directory(path.parent)


def _prepare_persistence_target(
    path: Path,
    updates: dict[str, object],
) -> _PersistenceTarget:
    if not path.parent.is_dir():
        raise PointerPersistenceError(
            f"Configuration directory does not exist: {path.parent}"
        )
    original_existed = path.is_file()
    if not original_existed:
        raise PointerPersistenceError(f"Configuration does not exist: {path}")
    original_bytes = path.read_bytes()
    try:
        payload = json.loads(original_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PointerPersistenceError(
            f"Configuration is not valid UTF-8 JSON: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise PointerPersistenceError(
            f"Configuration root must be an object: {path}"
        )

    changed = False
    for key, value in updates.items():
        if key in {"layout", "recovery_hints"} and isinstance(value, dict):
            nested = payload.get(key)
            if not isinstance(nested, dict):
                nested = {}
                payload[key] = nested
            for nested_key, nested_value in value.items():
                if nested.get(nested_key) != nested_value:
                    nested[nested_key] = nested_value
                    changed = True
        elif payload.get(key) != value:
            payload[key] = value
            changed = True
    replacement_bytes = (
        (json.dumps(payload, indent=2) + "\n").encode("utf-8")
        if changed
        else original_bytes
    )
    try:
        replacement_payload = json.loads(replacement_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PointerPersistenceError(
            f"Prepared replacement is not valid JSON: {path}"
        ) from error
    def update_matches(key: str, value: object) -> bool:
        if key not in {"layout", "recovery_hints"} or not isinstance(value, dict):
            return replacement_payload.get(key) == value
        nested = replacement_payload.get(key)
        return isinstance(nested, dict) and all(
            nested.get(nested_key) == nested_value
            for nested_key, nested_value in value.items()
        )

    if not isinstance(replacement_payload, dict) or any(
        not update_matches(key, value) for key, value in updates.items()
    ):
        raise PointerPersistenceError(
            f"Prepared replacement failed validation: {path}"
        )

    backup, replacement, backup_stage, restore = _target_paths(path)
    return _PersistenceTarget(
        path=path,
        original_existed=original_existed,
        original_bytes=original_bytes,
        replacement_bytes=replacement_bytes,
        changed=changed,
        backup_path=backup,
        backup_existed=backup.exists(),
        replacement_path=replacement,
        backup_stage_path=backup_stage,
        restore_path=restore,
    )


def _target_record(target: _PersistenceTarget) -> dict[str, object]:
    return {
        "path": str(target.path),
        "original_existed": target.original_existed,
        "original_base64": base64.b64encode(target.original_bytes).decode("ascii"),
        "changed": target.changed,
        "backup_existed": target.backup_existed,
    }


def _transaction_bytes(targets: tuple[_PersistenceTarget, ...]) -> bytes:
    payload = {
        "version": _POINTER_TRANSACTION_VERSION,
        "targets": [_target_record(target) for target in targets],
    }
    return (json.dumps(payload, indent=2) + "\n").encode("utf-8")


def _parse_transaction_target(value: object) -> _PersistenceTarget:
    if not isinstance(value, dict):
        raise PointerPersistenceError("Transaction target must be an object")
    path_value = value.get("path")
    original_existed = value.get("original_existed")
    original_base64 = value.get("original_base64")
    changed = value.get("changed")
    backup_existed = value.get("backup_existed")
    if (
        not isinstance(path_value, str)
        or not isinstance(original_existed, bool)
        or not isinstance(original_base64, str)
        or not isinstance(changed, bool)
        or not isinstance(backup_existed, bool)
    ):
        raise PointerPersistenceError("Transaction target fields are invalid")
    try:
        original_bytes = base64.b64decode(
            original_base64.encode("ascii"),
            validate=True,
        )
    except (UnicodeEncodeError, ValueError) as error:
        raise PointerPersistenceError(
            "Transaction target original bytes are invalid"
        ) from error
    path = _canonical_path(path_value)
    backup, replacement, backup_stage, restore = _target_paths(path)
    return _PersistenceTarget(
        path=path,
        original_existed=original_existed,
        original_bytes=original_bytes,
        replacement_bytes=b"",
        changed=changed,
        backup_path=backup,
        backup_existed=backup_existed,
        replacement_path=replacement,
        backup_stage_path=backup_stage,
        restore_path=restore,
    )


def _read_transaction(
    journal: Path,
    expected_paths: tuple[Path, Path],
) -> tuple[_PersistenceTarget, ...]:
    try:
        value = json.loads(journal.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PointerPersistenceError(
            f"Could not read pointer persistence journal: {journal}"
        ) from error
    if not isinstance(value, dict):
        raise PointerPersistenceError("Pointer persistence journal is invalid")
    if value.get("version") != _POINTER_TRANSACTION_VERSION:
        raise PointerPersistenceError(
            "Pointer persistence journal version is unsupported"
        )
    raw_targets = value.get("targets")
    if not isinstance(raw_targets, list) or len(raw_targets) != 2:
        raise PointerPersistenceError(
            "Pointer persistence journal must contain two targets"
        )
    targets = tuple(_parse_transaction_target(item) for item in raw_targets)
    if tuple(target.path for target in targets) != expected_paths:
        raise PointerPersistenceError(
            "Pointer persistence journal targets do not match this request"
        )
    return targets


def _cleanup_target_temps(targets: tuple[_PersistenceTarget, ...]) -> None:
    for target in targets:
        _remove_file(target.replacement_path)
        _remove_file(target.backup_stage_path)
        _remove_file(target.restore_path)


def _restore_persistence_target(target: _PersistenceTarget) -> None:
    if target.original_existed:
        current = target.path.read_bytes() if target.path.is_file() else None
        if current != target.original_bytes:
            _write_durable_file(target.restore_path, target.original_bytes)
            _atomic_replace(target.restore_path, target.path)
    elif target.path.exists():
        _remove_file(target.path)

    if target.changed and not target.backup_existed:
        _remove_file(target.backup_path)


def recover_interrupted_pointer_persistence(
    *,
    position_config_path: str | Path = DEFAULT_POSITION_CONFIG_PATH,
    monster_config_path: str | Path = DEFAULT_MONSTER_CONFIG_PATH,
) -> bool:
    """Rollback an interrupted two-config commit, if its journal is present."""

    position_path = _canonical_path(position_config_path)
    monster_path = _canonical_path(monster_config_path)
    if position_path == monster_path:
        raise PointerPersistenceError(
            "Position and monster configurations must be different files"
        )
    expected_paths = (position_path, monster_path)
    journal = _journal_path(position_path)
    journal_stage = _journal_stage_path(journal)
    with _PERSIST_LOCK:
        if not journal.exists():
            _remove_file(journal_stage)
            for path in expected_paths:
                backup, replacement, backup_stage, restore = _target_paths(path)
                del backup
                _remove_file(replacement)
                _remove_file(backup_stage)
                _remove_file(restore)
            return False

        targets = _read_transaction(journal, expected_paths)
        try:
            for target in targets:
                _restore_persistence_target(target)
            _cleanup_target_temps(targets)
            _remove_file(journal_stage)
            _remove_file(journal)
        except Exception as error:
            raise PointerPersistenceError(
                "Interrupted pointer persistence rollback is incomplete; "
                f"journal retained at {journal}"
            ) from error
        return True


def persist_recovered_pointer_offsets(
    recovery: PlayerPointerRecovery,
    *,
    position_config_path: str | Path = DEFAULT_POSITION_CONFIG_PATH,
    monster_config_path: str | Path = DEFAULT_MONSTER_CONFIG_PATH,
) -> None:
    if recovery.strategy == "anchored_movement" and not recovery.movement_validated:
        raise PointerPersistenceError(
            "Anchored pointer recovery requires movement validation before persistence"
        )
    if recovery.strategy == "anchored_independent":
        # Actor anchors are process-session addresses and no validated world
        # pointer exists. Persisting only half of the old pointer pair would make
        # the next attachment look configured while still requiring discovery.
        return
    with _PERSIST_LOCK:
        position_path = _canonical_path(position_config_path)
        monster_path = _canonical_path(monster_config_path)
        if position_path == monster_path:
            raise PointerPersistenceError(
                "Position and monster configurations must be different files"
            )
        recover_interrupted_pointer_persistence(
            position_config_path=position_path,
            monster_config_path=monster_path,
        )
        recovered_player_hint = f"0x{recovery.player_pointer_offset:X}"
        monster_updates: dict[str, object] = {
            "recovery_hints": {
                "player_pointer_offset": recovered_player_hint,
            },
            # Backward-compatible mirror for older external tooling. Production
            # reads the nested recovery_hints object first.
            "player_pointer_offset": recovered_player_hint,
        }
        if recovery.strategy == "anchored_movement":
            monster_updates["player_pointer_chain_offsets"] = [
                f"0x{offset:X}" for offset in recovery.player_pointer_chain_offsets
            ]
            monster_updates["world_pointer_chain_offsets"] = [
                f"0x{offset:X}" for offset in recovery.world_pointer_chain_offsets
            ]
        if recovery.world_pointer_offset is not None:
            recovered_world_hint = f"0x{recovery.world_pointer_offset:X}"
            recovery_hints = monster_updates["recovery_hints"]
            assert isinstance(recovery_hints, dict)
            recovery_hints["world_pointer_offset"] = recovered_world_hint
            monster_updates["world_pointer_offset"] = recovered_world_hint
        layout_updates: dict[str, str] = {}
        if recovery.world_field_offset is not None:
            layout_updates["world_offset"] = f"0x{recovery.world_field_offset:X}"
        if recovery.world_vtable_offset is not None:
            layout_updates["world_vtable_offset"] = (
                f"0x{recovery.world_vtable_offset:X}"
            )
        if recovery.world_vtable_field_offset is not None:
            layout_updates["world_vtable_field_offset"] = (
                f"0x{recovery.world_vtable_field_offset:X}"
            )
        if recovery.world_identity_kind is not None:
            layout_updates["world_identity_kind"] = recovery.world_identity_kind
        if recovery.self_pointer_offset is not None:
            layout_updates["self_pointer_offset"] = (
                f"0x{recovery.self_pointer_offset:X}"
            )
        for key, offset in (
            ("species_offset", recovery.species_offset),
            ("active_species_offset", recovery.active_species_offset),
            ("hp_offset", recovery.hp_offset),
            ("x_offset", recovery.x_offset),
            ("y_offset", recovery.y_offset),
            ("z_offset", recovery.z_offset),
        ):
            if offset is not None:
                layout_updates[key] = f"0x{offset:X}"
        if layout_updates:
            monster_updates["layout"] = layout_updates
        position_updates: dict[str, object] = {
            "pointer_hint_offset": recovered_player_hint,
            # Backward-compatible mirror for older external tooling.
            "pointer_offset": recovered_player_hint,
        }
        if recovery.strategy == "anchored_movement":
            position_updates["pointer_chain_offsets"] = [
                f"0x{offset:X}" for offset in recovery.player_pointer_chain_offsets
            ]
        position_layout_updates = {
            key: f"0x{offset:X}"
            for key, offset in (
                ("x_offset", recovery.x_offset),
                ("y_offset", recovery.y_offset),
                ("z_offset", recovery.z_offset),
            )
            if offset is not None
        }
        if position_layout_updates:
            position_updates["layout"] = position_layout_updates
        targets = (
            _prepare_persistence_target(position_path, position_updates),
            _prepare_persistence_target(monster_path, monster_updates),
        )
        if not any(target.changed for target in targets):
            return

        journal = _journal_path(position_path)
        journal_stage = _journal_stage_path(journal)
        try:
            for target in targets:
                if not target.changed:
                    continue
                _write_durable_file(
                    target.replacement_path,
                    target.replacement_bytes,
                )
                if not target.backup_existed:
                    _write_durable_file(
                        target.backup_stage_path,
                        target.original_bytes,
                    )

            # Re-read both sources only after every replacement has been
            # prepared and validated, but before the recovery journal or any
            # destination is touched.
            for target in targets:
                if (
                    target.path.is_file() != target.original_existed
                    or (
                        target.original_existed
                        and target.path.read_bytes() != target.original_bytes
                    )
                    or target.backup_path.exists() != target.backup_existed
                ):
                    raise PointerPersistenceError(
                        "Pointer configuration changed while preparing the "
                        f"transaction: {target.path}"
                    )

            _write_durable_file(journal_stage, _transaction_bytes(targets))
            _atomic_replace(journal_stage, journal)
        except Exception:
            _cleanup_target_temps(targets)
            _remove_file(journal_stage)
            raise

        try:
            for target in targets:
                if target.changed and not target.backup_existed:
                    _atomic_replace(
                        target.backup_stage_path,
                        target.backup_path,
                    )
            for target in targets:
                if target.changed:
                    _atomic_replace(target.replacement_path, target.path)
            _remove_file(journal)
        except Exception as commit_error:
            try:
                recover_interrupted_pointer_persistence(
                    position_config_path=position_path,
                    monster_config_path=monster_path,
                )
            except Exception as rollback_error:
                raise PointerPersistenceError(
                    "Pointer configuration commit failed and rollback is "
                    f"incomplete; recovery journal retained at {journal}"
                ) from rollback_error
            raise PointerPersistenceError(
                "Pointer configuration commit failed and was rolled back"
            ) from commit_error


def _persist_if_requested(
    recovery: PlayerPointerRecovery,
    *,
    requested: bool,
    position_config_path: str | Path | None = None,
    monster_config_path: str | Path | None = None,
) -> None:
    if not requested:
        return
    try:
        if position_config_path is None and monster_config_path is None:
            persist_recovered_pointer_offsets(recovery)
        elif position_config_path is None or monster_config_path is None:
            raise ValueError(
                "Both persistence config paths must be provided together"
            )
        else:
            persist_recovered_pointer_offsets(
                recovery,
                position_config_path=position_config_path,
                monster_config_path=monster_config_path,
            )
    except Exception as error:
        print(
            "Native pointer recovery succeeded but config persistence "
            f"failed: {type(error).__name__}: {error}"
        )


def _verify_cached(
    memory: PointerRecoveryMemory,
    cached: PlayerPointerRecovery,
    config: NativeMonsterConfig,
    module_base: int,
) -> bool:
    try:
        if cached.independent_discovery is not None:
            target = cached.independent_discovery.player
            if target is None:
                return False
            player = _u32(memory, cached.player_pointer_address)
            if player <= 0:
                return False
            if not any(
                _u32(memory, player + offset) == player
                for offset in target.self_pointer_offsets
            ):
                return False
            hp = struct.unpack("<i", memory.read(player + target.hp_offset, 4))[0]
            coordinates = tuple(
                struct.unpack("<f", memory.read(player + offset, 4))[0]
                for offset in (target.x_offset, target.y_offset, target.z_offset)
            )
            return hp >= 0 and all(math.isfinite(value) for value in coordinates)
        player = _u32(memory, cached.player_pointer_address)
        for offset in cached.player_pointer_chain_offsets:
            player = _u32(memory, player + offset)
        if cached.world_pointer_address is None:
            return False
        world = _u32(memory, cached.world_pointer_address)
        for offset in cached.world_pointer_chain_offsets:
            world = _u32(memory, world + offset)
        self_offset = (
            config.self_pointer_offset
            if cached.self_pointer_offset is None
            else cached.self_pointer_offset
        )
        world_field_offset = (
            config.world_offset
            if cached.world_field_offset is None
            else cached.world_field_offset
        )
        world_vtable_offset = (
            config.world_vtable_offset
            if cached.world_vtable_offset is None
            else cached.world_vtable_offset
        )
        world_vtable_field_offset = (
            config.world_vtable_field_offset
            if cached.world_vtable_field_offset is None
            else cached.world_vtable_field_offset
        )
        return (
            player == cached.player_base
            and world == cached.world_base
            and _u32(memory, cached.player_base + self_offset) == cached.player_base
            and (
                (
                    cached.player_pointer_address
                    == cached.world_pointer_address
                    and not cached.world_pointer_chain_offsets
                    and len(cached.player_pointer_chain_offsets) == 1
                )
                or _u32(memory, cached.player_base + world_field_offset)
                == cached.world_base
            )
            and (
                world_vtable_offset is None
                or _u32(
                    memory,
                    cached.world_base + world_vtable_field_offset,
                )
                == int(module_base) + world_vtable_offset
            )
        )
    except Exception:
        return False


def _progress_metrics(
    builder: _MetricsBuilder,
    control: _AttemptControl,
) -> PointerRecoveryMetrics:
    return builder.freeze("running", now=control.clock())


def _record_anchor_evidence(
    metrics: _MetricsBuilder,
    evidence: AnchoredDiscoveryEvidence,
) -> None:
    metrics.anchor_bytes_scanned = evidence.anchor_bytes_scanned
    metrics.anchor_regions_scanned = evidence.anchor_regions_scanned
    metrics.anchor_read_failures = evidence.anchor_read_failures
    metrics.species_value_matches = evidence.species_value_matches
    metrics.spawn_x_matches = evidence.spawn_x_matches
    metrics.monster_candidates = evidence.monster_candidates
    metrics.monster_species_rejections = evidence.monster_species_rejections
    metrics.monster_active_rejections = evidence.monster_active_rejections
    metrics.monster_hp_rejections = evidence.monster_hp_rejections
    metrics.monster_coordinate_rejections = evidence.monster_coordinate_rejections
    metrics.monster_base_hypotheses = evidence.monster_base_hypotheses
    metrics.monster_layout_species_support = (
        evidence.monster_layout_species_support
    )
    metrics.monster_layout_ties = evidence.monster_layout_ties
    metrics.monster_self_field_aliases = evidence.monster_self_field_aliases
    metrics.inferred_world_actor_support = evidence.inferred_world_actor_support
    metrics.inferred_world_species_support = evidence.inferred_world_species_support
    metrics.inferred_world_offset = evidence.inferred_world_offset
    metrics.inferred_world_vtable = evidence.inferred_world_vtable
    metrics.inferred_world_vtable_field_offset = (
        evidence.inferred_world_vtable_field_offset
    )
    metrics.inferred_world_identity_kind = evidence.inferred_world_identity_kind
    metrics.inferred_world_readable_pointer_fields = (
        evidence.inferred_world_readable_pointer_fields
    )
    metrics.inferred_world_distinct_values = (
        evidence.inferred_world_distinct_values
    )
    metrics.structural_world_hypotheses = evidence.structural_world_hypotheses
    metrics.world_object_rejections = evidence.world_object_rejections
    metrics.world_identity_candidates = evidence.world_identity_candidates
    metrics.world_identity_span_rejections = (
        evidence.world_identity_span_rejections
    )
    metrics.world_identity_vtable_misses = evidence.world_identity_vtable_misses
    metrics.world_identity_unstable_rejections = (
        evidence.world_identity_unstable_rejections
    )
    metrics.world_identity_module_pointer_fields = (
        evidence.world_identity_module_pointer_fields
    )
    metrics.world_identity_readable_pointer_fields = (
        evidence.world_identity_readable_pointer_fields
    )
    metrics.world_identity_distinct_values = (
        evidence.world_identity_distinct_values
    )
    metrics.world_identity_structural_rejections = (
        evidence.world_identity_structural_rejections
    )
    metrics.world_identity_marker_accepts = evidence.world_identity_marker_accepts
    metrics.world_near_actor_support = evidence.world_near_actor_support
    metrics.world_near_species_support = evidence.world_near_species_support
    metrics.world_near_module_references = evidence.world_near_module_references
    metrics.world_near_field_offset = evidence.world_near_field_offset
    metrics.world_near_target = evidence.world_near_target
    metrics.world_near_module_pointer_fields = (
        evidence.world_near_module_pointer_fields
    )
    metrics.world_near_readable_pointer_fields = (
        evidence.world_near_readable_pointer_fields
    )
    metrics.world_near_distinct_values = evidence.world_near_distinct_values
    metrics.inferred_self_actor_support = evidence.inferred_self_actor_support
    metrics.inferred_self_offset = evidence.inferred_self_offset
    metrics.inferred_species_offset = evidence.inferred_species_offset
    metrics.inferred_active_species_offset = (
        evidence.inferred_active_species_offset
    )
    metrics.inferred_hp_offset = evidence.inferred_hp_offset
    metrics.inferred_player_hp_offset = evidence.inferred_player_hp_offset
    metrics.inferred_player_max_hp_offset = evidence.inferred_player_max_hp_offset
    metrics.inferred_x_offset = evidence.inferred_x_offset
    metrics.inferred_y_offset = evidence.inferred_y_offset
    metrics.inferred_z_offset = evidence.inferred_z_offset
    metrics.spawn_structure_candidates = evidence.spawn_structure_candidates
    metrics.spawn_world_matches = evidence.spawn_world_matches
    metrics.spawn_world_hypothesis_matches = (
        evidence.spawn_world_hypothesis_matches
    )
    metrics.spawn_hp_matches = evidence.spawn_hp_matches
    metrics.spawn_player_matches = evidence.spawn_player_matches
    metrics.stable_spawn_candidates = evidence.stable_spawn_candidates
    metrics.direct_player_slot_candidates = evidence.direct_player_slot_candidates
    metrics.player_chain_candidates = evidence.player_chain_candidates
    metrics.direct_world_slot_candidates = evidence.direct_world_slot_candidates
    metrics.world_chain_candidates = evidence.world_chain_candidates
    metrics.player_reference_matches = evidence.player_reference_matches
    metrics.player_world_chain_candidates = evidence.player_world_chain_candidates
    metrics.player_world_rooted_matches = evidence.player_world_rooted_matches
    metrics.player_reference_ambiguities = evidence.player_reference_ambiguities
    metrics.movement_checks = evidence.movement_checks
    metrics.movement_observed = evidence.movement_observed


def _record_trace_evidence(
    metrics: _MetricsBuilder,
    evidence: TraceTargetEvidence,
) -> None:
    """Map the validated standalone discovery evidence into recovery metrics."""

    metrics.anchor_bytes_scanned = int(evidence.bytes_scanned)
    metrics.anchor_regions_scanned = int(evidence.regions_scanned)
    metrics.anchor_read_failures = int(evidence.read_failures)
    metrics.species_value_matches = int(evidence.species_hits)
    metrics.spawn_x_matches = int(evidence.spawn_x_hits)
    metrics.monster_candidates = int(evidence.monster_candidates)
    metrics.monster_hp_rejections = int(evidence.monster_hp_rejections)
    metrics.monster_coordinate_rejections = int(
        evidence.monster_coordinate_rejections
    )
    metrics.monster_base_hypotheses = int(evidence.monster_base_hypotheses)
    metrics.monster_layout_ties = int(evidence.monster_layout_ties)
    metrics.monster_self_field_aliases = int(evidence.monster_self_aliases)
    metrics.inferred_species_offset = evidence.inferred_species_offset
    metrics.inferred_active_species_offset = (
        evidence.inferred_active_species_offset
    )
    metrics.inferred_hp_offset = evidence.inferred_monster_hp_offset
    metrics.inferred_player_hp_offset = evidence.selected_player_hp_offset
    metrics.inferred_x_offset = evidence.inferred_x_offset
    metrics.inferred_y_offset = evidence.inferred_y_offset
    metrics.inferred_z_offset = evidence.inferred_z_offset
    metrics.spawn_structure_candidates = int(evidence.player_candidates)
    metrics.spawn_player_matches = int(evidence.player_candidates)
    metrics.stable_spawn_candidates = int(evidence.player_candidates)


def _scan_new_band(
    memory: PointerRecoveryMemory,
    *,
    start: int,
    stop: int,
    chunk_size: int,
    region_index: _RegionIndex,
    slot_refs: dict[int, list[int]],
    metrics: _MetricsBuilder,
    control: _AttemptControl,
) -> None:
    intervals = region_index.intersections(start, stop)
    metrics.scan_intervals += len(intervals)
    for interval_start, interval_stop in intervals:
        cursor = interval_start
        while cursor < interval_stop:
            control.check()
            amount = min(max(0x1000, int(chunk_size)), interval_stop - cursor)
            try:
                data = memory.read(cursor, amount)
            except Exception:
                metrics.read_failures += 1
                cursor += amount
                continue
            metrics.chunks_read += 1
            metrics.bytes_scanned += len(data)
            control.check()

            first = (-cursor) % 4
            for relative in range(first, len(data) - 3, 4):
                if (relative - first) % 0x400 == 0:
                    control.check()
                if (relative - first) % 0x10000 == 0:
                    # The scan is worker-owned, but explicitly yield the GIL so
                    # Tk/preview work remains smooth during CPU-heavy chunks.
                    sleep(0)
                metrics.aligned_words_examined += 1
                target = int(struct.unpack_from("<I", data, relative)[0])
                if target <= 0x10000 or target % 4 != 0:
                    continue
                metrics.pointer_like_words += 1
                metrics.containment_checks += 1
                # Keep module references to any readable target. Player-layout
                # containment is checked separately during candidate validation;
                # world objects need not occupy an actor-sized region.
                if not region_index.contains(target, 4):
                    continue
                slot = cursor + relative
                refs = slot_refs.setdefault(target, [])
                if len(refs) < 8:
                    refs.append(slot)
            cursor += amount


def _module_image_extent(
    memory: PointerRecoveryMemory,
    module_name: str,
    module_base: int,
) -> tuple[int, int] | None:
    """Return a trustworthy module image extent when the backend exposes it."""

    module_info = getattr(memory, "module_info", None)
    if not callable(module_info):
        return None
    try:
        info = module_info(module_name)
        base = int(getattr(info, "base_address"))
        size = int(getattr(info, "size"))
    except Exception:
        return None
    if base != int(module_base) or size <= 0:
        return None
    return base, base + size


def _anchored_recovery(
    candidate: AnchoredPointerCandidate,
    *,
    module_base: int,
    configured_player_pointer_offset: int,
    configured_world_pointer_offset: int,
    validated_candidates: int,
) -> PlayerPointerRecovery:
    return PlayerPointerRecovery(
        player_pointer_address=candidate.player_pointer_address,
        player_pointer_offset=candidate.player_pointer_address - module_base,
        player_base=candidate.player_base,
        world_base=candidate.world_base,
        world_pointer_address=candidate.world_pointer_address,
        world_pointer_offset=candidate.world_pointer_address - module_base,
        configured_player_pointer_offset=configured_player_pointer_offset,
        configured_world_pointer_offset=configured_world_pointer_offset,
        search_radius=0,
        validated_candidates=validated_candidates,
        strategy="anchored_movement",
        player_pointer_chain_offsets=candidate.player_pointer_chain_offsets,
        world_pointer_chain_offsets=candidate.world_pointer_chain_offsets,
        world_field_offset=candidate.world_field_offset,
        world_vtable_offset=candidate.world_vtable - module_base,
        world_vtable_field_offset=candidate.world_vtable_field_offset,
        world_identity_kind=candidate.world_identity_kind,
        self_pointer_offset=candidate.self_pointer_offset,
        species_offset=candidate.species_offset,
        active_species_offset=candidate.active_species_offset,
        hp_offset=candidate.monster_hp_offset,
        x_offset=candidate.x_offset,
        y_offset=candidate.y_offset,
        z_offset=candidate.z_offset,
        movement_validated=True,
    )


def _independent_trace_recovery(
    discovery: TraceTargetDiscovery,
    *,
    module_base: int,
    configured_player_pointer_offset: int,
    configured_world_pointer_offset: int,
    expected_full_hp_by_species: tuple[tuple[int, int], ...],
    validated_candidates: int,
) -> PlayerPointerRecovery | None:
    """Activate the exact discovery contract used by the proven tester."""

    player = discovery.player
    if discovery.outcome != "success" or player is None or not discovery.monsters:
        return None
    if not player.direct_module_slots:
        return None
    configured_slot = int(module_base) + int(configured_player_pointer_offset)
    player_slot = min(
        player.direct_module_slots,
        key=lambda slot: (abs(int(slot) - configured_slot), int(slot)),
    )
    monster_layout = discovery.monsters[0]
    return PlayerPointerRecovery(
        player_pointer_address=int(player_slot),
        player_pointer_offset=int(player_slot) - int(module_base),
        player_base=int(player.base),
        world_base=0,
        world_pointer_address=None,
        world_pointer_offset=None,
        configured_player_pointer_offset=int(configured_player_pointer_offset),
        configured_world_pointer_offset=int(configured_world_pointer_offset),
        search_radius=0,
        validated_candidates=int(validated_candidates),
        strategy="anchored_independent",
        self_pointer_offset=min(player.self_pointer_offsets, default=None),
        species_offset=monster_layout.species_offset,
        active_species_offset=monster_layout.active_species_offset,
        hp_offset=monster_layout.hp_offset,
        x_offset=monster_layout.x_offset,
        y_offset=monster_layout.y_offset,
        z_offset=monster_layout.z_offset,
        movement_validated=True,
        independent_discovery=discovery,
        independent_expected_full_hp_by_species=tuple(
            sorted(
                (int(species), int(hp))
                for species, hp in expected_full_hp_by_species
                if int(species) > 0 and int(hp) > 0
            )
        ),
    )


def _independent_anchored_recovery(
    anchored: AnchoredDiscoveryResult,
    *,
    module_base: int,
    configured_player_pointer_offset: int,
    configured_world_pointer_offset: int,
    hints: PointerRecoveryHints,
    validated_candidates: int,
) -> PlayerPointerRecovery | None:
    discovery = trace_discovery_from_anchored(anchored)
    player = discovery.player
    if discovery.outcome != "success" or player is None or not discovery.monsters:
        return None
    configured_slot = int(module_base) + int(configured_player_pointer_offset)
    player_slot = min(
        player.direct_module_slots,
        key=lambda slot: (abs(int(slot) - configured_slot), int(slot)),
    )
    monster_layout = discovery.monsters[0]
    player_self_offset = min(player.self_pointer_offsets, default=None)
    return PlayerPointerRecovery(
        player_pointer_address=int(player_slot),
        player_pointer_offset=int(player_slot) - int(module_base),
        player_base=int(player.base),
        world_base=0,
        world_pointer_address=None,
        world_pointer_offset=None,
        configured_player_pointer_offset=int(configured_player_pointer_offset),
        configured_world_pointer_offset=int(configured_world_pointer_offset),
        search_radius=0,
        validated_candidates=int(validated_candidates),
        strategy="anchored_independent",
        self_pointer_offset=player_self_offset,
        species_offset=monster_layout.species_offset,
        active_species_offset=monster_layout.active_species_offset,
        hp_offset=monster_layout.hp_offset,
        x_offset=monster_layout.x_offset,
        y_offset=monster_layout.y_offset,
        z_offset=monster_layout.z_offset,
        movement_validated=True,
        independent_discovery=discovery,
        independent_expected_full_hp_by_species=tuple(
            sorted(
                (int(species), int(hp))
                for species, hp in hints.monster_hp_by_species
                if int(species) > 0 and int(hp) > 0
            )
        ),
    )


def _perform_recovery_attempt(
    memory: PointerRecoveryMemory,
    *,
    module_base: int,
    configured_player_pointer_offset: int,
    config: NativeMonsterConfig,
    search_radii: tuple[int, ...],
    chunk_size: int,
    maximum_candidates: int,
    stability_samples: int,
    stability_delay_seconds: float,
    metrics: _MetricsBuilder,
    control: _AttemptControl,
    status_callback: PointerRecoveryStatusCallback | None,
    hints: PointerRecoveryHints | None,
    pending_candidate: AnchoredPointerCandidate | None,
    allow_independent: bool,
) -> tuple[
    PlayerPointerRecovery | None,
    str,
    AnchoredPointerCandidate | None,
]:
    readable_regions_fn = cast(
        Callable[..., tuple[object, ...]] | None,
        getattr(memory, "readable_regions", None),
    )
    if not callable(readable_regions_fn):
        return None, "unavailable", None

    if pending_candidate is not None:
        if hints is None:
            return None, "anchor_hints_required", pending_candidate
        metrics.strategy = "anchored_movement_confirmation"
        confirmation = confirm_anchored_movement(
            cast(AnchoredDiscoveryMemory, cast(object, memory)),
            pending_candidate,
            hints,
            check=control.check,
        )
        _record_anchor_evidence(metrics, confirmation.evidence)
        _notify(
            status_callback,
            confirmation.outcome,
            confirmation.message,
            _progress_metrics(metrics, control),
        )
        if confirmation.outcome != "movement_confirmed":
            retained = (
                pending_candidate
                if confirmation.outcome == "movement_not_observed"
                else None
            )
            return None, confirmation.outcome, retained
        return (
            _anchored_recovery(
                pending_candidate,
                module_base=module_base,
                configured_player_pointer_offset=configured_player_pointer_offset,
                configured_world_pointer_offset=config.world_pointer_offset,
                validated_candidates=metrics.candidates_validated,
            ),
            "success",
            None,
        )

    control.check()
    configured_slot = int(module_base) + int(configured_player_pointer_offset)
    configured_world_hint = _read_configured_world_hint(memory, module_base, config)
    control.check()

    metrics.region_enumerations = 1
    try:
        regions = tuple(
            readable_regions_fn(
                maximum_address=config.maximum_scan_address,
                private_only=False,
            )
        )
    except Exception:
        return None, "region_error", None
    control.check()

    metrics.region_count = len(regions)
    region_index = _RegionIndex.build(regions)
    _notify(
        status_callback,
        "regions_indexed",
        f"Indexed {len(region_index.starts)} merged readable intervals.",
        _progress_metrics(metrics, control),
    )

    max_actor_span = max(
        config.self_pointer_offset,
        config.active_species_offset,
        config.hp_offset,
        config.z_offset,
    ) + 4
    radii = tuple(sorted({max(0x1000, int(value)) for value in search_radii}))
    module_extent = _module_image_extent(memory, config.module_name, module_base)
    if module_extent is not None:
        module_start, module_stop = module_extent
        metrics.strategy = "module_image"
        metrics.module_size = module_stop - module_start
        scan_passes = (("module_image", module_start, module_stop, metrics.module_size),)
    else:
        metrics.strategy = "configured_neighborhood"
        scan_passes = tuple(
            (
                f"configured_radius_0x{radius:X}",
                max(int(module_base), configured_slot - radius),
                configured_slot + radius + 4,
                radius,
            )
            for radius in radii
        )
    slot_refs: dict[int, list[int]] = {}
    validated: list[_ValidatedCandidate] = []
    inspected_pairs: set[tuple[int, int]] = set()
    previous_start: int | None = None
    previous_stop: int | None = None

    for strategy, scan_start, scan_stop, radius in scan_passes:
        control.check()
        metrics.radii_started += 1
        _notify(
            status_callback,
            "scanning",
            (
                f"Scanning {config.module_name} image "
                f"(0x{scan_start:X}-0x{scan_stop:X})."
                if strategy == "module_image"
                else f"Scanning new memory bands for radius 0x{radius:X}."
            ),
            _progress_metrics(metrics, control),
        )

        if previous_start is None or previous_stop is None:
            bands = ((scan_start, scan_stop),)
        else:
            pending: list[tuple[int, int]] = []
            if scan_start < previous_start:
                pending.append((scan_start, previous_start))
            if scan_stop > previous_stop:
                pending.append((previous_stop, scan_stop))
            bands = tuple(pending)

        for band_start, band_stop in bands:
            _scan_new_band(
                memory,
                start=band_start,
                stop=band_stop,
                chunk_size=chunk_size,
                region_index=region_index,
                slot_refs=slot_refs,
                metrics=metrics,
                control=control,
            )
        previous_start = scan_start
        previous_stop = scan_stop
        metrics.candidate_targets = len(slot_refs)
        metrics.candidate_slots = sum(len(slots) for slots in slot_refs.values())

        ordered_slots: list[tuple[int, int, int]] = []
        for target, slots in slot_refs.items():
            metrics.containment_checks += 1
            if not region_index.contains(target, max_actor_span):
                continue
            for slot in slots:
                ordered_slots.append(
                    (abs(slot - configured_slot), slot, target)
                )
        ordered_slots.sort(key=lambda item: item[0])

        for _distance, slot, target in ordered_slots[: max(1, maximum_candidates)]:
            control.check()
            pair = (slot, target)
            if pair in inspected_pairs:
                continue
            inspected_pairs.add(pair)
            metrics.candidates_validated += 1
            candidate = _validate_candidate(
                memory,
                slot_address=slot,
                target_address=target,
                reference_count=len(slot_refs.get(target, ())),
                configured_slot_address=configured_slot,
                module_base=module_base,
                config=config,
                configured_world_hint=configured_world_hint,
                slot_refs=slot_refs,
                metrics=metrics,
            )
            if candidate is not None:
                validated.append(candidate)
                if not candidate.player_like:
                    metrics.non_player_rejections += 1
                elif candidate.world_pointer_slot is None:
                    metrics.missing_world_slot_rejections += 1

        strong = [
            candidate
            for candidate in validated
            if (
                candidate.player_like
                and candidate.world_pointer_slot is not None
                and (
                    candidate.pair_matches
                    or candidate.configured_world_matches
                    or (
                        candidate.reference_count == 1
                        and candidate.world_reference_count == 1
                    )
                )
            )
        ]
        strong.sort(
            key=lambda candidate: (
                int(candidate.pair_matches),
                int(candidate.configured_world_matches),
                min(candidate.world_reference_count, 8),
                min(candidate.reference_count, 8),
                -candidate.distance,
            ),
            reverse=True,
        )
        metrics.radii_completed += 1
        if not strong:
            _notify(
                status_callback,
                "radius_complete",
                (
                    "No strongly validated player/world pair in the module image."
                    if strategy == "module_image"
                    else f"No validated replacement inside radius 0x{radius:X}."
                ),
                _progress_metrics(metrics, control),
            )
            continue

        selected = strong[0]
        selected_evidence = (
            selected.pair_matches,
            selected.configured_world_matches,
            min(selected.world_reference_count, 8),
            min(selected.reference_count, 8),
        )
        competing_targets = {
            candidate.target_address
            for candidate in strong[1:]
            if (
                candidate.pair_matches,
                candidate.configured_world_matches,
                min(candidate.world_reference_count, 8),
                min(candidate.reference_count, 8),
            )
            == selected_evidence
        }
        competing_targets.discard(selected.target_address)
        if competing_targets:
            metrics.ambiguous_candidates = len(competing_targets) + 1
            _notify(
                status_callback,
                "ambiguous",
                "Multiple equally supported player candidates were rejected.",
                _progress_metrics(metrics, control),
            )
            continue
        if not _stable_candidate(
            memory,
            selected,
            config,
            samples=stability_samples,
            delay_seconds=stability_delay_seconds,
            control=control,
        ):
            metrics.unstable_rejections += 1
            _notify(
                status_callback,
                "radius_complete",
                f"Candidate inside radius 0x{radius:X} was not stable.",
                _progress_metrics(metrics, control),
            )
            continue

        player_offset = selected.slot_address - int(module_base)
        world_slot = selected.world_pointer_slot
        world_offset = (
            None if world_slot is None else world_slot - int(module_base)
        )
        return (
            PlayerPointerRecovery(
                player_pointer_address=selected.slot_address,
                player_pointer_offset=player_offset,
                player_base=selected.target_address,
                world_base=selected.world_base,
                world_pointer_address=world_slot,
                world_pointer_offset=world_offset,
                configured_player_pointer_offset=int(
                    configured_player_pointer_offset
                ),
                configured_world_pointer_offset=int(config.world_pointer_offset),
                search_radius=radius,
                validated_candidates=len(validated),
                strategy=strategy,
            ),
            "success",
            None,
        )

    if hints is None:
        return None, "not_found", None
    if module_extent is None:
        return None, "anchor_module_metadata_required", None
    if hints.require_verified_monster_hp and not hints.monster_hp_by_species:
        _notify(
            status_callback,
            "monster_hp_anchor_required",
            (
                "The selected monster set has no trusted full-HP recovery anchor; "
                "refusing to reuse the configured monster HP field."
            ),
            _progress_metrics(metrics, control),
        )
        return None, "monster_hp_anchor_required", None
    exact_species_hp = tuple(
        sorted(
            (int(species), int(hp))
            for species, hp in hints.monster_hp_by_species
            if int(species) > 0 and int(hp) > 0
        )
    )
    if allow_independent and hints.exact_player_hp_available and exact_species_hp:
        module_info_fn = cast(
            Callable[[str], object] | None,
            getattr(memory, "module_info", None),
        )
        if not callable(module_info_fn):
            return None, "anchor_module_metadata_required", None
        try:
            module_info = module_info_fn(config.module_name)
        except Exception:
            return None, "anchor_module_metadata_required", None
        metrics.strategy = "validated_trace_independent"
        _notify(
            status_callback,
            "trace_target_scanning",
            (
                "Running the exact independent target discovery path used by "
                "the validated pointer tester; "
                f"species_hp={exact_species_hp}, "
                f"spawn=({hints.player_spawn_x}, {hints.player_spawn_z}), "
                f"player_hp={hints.player_current_hp}."
            ),
            _progress_metrics(metrics, control),
        )

        def trace_progress(
            bytes_scanned: int,
            regions_scanned: int,
            counts: dict[str, int],
        ) -> None:
            metrics.anchor_bytes_scanned = int(bytes_scanned)
            metrics.anchor_regions_scanned = int(regions_scanned)
            metrics.species_value_matches = int(counts.get("species_hits", 0))
            metrics.spawn_x_matches = int(counts.get("spawn_x", 0))
            _notify(
                status_callback,
                "trace_target_scanning",
                (
                    f"Validated target scan read {bytes_scanned / (1 << 20):.0f} "
                    f"MiB; species_hits={counts.get('species_hits', 0)}, "
                    f"spawn_hits={counts.get('spawn_x', 0)}."
                ),
                _progress_metrics(metrics, control),
            )

        discovery = discover_trace_targets(
            cast(object, memory),
            regions=regions,
            readable=ReadableRegionIndex.build(regions),
            module=cast(object, module_info),
            species_hp=dict(exact_species_hp),
            spawn_x=float(hints.player_spawn_x),
            spawn_z=float(hints.player_spawn_z),
            player_hp=int(hints.player_current_hp),
            species_offset=config.species_offset,
            active_species_offset=config.active_species_offset,
            hp_offset=config.hp_offset,
            x_offset=config.x_offset,
            y_offset=config.y_offset,
            z_offset=config.z_offset,
            self_pointer_offset=config.self_pointer_offset,
            coordinate_limit=config.maximum_absolute_coordinate,
            chunk_size=max(config.discovery_chunk_bytes, chunk_size),
            maximum_monster_candidates=max(512, int(maximum_candidates)),
            stability_samples=stability_samples,
            stability_delay_seconds=stability_delay_seconds,
            check=control.check,
            progress=trace_progress,
        )
        _record_trace_evidence(metrics, discovery.evidence)
        _notify(
            status_callback,
            discovery.outcome,
            discovery.message,
            _progress_metrics(metrics, control),
        )
        exact_recovery = _independent_trace_recovery(
            discovery,
            module_base=module_base,
            configured_player_pointer_offset=configured_player_pointer_offset,
            configured_world_pointer_offset=config.world_pointer_offset,
            expected_full_hp_by_species=exact_species_hp,
            validated_candidates=metrics.candidates_validated,
        )
        if exact_recovery is None:
            # Exact anchors were available, so a structurally looser fallback
            # must not silently replace the tested discovery contract.
            return None, discovery.outcome, None
        metrics.strategy = "anchored_independent"
        _notify(
            status_callback,
            "independent_ready",
            (
                "Validated tester discovery produced the production player and "
                "live actor cohort; world pointer is not required."
            ),
            _progress_metrics(metrics, control),
        )
        return exact_recovery, "success", None

    metrics.strategy = "known_species_spawn_anchor"
    _notify(
        status_callback,
        "anchor_scanning",
        (
            "Scanning private memory for anchored actor layout; "
            f"species={hints.known_species_ids}, "
            f"monster_hp_anchors={hints.monster_hp_by_species}, "
            f"spawn=({hints.player_spawn_x}, {hints.player_spawn_z}), "
            + (
                f"player_hp={hints.player_current_hp}/{hints.player_max_hp}."
                if hints.exact_player_hp_available
                else "player_hp=unavailable (structural fallback enabled)."
            )
        ),
        _progress_metrics(metrics, control),
    )

    def anchor_progress(evidence: AnchoredDiscoveryEvidence) -> None:
        _record_anchor_evidence(metrics, evidence)
        _notify(
            status_callback,
            "anchor_scanning",
            (
                f"Anchor scan read {evidence.anchor_bytes_scanned / (1 << 20):.0f} "
                f"MiB; species_hits={evidence.species_value_matches}, "
                f"spawn_hits={evidence.spawn_x_matches}."
            ),
            _progress_metrics(metrics, control),
        )

    anchored = discover_anchored_pointer_candidate(
        cast(AnchoredDiscoveryMemory, cast(object, memory)),
        regions=regions,
        slot_refs=slot_refs,
        hints=hints,
        module_base=module_extent[0],
        module_stop=module_extent[1],
        configured_player_slot=configured_slot,
        configured_world_slot=module_base + config.world_pointer_offset,
        species_offset=config.species_offset,
        active_species_offset=config.active_species_offset,
        hp_offset=config.hp_offset,
        x_offset=config.x_offset,
        y_offset=config.y_offset,
        z_offset=config.z_offset,
        configured_world_field_offset=config.world_offset,
        configured_self_offset=config.self_pointer_offset,
        coordinate_limit=config.maximum_absolute_coordinate,
        maximum_address=config.maximum_scan_address,
        chunk_size=max(config.discovery_chunk_bytes, chunk_size),
        cancellation=control.cancellation,
        deadline=control.deadline,
        readable_contains=region_index.contains,
        check=control.check,
        stability_samples=stability_samples,
        stability_delay_seconds=stability_delay_seconds,
        progress_callback=anchor_progress,
    )
    _record_anchor_evidence(metrics, anchored.evidence)
    _notify(
        status_callback,
        anchored.outcome,
        anchored.message,
        _progress_metrics(metrics, control),
    )
    independent = (
        _independent_anchored_recovery(
            anchored,
            module_base=module_base,
            configured_player_pointer_offset=configured_player_pointer_offset,
            configured_world_pointer_offset=config.world_pointer_offset,
            hints=hints,
            validated_candidates=metrics.candidates_validated,
        )
        if allow_independent
        else None
    )
    if independent is not None:
        metrics.strategy = "anchored_independent"
        _notify(
            status_callback,
            "independent_ready",
            (
                "Anchored player and monster cohort are ready without a "
                "world pointer; production will use direct module/player aliases "
                "and cached actor slots."
            ),
            _progress_metrics(metrics, control),
        )
        return independent, "success", None
    return None, anchored.outcome, anchored.candidate


def recover_local_player_pointer(
    memory: PointerRecoveryMemory,
    *,
    module_base: int,
    configured_player_pointer_offset: int,
    configured_world_pointer_offset: int | None = None,
    state: PointerRecoveryState | None = None,
    monster_config: NativeMonsterConfig | None = None,
    search_radii: tuple[int, ...] = (0x20000, 0x100000, 0x400000, 0x800000),
    chunk_size: int = 0x10000,
    maximum_candidates: int = 4096,
    persist: bool = False,
    position_config_path: str | Path | None = None,
    monster_config_path: str | Path | None = None,
    cancellation: object | None = None,
    deadline: float | None = None,
    timeout_seconds: float = DEFAULT_RECOVERY_TIMEOUT_SECONDS,
    negative_cooldown_seconds: float = MINIMUM_NEGATIVE_COOLDOWN_SECONDS,
    status_callback: PointerRecoveryStatusCallback | None = None,
    clock: Callable[[], float] = monotonic,
    stability_samples: int = 3,
    stability_delay_seconds: float = 0.03,
    hints: PointerRecoveryHints | None = None,
    allow_independent: bool = False,
) -> PlayerPointerRecovery | None:
    """Explicitly recover a shifted Neuz local-player global.

    Ordinary position and monster reads never call this function. Explicit
    attempts sharing a :class:`PointerRecoveryState` are single-flight per
    ``(pid, module_base)``, enumerate and index readable regions once, scan each
    expanding memory band at most once, and stop cooperatively at the deadline
    or cancellation boundary. Failed complete attempts enter an
    attachment-scoped cooldown to prevent retry storms.

    Deadline checks surround each synchronous backend call and run throughout
    CPU scanning. Python cannot preempt a backend call already in progress, so
    ``PointerRecoveryMetrics.deadline_is_cooperative`` is explicit and callers
    must keep this diagnostic behind a bounded lifecycle owner.
    """

    recovery_state = state or PointerRecoveryState()
    base_config = monster_config or load_native_monster_config()
    world_hint_offset = (
        base_config.world_pointer_hint_offset
        if configured_world_pointer_offset is None
        else int(configured_world_pointer_offset)
    )
    config = replace(
        base_config,
        player_pointer_offset=int(configured_player_pointer_offset),
        world_pointer_offset=world_hint_offset,
    )
    pid = int(getattr(memory, "pid", 0))
    cache_key = (pid, int(module_base))
    started_at = clock()
    bounded_deadline = started_at + max(0.0, float(timeout_seconds))
    if deadline is not None:
        bounded_deadline = min(bounded_deadline, float(deadline))
    control = _AttemptControl(
        cancellation=cancellation,
        deadline=bounded_deadline,
        clock=clock,
    )
    builder = _MetricsBuilder(
        pid=pid,
        module_base=int(module_base),
        started_at=started_at,
    )
    minimum_cooldown = max(
        MINIMUM_NEGATIVE_COOLDOWN_SECONDS,
        float(negative_cooldown_seconds),
    )

    _notify(
        status_callback,
        "started",
        "Explicit native pointer recovery started.",
        _progress_metrics(builder, control),
    )

    while True:
        try:
            control.check()
        except _AttemptStopped as stopped:
            metrics = builder.freeze(stopped.outcome, now=clock())
            with recovery_state.lock:
                recovery_state.last_metrics[cache_key] = metrics
            _notify(
                status_callback,
                stopped.outcome,
                f"Pointer recovery stopped: {stopped.outcome}.",
                metrics,
            )
            return None

        now = clock()
        with recovery_state.lock:
            cached = recovery_state.recovery_cache.get(cache_key)
            cooldown_until = recovery_state.negative_cache.get(cache_key)
            flight = recovery_state.inflight.get(cache_key)
            if cooldown_until is not None and cooldown_until <= now:
                recovery_state.negative_cache.pop(cache_key, None)
                cooldown_until = None
            if cached is None and cooldown_until is None and flight is None:
                flight = _RecoveryFlight(persist_requested=bool(persist))
                recovery_state.inflight[cache_key] = flight
                owner = True
            else:
                owner = False
                if (
                    persist
                    and cached is None
                    and cooldown_until is None
                    and flight is not None
                ):
                    flight.persist_requested = True

        if cached is not None:
            try:
                control.check()
                cached_is_valid = _verify_cached(
                    memory,
                    cached,
                    config,
                    module_base,
                )
                control.check()
            except _AttemptStopped as stopped:
                metrics = builder.freeze(stopped.outcome, now=clock())
                with recovery_state.lock:
                    recovery_state.last_metrics[cache_key] = metrics
                _notify(
                    status_callback,
                    stopped.outcome,
                    f"Cached pointer verification stopped: {stopped.outcome}.",
                    metrics,
                )
                return None
            if cached_is_valid:
                _persist_if_requested(
                    cached,
                    requested=bool(persist),
                    position_config_path=position_config_path,
                    monster_config_path=monster_config_path,
                )
                metrics = builder.freeze(
                    "cache_hit",
                    now=clock(),
                    cache_hit=True,
                )
                with recovery_state.lock:
                    recovery_state.last_metrics[cache_key] = metrics
                _notify(
                    status_callback,
                    "cache_hit",
                    "Verified cached native pointer recovery.",
                    metrics,
                )
                return cached
            with recovery_state.lock:
                if recovery_state.recovery_cache.get(cache_key) is cached:
                    recovery_state.recovery_cache.pop(cache_key, None)
            continue

        if cooldown_until is not None:
            remaining = max(0.0, cooldown_until - now)
            metrics = builder.freeze(
                "negative_cache",
                now=clock(),
                negative_cache_hit=True,
                cooldown_remaining_seconds=remaining,
            )
            with recovery_state.lock:
                recovery_state.last_metrics[cache_key] = metrics
            _notify(
                status_callback,
                "negative_cache",
                f"Recovery cooldown active for {remaining:.3f}s.",
                metrics,
            )
            return None

        assert flight is not None
        if owner:
            break

        _notify(
            status_callback,
            "waiting_for_inflight",
            "Waiting for the in-flight recovery for this process/module.",
            _progress_metrics(builder, control),
        )
        try:
            while not flight.completed.is_set():
                control.check()
                remaining = max(0.0, control.deadline - clock())
                flight.completed.wait(min(0.02, remaining))
        except _AttemptStopped as stopped:
            metrics = builder.freeze(
                stopped.outcome,
                now=clock(),
                joined_existing_attempt=True,
            )
            with recovery_state.lock:
                recovery_state.last_metrics[cache_key] = metrics
            _notify(
                status_callback,
                stopped.outcome,
                f"Stopped waiting for recovery: {stopped.outcome}.",
                metrics,
            )
            return None

        shared = flight.metrics
        if shared is None:
            shared = builder.freeze("shared_unknown", now=clock())
        metrics = replace(
            shared,
            elapsed_seconds=max(0.0, clock() - started_at),
            joined_existing_attempt=True,
        )
        with recovery_state.lock:
            recovery_state.last_metrics[cache_key] = metrics
        _notify(
            status_callback,
            "inflight_complete",
            f"In-flight recovery completed with outcome {metrics.outcome}.",
            metrics,
        )
        return flight.result

    recovery: PlayerPointerRecovery | None = None
    with recovery_state.lock:
        pending_candidate = recovery_state.pending_candidates.get(cache_key)
    next_pending: AnchoredPointerCandidate | None = pending_candidate
    outcome = "not_found"
    try:
        recovery, outcome, next_pending = _perform_recovery_attempt(
            memory,
            module_base=int(module_base),
            configured_player_pointer_offset=int(
                configured_player_pointer_offset
            ),
            config=config,
            search_radii=search_radii,
            chunk_size=chunk_size,
            maximum_candidates=maximum_candidates,
            stability_samples=stability_samples,
            stability_delay_seconds=stability_delay_seconds,
            metrics=builder,
            control=control,
            status_callback=status_callback,
            hints=hints,
            pending_candidate=pending_candidate,
            allow_independent=bool(allow_independent),
        )
    except _AttemptStopped as stopped:
        outcome = stopped.outcome
    except Exception:
        outcome = "error"

    completed_at = clock()
    metrics = builder.freeze(outcome, now=completed_at)
    with recovery_state.lock:
        if recovery is not None:
            recovery_state.recovery_cache[cache_key] = recovery
            recovery_state.negative_cache.pop(cache_key, None)
            recovery_state.pending_candidates.pop(cache_key, None)
        elif next_pending is not None:
            recovery_state.pending_candidates[cache_key] = next_pending
            recovery_state.negative_cache.pop(cache_key, None)
        else:
            recovery_state.pending_candidates.pop(cache_key, None)
        if recovery is None and outcome in {"not_found", "deadline"}:
            recovery_state.negative_cache[cache_key] = (
                completed_at + minimum_cooldown
            )
        flight.result = recovery
        flight.metrics = metrics
        recovery_state.last_metrics[cache_key] = metrics
        if recovery_state.inflight.get(cache_key) is flight:
            recovery_state.inflight.pop(cache_key, None)
        persist_for_flight = bool(
            recovery is not None and flight.persist_requested
        )

    if recovery is None:
        flight.completed.set()
        _notify(
            status_callback,
            outcome,
            (
                f"Pointer recovery completed with outcome {outcome}; "
                f"strategy={metrics.strategy}, slots={metrics.candidate_slots}, "
                f"validated={metrics.candidates_validated}, "
                f"self_mismatch={metrics.self_mismatch_rejections}, "
                f"self_near_probed={metrics.self_mismatch_near_probed}, "
                f"self_near_world={metrics.self_mismatch_world_nonzero}, "
                f"self_near_world_ref={metrics.self_mismatch_world_module_ref}, "
                f"self_near_coords={metrics.self_mismatch_coordinate_plausible}, "
                f"self_near_hp={metrics.self_mismatch_hp_positive}, "
                f"self_near_player={metrics.self_mismatch_player_like}, "
                f"self_near_full={metrics.self_mismatch_full_near_matches}, "
                f"world_null={metrics.world_null_rejections}, "
                f"coordinate_invalid={metrics.coordinate_rejections}, "
                f"hp_invalid={metrics.hp_rejections}, "
                f"non_player={metrics.non_player_rejections}, "
                f"missing_world_slot={metrics.missing_world_slot_rejections}, "
                f"unstable={metrics.unstable_rejections}, "
                f"ambiguous={metrics.ambiguous_candidates}."
                f" anchor_bytes={metrics.anchor_bytes_scanned},"
                f" anchor_regions={metrics.anchor_regions_scanned},"
                f" anchor_read_failures={metrics.anchor_read_failures},"
                f" species_matches={metrics.species_value_matches},"
                f" monsters={metrics.monster_candidates},"
                f" monster_bases={metrics.monster_base_hypotheses},"
                f" layout_species={metrics.monster_layout_species_support},"
                f" layout_ties={metrics.monster_layout_ties},"
                f" self_aliases={metrics.monster_self_field_aliases},"
                f" monster_species_reject={metrics.monster_species_rejections},"
                f" monster_active_reject={metrics.monster_active_rejections},"
                f" monster_hp_reject={metrics.monster_hp_rejections},"
                f" monster_coord_reject={metrics.monster_coordinate_rejections},"
                f" world_support={metrics.inferred_world_actor_support},"
                f" world_vtable={None if metrics.inferred_world_vtable is None else f'0x{metrics.inferred_world_vtable:X}'},"
                f" world_vtable_field={None if metrics.inferred_world_vtable_field_offset is None else f'0x{metrics.inferred_world_vtable_field_offset:X}'},"
                f" world_identity_kind={metrics.inferred_world_identity_kind},"
                f" world_hypotheses={metrics.structural_world_hypotheses},"
                f" world_selected_structure=(readable_ptrs={metrics.inferred_world_readable_pointer_fields},"
                f"distinct={metrics.inferred_world_distinct_values}),"
                f" world_object_reject={metrics.world_object_rejections},"
                f" world_identity=(candidates={metrics.world_identity_candidates},"
                f"span_reject={metrics.world_identity_span_rejections},"
                f"vtable_miss={metrics.world_identity_vtable_misses},"
                f"unstable={metrics.world_identity_unstable_rejections},"
                f"module_fields={metrics.world_identity_module_pointer_fields}),"
                f" world_structure=(readable_ptrs={metrics.world_identity_readable_pointer_fields},"
                f"distinct={metrics.world_identity_distinct_values},"
                f"structural_reject={metrics.world_identity_structural_rejections},"
                f"marker_accept={metrics.world_identity_marker_accepts}),"
                f" world_near=(actors={metrics.world_near_actor_support},"
                f"species={metrics.world_near_species_support},"
                f"module_refs={metrics.world_near_module_references},"
                f"actor_field={None if metrics.world_near_field_offset is None else f'0x{metrics.world_near_field_offset:X}'},"
                f"target={None if metrics.world_near_target is None else f'0x{metrics.world_near_target:X}'},"
                f"module_fields={metrics.world_near_module_pointer_fields}),"
                f" world_near_structure=(readable_ptrs={metrics.world_near_readable_pointer_fields},"
                f"distinct={metrics.world_near_distinct_values}),"
                f" self_support={metrics.inferred_self_actor_support},"
                f" species_field={metrics.inferred_species_offset},"
                f" active_field={metrics.inferred_active_species_offset},"
                f" hp_field={metrics.inferred_hp_offset},"
                f" player_hp_fields=({metrics.inferred_player_hp_offset},"
                f"{metrics.inferred_player_max_hp_offset}),"
                f" xyz_fields=({metrics.inferred_x_offset},"
                f"{metrics.inferred_y_offset},{metrics.inferred_z_offset}),"
                f" spawn_structures={metrics.spawn_structure_candidates},"
                f" spawn_world={metrics.spawn_world_matches},"
                f" spawn_world_hypotheses={metrics.spawn_world_hypothesis_matches},"
                f" spawn_hp={metrics.spawn_hp_matches},"
                f" spawn_players={metrics.spawn_player_matches},"
                f" stable_spawn={metrics.stable_spawn_candidates},"
                f" player_slots={metrics.direct_player_slot_candidates},"
                f" player_chains={metrics.player_chain_candidates},"
                f" player_refs={metrics.player_reference_matches},"
                f" player_world_chains={metrics.player_world_chain_candidates},"
                f" player_world_rooted={metrics.player_world_rooted_matches},"
                f" player_ref_ambiguous={metrics.player_reference_ambiguities},"
                f" movement={metrics.movement_observed}."
            ),
            metrics,
        )
        return None

    _persist_if_requested(
        recovery,
        requested=persist_for_flight,
        position_config_path=position_config_path,
        monster_config_path=monster_config_path,
    )
    flight.completed.set()
    _notify(
        status_callback,
        "success",
        (
            "Native player pointer recovered and strongly validated; "
            f"strategy={metrics.strategy}, module_size=0x{metrics.module_size:X}."
        ),
        metrics,
    )
    print(
        "Native player pointer recovered | "
        f"player=0x{configured_player_pointer_offset:X}->"
        f"0x{recovery.player_pointer_offset:X} "
        + (
            "world=unchanged/unknown "
            if recovery.world_pointer_offset is None
            else (
                f"world=0x{config.world_pointer_offset:X}->"
                f"0x{recovery.world_pointer_offset:X} "
            )
        )
        + (
            f"radius=0x{recovery.search_radius:X} "
            f"candidates={recovery.validated_candidates}"
        )
    )
    return recovery
