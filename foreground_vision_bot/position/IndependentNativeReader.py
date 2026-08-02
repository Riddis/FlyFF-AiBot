from __future__ import annotations

import math
import struct
from collections import Counter
from dataclasses import asdict, dataclass
from threading import RLock
from time import monotonic
from typing import Callable, Iterable, Mapping, Protocol

from .AuthoritativeActorDiscovery import (
    AuthoritativeActorDiscovery,
    AuthoritativeActorRefresh,
    ActiveFieldCandidate,
    discover_authoritative_actors,
    refresh_authoritative_actors,
)
from .NativeTraceTargets import (
    TraceMonsterTarget,
    TracePlayerTarget,
    TraceTargetDiscovery,
)
from .Win32ProcessMemory import ModuleInfo


class IndependentReaderMemory(Protocol):
    def read(self, address: int, size: int) -> bytes: ...


class IndependentNativeReadError(RuntimeError):
    """A recovered player alias or actor slot could not be read safely."""


@dataclass(frozen=True, slots=True)
class IndependentPlayerRead:
    base: int
    pointer_slot: int
    hp: int
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class IndependentActorSlotRead:
    base: int
    species: int
    hp: int
    x: float
    y: float
    z: float
    active_species: int
    active_matches_species: bool
    target_species: bool
    state: str
    distance_native: float
    hp_offset: int = 0
    hp_candidates: tuple[tuple[int, int], ...] = ()
    hp_offset_validated: bool = True


@dataclass(frozen=True, slots=True)
class IndependentMonsterRead:
    base: int
    species: int
    hp: int
    full_hp: int
    x: float
    y: float
    z: float
    active_species: int
    distance_native: float
    hp_offset: int = 0
    hp_candidates: tuple[tuple[int, int], ...] = ()
    hp_offset_validated: bool = True

    @property
    def damaged(self) -> bool:
        return 0 < self.hp < self.full_hp


@dataclass(frozen=True, slots=True)
class IndependentNativeSnapshot:
    player: IndependentPlayerRead
    monsters: tuple[IndependentMonsterRead, ...]
    actor_states: tuple[IndependentActorSlotRead, ...]
    living_actor_bases: tuple[int, ...]
    cached_actor_slots: int
    occupied_slots: int
    tracked_species_slots: int
    instantiated_monsters: int
    living_monsters: int
    visible_living_monsters: int
    damaged_monsters: int
    zero_hp_monsters: int
    other_species_slots: int
    empty_slots: int
    invalid_hp_slots: int
    active_field_mismatches: int
    active_field_reliable: bool
    dead_or_dormant_slots: int
    unreadable_slots: int
    actor_stride: int | None
    monster_hp_offset: int = 0
    monster_hp_candidate_offsets: tuple[int, ...] = ()
    monster_hp_offset_validated: bool = True
    monster_hp_transition_support: tuple[tuple[int, int], ...] = ()
    actor_source: str = "bounded_slab_fallback"
    authoritative_relation_offset: int | None = None
    authoritative_relation_value: int | None = None
    authoritative_relation_validated: bool = False
    authoritative_species_counts: tuple[tuple[int, int], ...] = ()
    authoritative_refreshes: int = 0
    authoritative_refresh_failures: int = 0
    authoritative_last_error: str | None = None
    active_species_offset: int | None = None
    active_species_validated: bool = False
    active_species_candidates: tuple[tuple[int, int, int, int, int, bool], ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        # Actor states are used for high-frequency lifecycle tracking. Keeping all
        # of them in every 0.5-second report sample makes reports unnecessarily
        # huge, so event records carry only changed slots instead.
        payload.pop("actor_states", None)
        return payload


@dataclass(frozen=True, slots=True)
class ActorCacheMergeResult:
    anchors_seen: int
    new_anchors: int
    new_slots: int
    total_anchors: int
    total_slots: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ActorSlotProbeResult:
    probed_slots: int
    promoted_slots: int
    cached_slots: int
    pending_slots: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _MonsterScan:
    visible_living: tuple[IndependentMonsterRead, ...]
    actor_states: tuple[IndependentActorSlotRead, ...]
    living_bases: tuple[int, ...]
    occupied: int
    tracked: int
    instantiated: int
    living: int
    damaged: int
    zero_hp: int
    other_species: int
    empty: int
    invalid_hp: int
    active_mismatches: int
    dead_or_dormant: int
    unreadable: int


def _u32(data: bytes, offset: int = 0) -> int:
    return int(struct.unpack_from("<I", data, offset)[0])


def _i32(data: bytes, offset: int = 0) -> int:
    return int(struct.unpack_from("<i", data, offset)[0])


def _f32(data: bytes, offset: int = 0) -> float:
    return float(struct.unpack_from("<f", data, offset)[0])


def infer_actor_stride(bases: tuple[int, ...]) -> int | None:
    """Infer the repeated actor-slot stride from exact monster anchors."""

    ordered = tuple(sorted(set(int(value) for value in bases if int(value) > 0)))
    if len(ordered) < 3:
        return None
    counts: Counter[int] = Counter()
    # Nearby anchors from one slab may have missing slots, so count a bounded
    # set of pair distances and prefer the smallest well-supported divisor.
    for index, base in enumerate(ordered):
        for other in ordered[index + 1 : index + 9]:
            delta = other - base
            if delta <= 0 or delta > 0x20000 or delta % 4:
                continue
            counts[delta] += 1
    if not counts:
        return None
    candidates = sorted(
        counts,
        key=lambda delta: (-counts[delta], delta),
    )
    for candidate in candidates:
        # A direct adjacent-slot stride should divide several larger observed
        # distances even when not every slot was full-health during discovery.
        support = sum(count for delta, count in counts.items() if delta % candidate == 0)
        if counts[candidate] >= 2 or support >= 4:
            return candidate
    return candidates[0]


class IndependentNativeReader:
    """Read the proven player and actor cohort without any world pointer.

    Discovery remains expensive and is performed once by ``discover_trace_targets``.
    This class then performs only direct module-slot and cached actor-slot reads.
    It never attaches a debugger and never writes process memory.

    Player and monster HP offsets are discovered independently, but in the
    current client both resolve to +0x81C. For monsters the exact full-health
    value is a discovery-time anchor at the live HP field, not a separate max-HP
    field. A dead actor remains in its reusable slot with HP zero until that slot
    is populated again, possibly by another species.

    The historical active/loaded fields are diagnostic only in independent
    mode and never gate actor presence. In the validated layout, the apparent
    dynamic +0x217C field is the next slot's +0x174 species value, while the old
    configured +0x1DBC field is zero for valid live actors. Species, self alias,
    coordinates, and live HP are sufficient to evaluate every bounded slab slot.
    """

    def __init__(
        self,
        memory: IndependentReaderMemory,
        module: ModuleInfo,
        discovery: TraceTargetDiscovery,
        *,
        configured_player_offset: int | None = None,
        monster_current_hp_offset: int | None = None,
        monster_active_species_offset: int | None = None,
        current_hp_offset: int | None = None,
        expected_full_hp_by_species: Mapping[int, int] | None = None,
        object_span: int = 0x4000,
        slots_each_direction: int = 0,
        selected_species_ids: Iterable[int] | None = None,
        maximum_scan_address: int = 0x7FFFFFFF,
        private_memory_only: bool = True,
        discovery_chunk_bytes: int = 1 << 20,
        coordinate_limit: float = 100_000.0,
        authoritative_refresh_interval_seconds: float = 20.0,
        cancellation: object | None = None,
        deadline: float | None = None,
        status_callback: Callable[[str], None] | None = None,
        restored_authoritative: AuthoritativeActorRefresh | None = None,
        restored_relation_offset: int | None = None,
        restored_relation_value: int | None = None,
        known_actor_stride: int | None = None,
    ) -> None:
        if discovery.outcome != "success" or discovery.player is None:
            raise ValueError("successful target discovery with one player is required")
        if not discovery.monsters:
            raise ValueError("at least one validated monster anchor is required")
        self._memory = memory
        self.module = module
        self._trace_discovery = discovery
        self.player_target = discovery.player
        self.monster_targets = discovery.monsters
        self.object_span = max(0x4000, int(object_span))
        self.player_hp_offset = int(discovery.player.hp_offset)
        self.monster_hp_offset = int(discovery.monsters[0].hp_offset)
        # Compatibility aliases. They intentionally point at the same live HP
        # field; there is no independently proven monster max-HP field.
        self.monster_current_hp_offset = self.monster_hp_offset
        self.monster_full_hp_offset = self.monster_hp_offset
        self.configured_monster_hp_offset = int(
            0x814 if monster_current_hp_offset is None else monster_current_hp_offset
        )
        self.monster_active_species_offset = int(
            0x1DBC
            if monster_active_species_offset is None
            else monster_active_species_offset
        )
        legacy_current = None if current_hp_offset is None else int(current_hp_offset)
        for name, value in (
            ("player_hp_offset", self.player_hp_offset),
            ("monster_hp_offset", self.monster_hp_offset),
            ("configured_monster_hp_offset", self.configured_monster_hp_offset),
            ("monster_active_species_offset", self.monster_active_species_offset),
        ):
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        # Kept only so old commands fail softly. It is diagnostic only and does
        # not override either dynamically discovered live HP field.
        self.deprecated_current_hp_offset = legacy_current
        self.expected_full_hp_by_species = {
            int(species): int(hp)
            for species, hp in (expected_full_hp_by_species or {}).items()
            if int(species) > 0 and int(hp) > 0
        }
        inferred_stride = infer_actor_stride(
            tuple(item.base for item in discovery.monsters)
        )
        self.actor_stride = (
            int(known_actor_stride)
            if known_actor_stride is not None and int(known_actor_stride) > 0
            else inferred_stride
        )
        self._player_slots = self._rank_player_slots(
            discovery.player,
            configured_player_offset=configured_player_offset,
        )
        if not self._player_slots:
            raise IndependentNativeReadError(
                "No direct module alias still resolves to the validated player"
            )
        self._actor_self_offsets = tuple(
            dict.fromkeys(
                offset
                for monster in discovery.monsters
                for offset in monster.self_pointer_offsets
            )
        )
        self._slots_each_direction = max(0, int(slots_each_direction))
        self._cache_lock = RLock()
        self._selected_species_ids = {
            int(value)
            for value in (selected_species_ids or ())
            if int(value) > 0
        }
        self._selected_species_ids.update(
            int(item.species) for item in discovery.monsters
        )
        self._maximum_scan_address = int(maximum_scan_address)
        self._private_memory_only = bool(private_memory_only)
        self._discovery_chunk_bytes = max(4096, int(discovery_chunk_bytes))
        self._coordinate_limit = float(coordinate_limit)
        self._authoritative_refresh_interval_seconds = max(
            0.0, float(authoritative_refresh_interval_seconds)
        )
        self._last_authoritative_refresh_at = -math.inf
        self._authoritative_refreshes = 0
        self._authoritative_refresh_failures = 0
        self._authoritative_last_error: str | None = None
        self._authoritative_discovery: AuthoritativeActorDiscovery | None = None
        self._authoritative_relation_offset: int | None = None
        self._authoritative_relation_value: int | None = None
        self._authoritative_species_counts: tuple[tuple[int, int], ...] = ()
        self._authoritative_active_candidates: tuple[ActiveFieldCandidate, ...] = ()
        self._active_species_offset: int | None = None
        self._active_species_validated = False
        self._active_runtime_support: dict[int, list[int]] = {}
        self._actor_source = "bounded_slab_fallback"

        fallback_slots = self._build_actor_slot_cache(
            slots_each_direction=self._slots_each_direction
        )
        self._actor_slots = fallback_slots
        finder = getattr(self._memory, "find_u32", None)
        if (
            restored_authoritative is not None
            and restored_relation_offset is not None
            and restored_relation_value is not None
            and restored_authoritative.actor_bases
        ):
            self._actor_source = "authoritative_global"
            self._actor_slots = tuple(restored_authoritative.actor_bases)
            self._authoritative_relation_offset = int(restored_relation_offset)
            self._authoritative_relation_value = int(restored_relation_value)
            self._authoritative_species_counts = tuple(
                restored_authoritative.species_counts
            )
            self._authoritative_active_candidates = tuple(
                restored_authoritative.active_candidates
            )
            self._active_species_offset = restored_authoritative.active_species_offset
            self._active_species_validated = bool(
                restored_authoritative.active_species_validated
            )
            self._last_authoritative_refresh_at = monotonic()
            if status_callback is not None:
                status_callback(
                    "Validated the last known native actor profile against the "
                    f"current process; {len(self._actor_slots)} selected actors "
                    "were recovered without relation inference."
                )
        elif callable(finder):
            if status_callback is not None:
                status_callback(
                    "Inferring the current authoritative global actor relation "
                    "from the validated player and monster anchors..."
                )
            try:
                authoritative = discover_authoritative_actors(
                    self._memory,  # type: ignore[arg-type]
                    discovery,
                    selected_species_ids=self._selected_species_ids,
                    actor_stride=self.actor_stride,
                    object_span=self.object_span,
                    maximum_address=self._maximum_scan_address,
                    private_memory_only=self._private_memory_only,
                    chunk_size=self._discovery_chunk_bytes,
                    coordinate_limit=self._coordinate_limit,
                    cancellation=cancellation,
                    deadline=deadline,
                )
                self._authoritative_discovery = authoritative
                if authoritative.succeeded:
                    self._actor_source = "authoritative_global"
                    self._actor_slots = authoritative.actor_bases
                    self._authoritative_relation_offset = (
                        authoritative.relation_offset
                    )
                    self._authoritative_relation_value = (
                        authoritative.relation_value
                    )
                    self._authoritative_species_counts = (
                        authoritative.species_counts
                    )
                    self._authoritative_active_candidates = (
                        authoritative.active_candidates
                    )
                    self._active_species_offset = (
                        authoritative.active_species_offset
                    )
                    self._active_species_validated = bool(
                        authoritative.active_species_validated
                    )
                    self._last_authoritative_refresh_at = monotonic()
                    if status_callback is not None:
                        status_callback(authoritative.message)
                else:
                    self._authoritative_last_error = authoritative.message
                    if status_callback is not None:
                        status_callback(
                            "Authoritative actor relation was not proven; using "
                            "the bounded slab fallback and recording full evidence. "
                            + authoritative.message
                        )
            except Exception as error:
                self._authoritative_last_error = (
                    f"{type(error).__name__}: {error}"
                )
                if status_callback is not None:
                    status_callback(
                        "Authoritative actor discovery failed safely; using the "
                        "bounded slab fallback. " + self._authoritative_last_error
                    )

        self._pending_actor_slots: set[int] = set()
        self._pending_actor_slot_order: tuple[int, ...] = ()
        self._pending_actor_slot_cursor = 0
        self._last_runtime_slot_probe_at = -math.inf
        self._runtime_slot_probe_interval_seconds = 0.20
        self._runtime_slot_probe_batch_size = 0
        self._runtime_promoted_slots = 0
        self._active_species_is_cross_slot_alias = bool(
            self.actor_stride
            and (
                self.monster_targets[0].active_species_offset
                - self.monster_targets[0].species_offset
            ) > 0
            and (
                self.monster_targets[0].active_species_offset
                - self.monster_targets[0].species_offset
            )
            % int(self.actor_stride)
            == 0
        )
        (
            self.active_species_matches,
            self.active_species_samples,
            _configured_active_field_matches,
        ) = self._probe_field_matches_species(self.monster_active_species_offset)
        del _configured_active_field_matches
        self.active_species_reliable = self._active_species_validated
        (
            self.configured_hp_matches,
            self.configured_hp_samples,
        ) = self._probe_configured_hp_anchor()

    def _probe_field_matches_species(self, offset: int) -> tuple[int, int, bool]:
        matches = 0
        readable = 0
        for target in self.monster_targets[:64]:
            try:
                value = _i32(self._memory.read(target.base + int(offset), 4))
            except Exception:
                continue
            readable += 1
            matches += int(value == target.species)
        reliable = readable >= 2 and matches * 5 >= readable * 4
        return matches, readable, reliable

    def _probe_configured_hp_anchor(self) -> tuple[int, int]:
        matches = 0
        readable = 0
        for target in self.monster_targets[:64]:
            try:
                value = _i32(
                    self._memory.read(
                        target.base + self.configured_monster_hp_offset, 4
                    )
                )
            except Exception:
                continue
            readable += 1
            matches += int(value == target.hp)
        return matches, readable

    @property
    def player_slots(self) -> tuple[int, ...]:
        return self._player_slots

    @property
    def selected_player_slot(self) -> int:
        return self._player_slots[0]

    @property
    def actor_slots(self) -> tuple[int, ...]:
        with self._cache_lock:
            return self._actor_slots

    @property
    def pending_actor_slots(self) -> int:
        with self._cache_lock:
            return len(self._pending_actor_slots)

    @property
    def runtime_promoted_slots(self) -> int:
        with self._cache_lock:
            return self._runtime_promoted_slots

    @property
    def actor_source(self) -> str:
        return self._actor_source

    @property
    def authoritative_relation_offset(self) -> int | None:
        return self._authoritative_relation_offset

    @property
    def authoritative_relation_value(self) -> int | None:
        return self._authoritative_relation_value

    @property
    def authoritative_relation_validated(self) -> bool:
        return self._actor_source == "authoritative_global"

    @property
    def authoritative_species_counts(self) -> tuple[tuple[int, int], ...]:
        return self._authoritative_species_counts

    @property
    def authoritative_refreshes(self) -> int:
        return self._authoritative_refreshes

    @property
    def authoritative_refresh_failures(self) -> int:
        return self._authoritative_refresh_failures

    @property
    def authoritative_last_error(self) -> str | None:
        return self._authoritative_last_error

    @property
    def active_species_offset(self) -> int | None:
        return self._active_species_offset

    @property
    def active_species_validated(self) -> bool:
        return self._active_species_validated

    @property
    def active_species_candidates(self) -> tuple[ActiveFieldCandidate, ...]:
        return self._authoritative_active_candidates

    def authoritative_diagnostics(self) -> dict[str, object]:
        discovery = self._authoritative_discovery
        return {
            "actor_source": self._actor_source,
            "relation_offset": self._authoritative_relation_offset,
            "relation_value": self._authoritative_relation_value,
            "relation_validated": self.authoritative_relation_validated,
            "actor_count": len(self.actor_slots),
            "species_counts": [list(item) for item in self._authoritative_species_counts],
            "refreshes": self._authoritative_refreshes,
            "refresh_failures": self._authoritative_refresh_failures,
            "last_error": self._authoritative_last_error,
            "active_species_offset": self._active_species_offset,
            "active_species_validated": self._active_species_validated,
            "active_candidates": [
                item.to_dict() for item in self._authoritative_active_candidates
            ],
            "initial_discovery": None if discovery is None else discovery.to_dict(),
        }

    @property
    def active_species_is_cross_slot_alias(self) -> bool:
        return self._active_species_is_cross_slot_alias

    @property
    def hp_offset(self) -> int:
        """Backward-compatible alias for the player live-HP offset."""

        return self.player_hp_offset

    @property
    def monster_hp_candidate_offsets(self) -> tuple[int, ...]:
        """The exact tester discovery already proved one live monster HP field."""

        return (self.monster_hp_offset,)

    @property
    def monster_hp_offset_validated(self) -> bool:
        return True

    @property
    def monster_hp_transition_support(self) -> tuple[tuple[int, int], ...]:
        return ()

    def _rank_player_slots(
        self,
        player: TracePlayerTarget,
        *,
        configured_player_offset: int | None,
    ) -> tuple[int, ...]:
        valid: list[int] = []
        for slot in player.direct_module_slots:
            if not (
                self.module.base_address
                <= slot
                < self.module.base_address + self.module.size
            ):
                continue
            try:
                if _u32(self._memory.read(slot, 4)) == player.base:
                    valid.append(int(slot))
            except Exception:
                continue
        configured_slot = (
            None
            if configured_player_offset is None
            else self.module.base_address + int(configured_player_offset)
        )
        return tuple(
            sorted(
                set(valid),
                key=lambda slot: (
                    0 if configured_slot is None else abs(slot - configured_slot),
                    slot,
                ),
            )
        )

    def _has_self_alias(self, base: int, offsets: tuple[int, ...]) -> bool:
        for offset in offsets:
            try:
                if _u32(self._memory.read(base + offset, 4)) == base:
                    return True
            except Exception:
                continue
        return False

    def _is_actor_slot(self, base: int) -> bool:
        if base <= 0x10000 or not self._actor_self_offsets:
            return False
        if not self._has_self_alias(base, self._actor_self_offsets):
            return False
        layout = self.monster_targets[0]
        try:
            x = _f32(self._memory.read(base + layout.x_offset, 4))
            y = _f32(self._memory.read(base + layout.y_offset, 4))
            z = _f32(self._memory.read(base + layout.z_offset, 4))
        except Exception:
            return False
        return all(
            math.isfinite(value) and abs(value) <= 100_000_000.0
            for value in (x, y, z)
        )

    def _expand_actor_slots(
        self,
        anchors: set[int],
        *,
        existing: set[int] | None = None,
        slots_each_direction: int,
    ) -> set[int]:
        slots = set() if existing is None else set(existing)
        slots.update(anchors)
        stride = self.actor_stride
        if stride is None or stride <= 0 or slots_each_direction <= 0:
            return slots
        for anchor in tuple(sorted(anchors)):
            for direction in (-1, 1):
                for step in range(1, slots_each_direction + 1):
                    candidate = anchor + direction * step * stride
                    if candidate in slots:
                        continue
                    if not self._is_actor_slot(candidate):
                        # Stop at the first allocation boundary in this direction.
                        break
                    slots.add(candidate)
        return slots

    def _build_pending_actor_slot_cache(
        self,
        anchors: set[int],
        *,
        slots_each_direction: int,
        existing: set[int] | None = None,
    ) -> set[int]:
        """Remember every possible slot in each proven 32-slot neighborhood.

        Some slots are not instantiated while pointer recovery is running. They
        may become Dantalian (or another selected species) later when the player
        enters that part of the map. These addresses are cheap dormant probes;
        only slots that later pass the already-proven actor-layout checks are
        promoted into the high-frequency cache.
        """

        pending: set[int] = set()
        stride = self.actor_stride
        if stride is None or stride <= 0 or slots_each_direction <= 0:
            return pending
        occupied = set() if existing is None else set(existing)
        for anchor in anchors:
            for direction in (-1, 1):
                for step in range(1, slots_each_direction + 1):
                    candidate = int(anchor) + direction * step * int(stride)
                    if 0x10000 < candidate <= 0x7FFFFFFF and candidate not in occupied:
                        pending.add(candidate)
        return pending

    def _rebuild_pending_actor_slot_order(self) -> None:
        self._pending_actor_slot_order = tuple(sorted(self._pending_actor_slots))
        if not self._pending_actor_slot_order:
            self._pending_actor_slot_cursor = 0
        else:
            self._pending_actor_slot_cursor %= len(self._pending_actor_slot_order)

    def refresh_runtime_actor_slots(
        self,
        *,
        maximum_probes: int | None = None,
        force: bool = False,
        cancellation: object | None = None,
        deadline: float | None = None,
    ) -> ActorSlotProbeResult:
        """Refresh the proven global actor relation without rerunning recovery.

        In authoritative mode this repeats only the old-style global reference
        scan for the already inferred relation value. Player recovery, layout
        inference, and HP-field discovery are never touched. The bounded slab
        fallback remains a no-op for compatibility.
        """

        del maximum_probes
        if (
            self._actor_source != "authoritative_global"
            or self._authoritative_relation_offset is None
            or self._authoritative_relation_value is None
        ):
            with self._cache_lock:
                return ActorSlotProbeResult(
                    probed_slots=0,
                    promoted_slots=0,
                    cached_slots=len(self._actor_slots),
                    pending_slots=0,
                )

        now = monotonic()
        age = now - self._last_authoritative_refresh_at
        # A forced preflight refresh immediately after constructor discovery
        # would scan the whole process twice for no new information.
        if age < 2.0 or (
            not force
            and age < self._authoritative_refresh_interval_seconds
        ):
            with self._cache_lock:
                return ActorSlotProbeResult(
                    probed_slots=0,
                    promoted_slots=0,
                    cached_slots=len(self._actor_slots),
                    pending_slots=0,
                )

        try:
            refreshed = refresh_authoritative_actors(
                self._memory,  # type: ignore[arg-type]
                self._trace_discovery_contract(),
                relation_offset=self._authoritative_relation_offset,
                relation_value=self._authoritative_relation_value,
                selected_species_ids=self._selected_species_ids,
                actor_stride=self.actor_stride,
                maximum_address=self._maximum_scan_address,
                private_memory_only=self._private_memory_only,
                chunk_size=self._discovery_chunk_bytes,
                coordinate_limit=self._coordinate_limit,
                cancellation=cancellation,
                deadline=deadline,
            )
            minimum_coverage = max(
                2, math.ceil(len(self.monster_targets) * 0.60)
            )
            if refreshed.evidence.exact_anchor_coverage < minimum_coverage:
                raise IndependentNativeReadError(
                    "Authoritative refresh lost exact-anchor coverage: "
                    f"{refreshed.evidence.exact_anchor_coverage}/"
                    f"{len(self.monster_targets)}"
                )
            with self._cache_lock:
                previous = set(self._actor_slots)
                self._actor_slots = refreshed.actor_bases
                current = set(self._actor_slots)
                self._authoritative_species_counts = refreshed.species_counts
                self._authoritative_active_candidates = (
                    refreshed.active_candidates
                )
                if refreshed.active_species_validated:
                    self._active_species_offset = (
                        refreshed.active_species_offset
                    )
                    self._active_species_validated = True
                    self.active_species_reliable = True
                self._last_authoritative_refresh_at = now
                self._authoritative_refreshes += 1
                self._authoritative_last_error = None
                return ActorSlotProbeResult(
                    probed_slots=refreshed.evidence.references,
                    promoted_slots=len(current - previous),
                    cached_slots=len(self._actor_slots),
                    pending_slots=0,
                )
        except Exception as error:
            self._last_authoritative_refresh_at = now
            self._authoritative_refresh_failures += 1
            self._authoritative_last_error = f"{type(error).__name__}: {error}"
            with self._cache_lock:
                return ActorSlotProbeResult(
                    probed_slots=0,
                    promoted_slots=0,
                    cached_slots=len(self._actor_slots),
                    pending_slots=0,
                )

    def _trace_discovery_contract(self) -> TraceTargetDiscovery:
        """Rebuild the immutable discovery contract for relation refreshes."""

        # Evidence is not consumed by authoritative refresh. Reuse the original
        # object when available rather than manufacturing pointer-recovery data.
        discovery = getattr(self, "_trace_discovery", None)
        if isinstance(discovery, TraceTargetDiscovery):
            return discovery
        raise IndependentNativeReadError(
            "The original validated trace discovery is unavailable"
        )

    def _build_actor_slot_cache(self, *, slots_each_direction: int) -> tuple[int, ...]:
        """Build the complete bounded slot universe around proven anchors.

        FlyFF uses repeated 32-slot actor slabs. A slot may be empty while
        recovery runs and become a Dantalian later. The old implementation put
        such addresses into a rotating pending queue and only promoted them
        after a timed probe. That made correctness depend on timing. The full
        bounded universe is small enough to inspect directly on each ordinary
        actor snapshot, so every possible slab address is retained here.
        """

        anchors = {int(item.base) for item in self.monster_targets}
        slots = set(anchors)
        slots.update(
            self._build_pending_actor_slot_cache(
                anchors,
                slots_each_direction=slots_each_direction,
                existing=anchors,
            )
        )
        return tuple(sorted(slots))

    def merge_monster_targets(
        self,
        targets: tuple[TraceMonsterTarget, ...],
        *,
        slots_each_direction: int | None = None,
    ) -> ActorCacheMergeResult:
        """Merge newly discovered exact anchors and expand their actor slabs."""

        template = self.monster_targets[0]
        compatible: dict[int, TraceMonsterTarget] = {}
        for target in targets:
            if (
                target.species_offset != template.species_offset
                or target.hp_offset != template.hp_offset
                or target.x_offset != template.x_offset
                or target.y_offset != template.y_offset
                or target.z_offset != template.z_offset
            ):
                continue
            compatible[int(target.base)] = target
        old_targets = {int(target.base): target for target in self.monster_targets}
        new_anchor_bases = set(compatible) - set(old_targets)
        old_targets.update(compatible)
        self.monster_targets = tuple(
            sorted(old_targets.values(), key=lambda target: target.base)
        )
        self._actor_self_offsets = tuple(
            dict.fromkeys(
                offset
                for target in self.monster_targets
                for offset in target.self_pointer_offsets
            )
        )
        if self.actor_stride is None:
            self.actor_stride = infer_actor_stride(tuple(old_targets))
        expansion = (
            self._slots_each_direction
            if slots_each_direction is None
            else max(0, int(slots_each_direction))
        )
        self._slots_each_direction = expansion
        with self._cache_lock:
            previous_slots = set(self._actor_slots)
            # Rebuild the complete bounded slab universe from every proven
            # anchor. Newly instantiated species do not need a promotion pass.
            self._actor_slots = self._build_actor_slot_cache(
                slots_each_direction=expansion
            )
            merged_slots = set(self._actor_slots)
            self._pending_actor_slots.clear()
            self._pending_actor_slot_order = ()
            self._pending_actor_slot_cursor = 0
            total_slots = len(self._actor_slots)
        return ActorCacheMergeResult(
            anchors_seen=len(compatible),
            new_anchors=len(new_anchor_bases),
            new_slots=len(merged_slots - previous_slots),
            total_anchors=len(self.monster_targets),
            total_slots=total_slots,
        )

    def read_player(self) -> IndependentPlayerRead:
        target = self.player_target
        failures: list[str] = []
        for slot in self._player_slots:
            try:
                base = _u32(self._memory.read(slot, 4))
                if base <= 0:
                    failures.append(f"0x{slot:X}: null")
                    continue
                if not self._has_self_alias(base, target.self_pointer_offsets):
                    failures.append(f"0x{slot:X}: self mismatch")
                    continue
                hp = _i32(self._memory.read(base + target.hp_offset, 4))
                x = _f32(self._memory.read(base + target.x_offset, 4))
                y = _f32(self._memory.read(base + target.y_offset, 4))
                z = _f32(self._memory.read(base + target.z_offset, 4))
                if hp < 0 or not all(math.isfinite(value) for value in (x, y, z)):
                    failures.append(f"0x{slot:X}: invalid fields")
                    continue
                return IndependentPlayerRead(base, slot, hp, x, y, z)
            except Exception as error:
                failures.append(f"0x{slot:X}: {type(error).__name__}")
        raise IndependentNativeReadError(
            "All recovered player aliases failed: " + "; ".join(failures)
        )

    def _read_monster_runtime_fields(
        self,
        base: int,
    ) -> tuple[int, int, int, float, float, float]:
        """Read one actor using only dynamically proven layout fields."""

        layout = self.monster_targets[0]
        active_offset = (
            self._active_species_offset
            if self._active_species_offset is not None
            else self.monster_active_species_offset
        )
        offsets = [
            layout.species_offset,
            self.monster_hp_offset,
            layout.x_offset,
            layout.y_offset,
            layout.z_offset,
        ]
        if active_offset is not None:
            offsets.append(active_offset)
        first = min(offsets)
        last = max(offsets) + 4
        try:
            data = self._memory.read(base + first, last - first)
            if len(data) != last - first:
                raise OSError("short actor read")
            active = (
                0
                if active_offset is None
                else _i32(data, active_offset - first)
            )
            return (
                _i32(data, layout.species_offset - first),
                active,
                _i32(data, self.monster_hp_offset - first),
                _f32(data, layout.x_offset - first),
                _f32(data, layout.y_offset - first),
                _f32(data, layout.z_offset - first),
            )
        except Exception:
            active = 0
            if active_offset is not None:
                active = _i32(self._memory.read(base + active_offset, 4))
            return (
                _i32(self._memory.read(base + layout.species_offset, 4)),
                active,
                _i32(self._memory.read(base + self.monster_hp_offset, 4)),
                _f32(self._memory.read(base + layout.x_offset, 4)),
                _f32(self._memory.read(base + layout.y_offset, 4)),
                _f32(self._memory.read(base + layout.z_offset, 4)),
            )

    def _observe_active_candidates(
        self,
        base: int,
        species: int,
        hp: int,
    ) -> None:
        """Accumulate runtime proof for candidate loaded/instantiated fields."""

        if species <= 0 or hp < 0 or self._active_species_validated:
            return
        for candidate in self._authoritative_active_candidates[:8]:
            offset = int(candidate.offset)
            try:
                value = _i32(self._memory.read(base + offset, 4))
            except Exception:
                continue
            support = self._active_runtime_support.setdefault(offset, [0, 0, 0, 0])
            if hp == 0:
                support[3] += 1
                support[2] += int(value == species)
            else:
                support[1] += 1
                support[0] += int(value == species)

        ranked: list[tuple[float, float, int]] = []
        for offset, (live_match, live_total, dead_match, dead_total) in (
            self._active_runtime_support.items()
        ):
            if live_total < 20 or dead_total < 3:
                continue
            live_ratio = live_match / live_total
            dead_ratio = dead_match / dead_total
            if live_ratio >= 0.80 and dead_ratio <= 0.20:
                ranked.append((live_ratio, -dead_ratio, offset))
        if ranked:
            _live, _dead, offset = max(ranked)
            self._active_species_offset = int(offset)
            self._active_species_validated = True
            self.active_species_reliable = True

    def read_actor_hp_states(
        self,
        candidates: Iterable[tuple[int, int]],
    ) -> dict[tuple[int, int], int]:
        """Read live species/HP directly for already validated cast targets.

        This deliberately bypasses visibility radius, map filtering, active-field
        heuristics, and the actor snapshot cache. A cast candidate was already a
        valid selected actor when EVA was pressed; post-cast confirmation only
        needs to know whether that same base/species reached HP zero.
        """

        layout = self.monster_targets[0]
        first = min(layout.species_offset, self.monster_hp_offset)
        last = max(layout.species_offset, self.monster_hp_offset) + 4
        result: dict[tuple[int, int], int] = {}
        for raw_base, raw_species in candidates:
            base = int(raw_base)
            expected_species = int(raw_species)
            if base <= 0x10000 or expected_species <= 0:
                continue
            try:
                data = self._memory.read(base + first, last - first)
                if len(data) != last - first:
                    raise OSError("short actor HP read")
                species = _i32(data, layout.species_offset - first)
                hp = _i32(data, self.monster_hp_offset - first)
            except Exception:
                try:
                    species = _i32(
                        self._memory.read(base + layout.species_offset, 4)
                    )
                    hp = _i32(
                        self._memory.read(base + self.monster_hp_offset, 4)
                    )
                except Exception:
                    continue
            if species == expected_species and hp >= 0:
                result[(base, expected_species)] = hp
        return result

    def _scan_monsters(
        self,
        player: IndependentPlayerRead,
        *,
        allowed_species: set[int] | None,
        vision_radius_native: float | None,
    ) -> _MonsterScan:
        visible_living: list[IndependentMonsterRead] = []
        actor_states: list[IndependentActorSlotRead] = []
        living_bases: list[int] = []
        occupied = 0
        tracked = 0
        instantiated = 0
        living = 0
        damaged = 0
        zero_hp = 0
        other_species = 0
        empty = 0
        invalid_hp = 0
        active_mismatches = 0
        unreadable = 0
        active_probe_budget = 32
        with self._cache_lock:
            actor_slots = self._actor_slots
        for base in actor_slots:
            if base == player.base:
                continue
            try:
                if not self._has_self_alias(base, self._actor_self_offsets):
                    unreadable += 1
                    continue
                species, active, hp, x, y, z = self._read_monster_runtime_fields(base)
                if not all(math.isfinite(value) for value in (x, y, z)):
                    unreadable += 1
                    continue
                distance = math.hypot(x - player.x, z - player.z)
                if hp == 0 or active_probe_budget > 0:
                    self._observe_active_candidates(base, species, hp)
                    if hp > 0:
                        active_probe_budget -= 1
                raw_active_match = species > 0 and active == species
                # The active/loaded field is valuable diagnostics, but it is
                # not a correctness gate. The current client has already shown
                # stale and cross-slot aliases at historical offsets. Excluding
                # an actor here would hide it from both the minimap and kill
                # tracking. Authoritative relation + self/species/HP/coordinates
                # remain the inclusion contract.
                active_match = True
                target_species = (
                    species > 0
                    and (allowed_species is None or species in allowed_species)
                )
                if species <= 0:
                    empty += 1
                    state = "empty"
                else:
                    occupied += 1
                    if not raw_active_match:
                        active_mismatches += 1
                    if not target_species:
                        other_species += 1
                        state = "other_species"
                    else:
                        tracked += 1
                        # Actor presence is never gated by the historical
                        # active/loaded fields. HP zero is the explicit death
                        # state and the reusable slot remains in the slab set.
                        if hp == 0:
                            instantiated += 1
                            zero_hp += 1
                            state = "dead"
                        else:
                            reference_full_hp = self.expected_full_hp_by_species.get(
                                species
                            )
                            if hp < 0 or (
                                reference_full_hp is not None
                                and hp > reference_full_hp
                            ):
                                invalid_hp += 1
                                state = "invalid_hp"
                            else:
                                instantiated += 1
                                living += 1
                                living_bases.append(base)
                                state = "living"
                                if (
                                    reference_full_hp is not None
                                    and 0 < hp < reference_full_hp
                                ):
                                    damaged += 1
                                if (
                                    vision_radius_native is None
                                    or distance <= float(vision_radius_native)
                                ):
                                    full_hp = (
                                        hp
                                        if reference_full_hp is None
                                        else reference_full_hp
                                    )
                                    visible_living.append(
                                        IndependentMonsterRead(
                                            base=base,
                                            species=species,
                                            hp=hp,
                                            full_hp=full_hp,
                                            x=x,
                                            y=y,
                                            z=z,
                                            active_species=active,
                                            distance_native=distance,
                                            hp_offset=self.monster_hp_offset,
                                            hp_candidates=((self.monster_hp_offset, hp),),
                                            hp_offset_validated=True,
                                        )
                                    )
                actor_states.append(
                    IndependentActorSlotRead(
                        base=base,
                        species=species,
                        hp=hp,
                        x=x,
                        y=y,
                        z=z,
                        active_species=active,
                        active_matches_species=raw_active_match,
                        target_species=target_species,
                        state=state,
                        distance_native=distance,
                        hp_offset=self.monster_hp_offset,
                        hp_candidates=((self.monster_hp_offset, hp),),
                        hp_offset_validated=True,
                    )
                )
            except Exception:
                unreadable += 1
        visible_living.sort(key=lambda item: (item.distance_native, item.base))
        actor_states.sort(key=lambda item: item.base)
        dead_or_dormant = zero_hp + empty
        return _MonsterScan(
            visible_living=tuple(visible_living),
            actor_states=tuple(actor_states),
            living_bases=tuple(sorted(living_bases)),
            occupied=occupied,
            tracked=tracked,
            instantiated=instantiated,
            living=living,
            damaged=damaged,
            zero_hp=zero_hp,
            other_species=other_species,
            empty=empty,
            invalid_hp=invalid_hp,
            active_mismatches=active_mismatches,
            dead_or_dormant=dead_or_dormant,
            unreadable=unreadable,
        )

    def read_monsters(
        self,
        player: IndependentPlayerRead,
        *,
        allowed_species: set[int] | None = None,
        vision_radius_native: float | None = None,
    ) -> tuple[IndependentMonsterRead, ...]:
        """Return living monsters within the requested vision radius."""

        self.refresh_runtime_actor_slots(
            maximum_probes=self._runtime_slot_probe_batch_size
        )
        return self._scan_monsters(
            player,
            allowed_species=allowed_species,
            vision_radius_native=vision_radius_native,
        ).visible_living

    def snapshot(
        self,
        *,
        allowed_species: set[int] | None = None,
        vision_radius_native: float | None = None,
    ) -> IndependentNativeSnapshot:
        player = self.read_player()
        self.refresh_runtime_actor_slots(
            maximum_probes=self._runtime_slot_probe_batch_size
        )
        scan = self._scan_monsters(
            player,
            allowed_species=allowed_species,
            vision_radius_native=vision_radius_native,
        )
        return IndependentNativeSnapshot(
            player=player,
            monsters=scan.visible_living,
            actor_states=scan.actor_states,
            living_actor_bases=scan.living_bases,
            cached_actor_slots=len(self.actor_slots),
            occupied_slots=scan.occupied,
            tracked_species_slots=scan.tracked,
            instantiated_monsters=scan.instantiated,
            living_monsters=scan.living,
            visible_living_monsters=len(scan.visible_living),
            damaged_monsters=scan.damaged,
            zero_hp_monsters=scan.zero_hp,
            other_species_slots=scan.other_species,
            empty_slots=scan.empty,
            invalid_hp_slots=scan.invalid_hp,
            active_field_mismatches=scan.active_mismatches,
            active_field_reliable=self.active_species_reliable,
            dead_or_dormant_slots=scan.dead_or_dormant,
            unreadable_slots=scan.unreadable,
            actor_stride=self.actor_stride,
            monster_hp_offset=self.monster_hp_offset,
            monster_hp_candidate_offsets=(self.monster_hp_offset,),
            monster_hp_offset_validated=True,
            monster_hp_transition_support=(),
            actor_source=self._actor_source,
            authoritative_relation_offset=self._authoritative_relation_offset,
            authoritative_relation_value=self._authoritative_relation_value,
            authoritative_relation_validated=self.authoritative_relation_validated,
            authoritative_species_counts=self._authoritative_species_counts,
            authoritative_refreshes=self._authoritative_refreshes,
            authoritative_refresh_failures=self._authoritative_refresh_failures,
            authoritative_last_error=self._authoritative_last_error,
            active_species_offset=self._active_species_offset,
            active_species_validated=self._active_species_validated,
            active_species_candidates=tuple(
                (
                    int(item.offset),
                    int(item.living_matches),
                    int(item.living_samples),
                    int(item.zero_hp_matches),
                    int(item.zero_hp_samples),
                    bool(item.validated),
                )
                for item in self._authoritative_active_candidates
            ),
        )
