from __future__ import annotations

import math
import struct
from collections import Counter
from dataclasses import asdict, dataclass
from threading import RLock
from time import monotonic
from typing import Iterable, Mapping, Protocol

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
    ) -> None:
        if discovery.outcome != "success" or discovery.player is None:
            raise ValueError("successful target discovery with one player is required")
        if not discovery.monsters:
            raise ValueError("at least one validated monster anchor is required")
        self._memory = memory
        self.module = module
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
        self.actor_stride = infer_actor_stride(
            tuple(item.base for item in discovery.monsters)
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
        # The exact discovery anchors prove the actor stride and slab layout.
        # Keep every bounded stride-neighbour address in the ordinary read set.
        # Do not require a slot to be instantiated during recovery before it can
        # appear on the map or participate in kill tracking later.
        self._actor_slots = self._build_actor_slot_cache(
            slots_each_direction=self._slots_each_direction
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
        # Never use an active/loaded field as a correctness gate in independent
        # mode. The dynamic candidate was a cross-slot species alias and the old
        # configured field is stale on the validated client. Keep match counts
        # only as diagnostics.
        del _configured_active_field_matches
        self.active_species_reliable = False
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
    ) -> ActorSlotProbeResult:
        """Compatibility no-op for the removed promotion queue.

        Every bounded slab address is already part of ``actor_slots`` and is
        inspected directly by ``snapshot``. No slot has to be promoted first.
        """

        del maximum_probes, force
        with self._cache_lock:
            return ActorSlotProbeResult(
                probed_slots=0,
                promoted_slots=0,
                cached_slots=len(self._actor_slots),
                pending_slots=0,
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
        """Read one reusable actor slot with one compact request.

        No active/loaded field participates in this read. The old configured
        field is stale and discovery's apparent +0x217C match is the next actor
        slot's species field. Reading only species, live HP, and coordinates is
        both cheaper and correct for living, dead, dormant, and reused slots.
        """

        layout = self.monster_targets[0]
        offsets = (
            layout.species_offset,
            self.monster_hp_offset,
            layout.x_offset,
            layout.y_offset,
            layout.z_offset,
        )
        first = min(offsets)
        last = max(offsets) + 4
        try:
            data = self._memory.read(base + first, last - first)
            if len(data) != last - first:
                raise OSError("short actor read")
            return (
                _i32(data, layout.species_offset - first),
                0,
                _i32(data, self.monster_hp_offset - first),
                _f32(data, layout.x_offset - first),
                _f32(data, layout.y_offset - first),
                _f32(data, layout.z_offset - first),
            )
        except Exception:
            return (
                _i32(self._memory.read(base + layout.species_offset, 4)),
                0,
                _i32(self._memory.read(base + self.monster_hp_offset, 4)),
                _f32(self._memory.read(base + layout.x_offset, 4)),
                _f32(self._memory.read(base + layout.y_offset, 4)),
                _f32(self._memory.read(base + layout.z_offset, 4)),
            )

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
                active_match = species > 0 and active == species
                target_species = (
                    species > 0
                    and (allowed_species is None or species in allowed_species)
                )
                if species <= 0:
                    empty += 1
                    state = "empty"
                else:
                    occupied += 1
                    if not active_match:
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
                        active_matches_species=active_match,
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
        )
