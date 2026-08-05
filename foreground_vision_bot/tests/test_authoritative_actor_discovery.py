from __future__ import annotations

import struct
from dataclasses import dataclass
from types import SimpleNamespace
from threading import Event, Thread

import pytest

from position.AuthoritativeActorDiscovery import discover_authoritative_actors
from position.IndependentNativeReader import IndependentNativeReader
from position.NativeTraceTargets import (
    TraceMonsterTarget,
    TracePlayerTarget,
    TraceTargetDiscovery,
    TraceTargetEvidence,
)
from position.Win32ProcessMemory import MemorySearchCancelled, ModuleInfo


SPECIES_OFFSET = 0x174
X_OFFSET = 0x160
Y_OFFSET = 0x164
Z_OFFSET = 0x168
HP_OFFSET = 0x81C
SELF_OFFSET = 0x3C8
PLAYER_SELF_OFFSET = 0x1EF0
ACTIVE_OFFSET = 0x1A20
PRESENCE_OFFSET = 0x19A4
HISTORICAL_PRESENCE_OFFSET = 0x1DCC
STRIDE = 0x2008
OBJECT_SPAN = 0x4000
WORLD_VALUE = 0x61000000
MIRROR_VALUE = 0x62000000


@dataclass
class Region:
    base_address: int
    data: bytearray


class Memory:
    def __init__(self) -> None:
        self.regions: list[Region] = []
        self.last_search_diagnostics = SimpleNamespace(
            bytes_read=0,
            regions_read=0,
            matches=0,
        )
        self.find_calls = 0

    def allocate(self, base: int, size: int) -> None:
        self.regions.append(Region(base, bytearray(size)))

    def _region(self, address: int, size: int) -> tuple[Region, int]:
        for region in self.regions:
            offset = address - region.base_address
            if 0 <= offset and offset + size <= len(region.data):
                return region, offset
        raise OSError(f"unmapped 0x{address:X}+0x{size:X}")

    def read(self, address: int, size: int) -> bytes:
        region, offset = self._region(address, size)
        return bytes(region.data[offset : offset + size])

    def write_u32(self, address: int, value: int) -> None:
        region, offset = self._region(address, 4)
        struct.pack_into("<I", region.data, offset, value)

    def write_i32(self, address: int, value: int) -> None:
        region, offset = self._region(address, 4)
        struct.pack_into("<i", region.data, offset, value)

    def write_f32(self, address: int, value: float) -> None:
        region, offset = self._region(address, 4)
        struct.pack_into("<f", region.data, offset, value)

    def find_u32(
        self,
        value: int,
        *,
        maximum_address: int,
        private_only: bool,
        chunk_size: int,
        cancellation: object | None = None,
        deadline: float | None = None,
    ) -> tuple[int, ...]:
        del private_only, chunk_size, cancellation, deadline
        self.find_calls += 1
        needle = struct.pack("<I", value)
        matches: list[int] = []
        bytes_read = 0
        regions_read = 0
        for region in self.regions:
            if region.base_address > maximum_address:
                continue
            regions_read += 1
            bytes_read += len(region.data)
            data = bytes(region.data)
            start = 0
            while True:
                found = data.find(needle, start)
                if found < 0:
                    break
                address = region.base_address + found
                if address % 4 == 0 and address <= maximum_address:
                    matches.append(address)
                start = found + 1
        self.last_search_diagnostics = SimpleNamespace(
            bytes_read=bytes_read,
            regions_read=regions_read,
            matches=len(matches),
        )
        return tuple(sorted(set(matches)))


def _evidence() -> TraceTargetEvidence:
    return TraceTargetEvidence(
        bytes_scanned=0,
        regions_scanned=0,
        read_failures=0,
        species_hits=0,
        spawn_x_hits=0,
        monster_candidates=3,
        player_candidates=1,
        monster_hp_rejections=0,
        monster_coordinate_rejections=0,
        player_hp_rejections=0,
        player_coordinate_rejections=0,
        observed_hp_values=(),
        monster_base_hypotheses=3,
        monster_layout_ties=0,
        monster_self_aliases=1,
        player_self_rejections=0,
        inferred_species_offset=SPECIES_OFFSET,
        inferred_active_species_offset=0x217C,
        inferred_monster_hp_offset=HP_OFFSET,
        inferred_x_offset=X_OFFSET,
        inferred_y_offset=Y_OFFSET,
        inferred_z_offset=Z_OFFSET,
        selected_player_hp_offset=HP_OFFSET,
    )


def _write_actor(
    memory: Memory,
    base: int,
    *,
    relation_offset: int,
    relation_value: int,
    species: int,
    hp: int,
    x: float,
    z: float,
    active: int | None = None,
    presence: int | None = None,
) -> None:
    memory.allocate(base, OBJECT_SPAN)
    memory.write_u32(base + SELF_OFFSET, base)
    memory.write_u32(base + relation_offset, relation_value)
    memory.write_i32(base + SPECIES_OFFSET, species)
    memory.write_i32(base + HP_OFFSET, hp)
    memory.write_i32(
        base + ACTIVE_OFFSET,
        species if active is None and hp > 0 else 0 if active is None else active,
    )
    memory.write_i32(
        base + PRESENCE_OFFSET,
        species if presence is None and species > 0 else 0 if presence is None else presence,
    )
    memory.write_f32(base + X_OFFSET, x)
    memory.write_f32(base + Y_OFFSET, 100.0)
    memory.write_f32(base + Z_OFFSET, z)


def _monster(base: int, *, species: int = 944, hp: int = 400236) -> TraceMonsterTarget:
    return TraceMonsterTarget(
        base=base,
        species=species,
        hp=hp,
        x=250.0,
        y=100.0,
        z=86.0,
        self_pointer_offsets=(SELF_OFFSET,),
        species_offset=SPECIES_OFFSET,
        active_species_offset=0x217C,
        hp_offset=HP_OFFSET,
        x_offset=X_OFFSET,
        y_offset=Y_OFFSET,
        z_offset=Z_OFFSET,
    )


def _fixture(
    relation_offset: int = 0x2A4,
) -> tuple[Memory, ModuleInfo, TraceTargetDiscovery, dict[str, tuple[int, ...] | int]]:
    memory = Memory()
    module = ModuleInfo("Neuz.exe", "Neuz.exe", 0x100000, 0x900000)
    memory.allocate(module.base_address, module.size)
    memory.allocate(WORLD_VALUE, 0x2000)
    memory.allocate(MIRROR_VALUE, 0x2000)

    player = 0x50000000
    memory.allocate(player, OBJECT_SPAN)
    memory.write_u32(player + PLAYER_SELF_OFFSET, player)
    memory.write_u32(player + relation_offset, WORLD_VALUE)
    memory.write_i32(player + HP_OFFSET, 26930)
    memory.write_f32(player + X_OFFSET, 253.0)
    memory.write_f32(player + Y_OFFSET, 100.0)
    memory.write_f32(player + Z_OFFSET, 86.0)
    player_slot = module.base_address + 0x586318
    memory.write_u32(player_slot, player)

    asterions = (0x20000000, 0x23000000, 0x28000000)
    for index, base in enumerate(asterions):
        _write_actor(
            memory,
            base,
            relation_offset=relation_offset,
            relation_value=WORLD_VALUE,
            species=944,
            hp=400236,
            x=250.0 + index,
            z=86.0 + index,
        )
    # A single coincidental historical-offset match must not establish a field.
    memory.write_i32(asterions[0] + HISTORICAL_PRESENCE_OFFSET, 944)

    dantalians = (0x31000000, 0x36000000, 0x3B000000, 0x3F000000)
    for index, base in enumerate(dantalians):
        _write_actor(
            memory,
            base,
            relation_offset=relation_offset,
            relation_value=WORLD_VALUE,
            species=948,
            hp=275000 if index < 2 else 0,
            x=320.0 + index,
            z=400.0 + index,
        )

    mirrors = (0x41000000, 0x43000000)
    for index, base in enumerate(mirrors):
        _write_actor(
            memory,
            base,
            relation_offset=relation_offset,
            relation_value=MIRROR_VALUE,
            species=948,
            hp=275000,
            x=321.0 + index,
            z=401.0 + index,
        )

    dormant = (0x45000000, 0x47000000)
    for base in dormant:
        _write_actor(
            memory,
            base,
            relation_offset=relation_offset,
            relation_value=WORLD_VALUE,
            species=0,
            hp=0,
            x=0.0,
            z=0.0,
        )

    # The manager object containing actor references is only a ranking aid; the
    # discovery must still validate every process-wide reference independently.
    memory.write_u32(WORLD_VALUE + 0x20, player)
    for index, base in enumerate(asterions):
        memory.write_u32(WORLD_VALUE + 0x100 + index * 4, base)

    discovery = TraceTargetDiscovery(
        player=TracePlayerTarget(
            base=player,
            hp=26930,
            x=253.0,
            y=100.0,
            z=86.0,
            self_pointer_offsets=(PLAYER_SELF_OFFSET,),
            direct_module_slots=(player_slot,),
            hp_offset=HP_OFFSET,
            species_offset=SPECIES_OFFSET,
            active_species_offset=0x1DBC,
            x_offset=X_OFFSET,
            y_offset=Y_OFFSET,
            z_offset=Z_OFFSET,
        ),
        monsters=tuple(_monster(base) for base in asterions),
        evidence=_evidence(),
        outcome="success",
        message="validated exact anchors",
    )
    return memory, module, discovery, {
        "player": player,
        "player_slot": player_slot,
        "asterions": asterions,
        "dantalians": dantalians,
        "mirrors": mirrors,
        "dormant": dormant,
        "relation_offset": relation_offset,
    }


@pytest.mark.parametrize("relation_offset", [0x2A4, 0x6F0])
def test_dynamically_recovers_global_relation_and_excludes_mirror_objects(
    relation_offset: int,
) -> None:
    memory, _module, discovery, addresses = _fixture(relation_offset)

    result = discover_authoritative_actors(
        memory,
        discovery,
        selected_species_ids={944, 948},
        actor_stride=STRIDE,
        object_span=OBJECT_SPAN,
        maximum_address=0x7FFFFFFF,
        private_memory_only=True,
        chunk_size=1 << 20,
        coordinate_limit=100_000.0,
    )

    assert result.succeeded
    assert result.relation_offset == relation_offset
    assert result.relation_value == WORLD_VALUE
    expected = set(addresses["asterions"]) | set(addresses["dantalians"])
    assert set(result.actor_bases) == expected
    assert not (set(addresses["mirrors"]) & set(result.actor_bases))
    assert dict(result.species_counts) == {944: 3, 948: 4}
    assert result.active_species_offset == ACTIVE_OFFSET
    assert result.active_species_validated is True
    assert result.presence_species_offset == PRESENCE_OFFSET
    assert result.presence_species_validated is True
    assert result.presence_candidates[0].offset == PRESENCE_OFFSET
    assert all(
        item.offset != HISTORICAL_PRESENCE_OFFSET
        for item in result.presence_candidates
    )
    assert result.relation_scans[0].exact_anchor_coverage == 3


def test_reader_enables_only_the_dynamically_recovered_moved_presence_field() -> None:
    memory, module, discovery, _addresses = _fixture()
    reader = IndependentNativeReader(
        memory,
        module,
        discovery,
        configured_player_offset=0x586318,
        expected_full_hp_by_species={944: 400236},
        selected_species_ids={944, 948},
        known_actor_stride=STRIDE,
    )

    assert reader.recovered_presence_species_offset == PRESENCE_OFFSET
    assert reader.presence_species_validated is True
    assert reader.enable_presence_optimized_sampling(
        selected_species_ids={944, 948}
    ) is True
    diagnostics = reader.presence_sampler_diagnostics()
    assert diagnostics.enabled is True
    assert diagnostics.offset == PRESENCE_OFFSET


def test_provisional_presence_candidate_keeps_correctness_first_full_reads() -> None:
    memory, module, discovery, addresses = _fixture()
    for base in addresses["dormant"]:
        memory.write_u32(base + addresses["relation_offset"], MIRROR_VALUE)
    reader = IndependentNativeReader(
        memory,
        module,
        discovery,
        configured_player_offset=0x586318,
        expected_full_hp_by_species={944: 400236},
        selected_species_ids={944, 948},
        known_actor_stride=STRIDE,
    )

    assert reader.recovered_presence_species_offset == PRESENCE_OFFSET
    assert reader.presence_species_validated is False
    assert reader.enable_presence_optimized_sampling(
        selected_species_ids={944, 948}
    ) is False
    assert reader.presence_sampler_diagnostics().enabled is False


def test_reader_uses_authoritative_global_set_for_map_and_hp_zero_kills() -> None:
    memory, module, discovery, addresses = _fixture()
    reader = IndependentNativeReader(
        memory,
        module,
        discovery,
        configured_player_offset=0x586318,
        expected_full_hp_by_species={944: 400236},
        selected_species_ids={944, 948},
        slots_each_direction=31,
        authoritative_refresh_interval_seconds=20.0,
    )

    assert reader.actor_source == "authoritative_global"
    assert reader.authoritative_relation_validated is True
    assert set(reader.actor_slots) == (
        set(addresses["asterions"]) | set(addresses["dantalians"])
    )
    snapshot = reader.snapshot(
        allowed_species={944, 948},
        vision_radius_native=1_000_000.0,
    )
    assert snapshot.living_monsters == 5
    assert snapshot.zero_hp_monsters == 2
    assert {item.species for item in snapshot.monsters} == {944, 948}
    assert snapshot.actor_source == "authoritative_global"
    assert snapshot.authoritative_relation_offset == addresses["relation_offset"]
    assert snapshot.active_species_offset == ACTIVE_OFFSET
    assert snapshot.active_species_validated is True

    killed = addresses["dantalians"][0]
    memory.write_i32(killed + HP_OFFSET, 0)
    memory.write_i32(killed + ACTIVE_OFFSET, 0)
    assert reader.read_actor_hp_states(((killed, 948),)) == {(killed, 948): 0}
    after = reader.snapshot(
        allowed_species={944, 948},
        vision_radius_native=1_000_000.0,
    )
    state = next(item for item in after.actor_states if item.base == killed)
    assert state.hp == 0
    assert state.state == "dead"


def test_authoritative_refresh_finds_actor_allocated_after_recovery() -> None:
    memory, module, discovery, addresses = _fixture()
    reader = IndependentNativeReader(
        memory,
        module,
        discovery,
        configured_player_offset=0x586318,
        expected_full_hp_by_species={944: 400236},
        selected_species_ids={944, 948},
        slots_each_direction=31,
        authoritative_refresh_interval_seconds=20.0,
    )
    late = 0x47000000
    _write_actor(
        memory,
        late,
        relation_offset=int(addresses["relation_offset"]),
        relation_value=WORLD_VALUE,
        species=948,
        hp=275000,
        x=450.0,
        z=450.0,
    )
    assert late not in reader.actor_slots

    reader._last_authoritative_refresh_at = float("-inf")
    refresh = reader.refresh_runtime_actor_slots(force=True)

    assert refresh.promoted_slots == 1
    assert late in reader.actor_slots
    assert dict(reader.authoritative_species_counts) == {944: 3, 948: 5}



def test_ordinary_snapshots_never_repeat_process_wide_authoritative_scan() -> None:
    memory, module, discovery, _addresses = _fixture()
    reader = IndependentNativeReader(
        memory,
        module,
        discovery,
        configured_player_offset=0x586318,
        expected_full_hp_by_species={944: 400236},
        selected_species_ids={944, 948},
        slots_each_direction=31,
        authoritative_refresh_interval_seconds=0.0,
    )
    calls_after_discovery = memory.find_calls

    for _ in range(5):
        reader.snapshot(
            allowed_species={944, 948},
            vision_radius_native=1_000_000.0,
        )
        reader.read_monsters(
            reader.read_player(),
            allowed_species={944, 948},
            vision_radius_native=1_000_000.0,
        )

    assert memory.find_calls == calls_after_discovery
    forced = reader.refresh_runtime_actor_slots(force=True)
    assert forced.cached_slots >= 2
    assert memory.find_calls == calls_after_discovery + 1



def test_authoritative_refresh_is_single_flight_across_concurrent_readers() -> None:
    memory, module, discovery, _addresses = _fixture()
    reader = IndependentNativeReader(
        memory,
        module,
        discovery,
        configured_player_offset=0x586318,
        expected_full_hp_by_species={944: 400236},
        selected_species_ids={944, 948},
        slots_each_direction=31,
        authoritative_refresh_interval_seconds=20.0,
    )
    calls_after_discovery = memory.find_calls
    reader._last_authoritative_refresh_at = float("-inf")
    reader._authoritative_species_counts = ((944, 1), (948, 1))
    original_find = memory.find_u32
    entered = Event()
    release = Event()

    def blocking_find(*args, **kwargs):
        entered.set()
        assert release.wait(2.0)
        return original_find(*args, **kwargs)

    memory.find_u32 = blocking_find  # type: ignore[method-assign]
    results = []
    first = Thread(
        target=lambda: results.append(reader.refresh_runtime_actor_slots(force=True))
    )
    first.start()
    assert entered.wait(1.0)
    second = reader.refresh_runtime_actor_slots(force=True)
    release.set()
    first.join(2.0)

    assert not first.is_alive()
    assert second.probed_slots == 0
    assert memory.find_calls == calls_after_discovery + 1
    assert len(results) == 1

def test_reader_falls_back_safely_when_no_player_shared_relation_exists() -> None:
    memory, module, discovery, addresses = _fixture()
    relation_offset = int(addresses["relation_offset"])
    memory.write_u32(int(addresses["player"]) + relation_offset, MIRROR_VALUE)

    reader = IndependentNativeReader(
        memory,
        module,
        discovery,
        configured_player_offset=0x586318,
        expected_full_hp_by_species={944: 400236},
        selected_species_ids={944, 948},
        slots_each_direction=1,
    )

    assert reader.actor_source == "bounded_slab_fallback"
    assert reader.authoritative_relation_validated is False
    diagnostics = reader.authoritative_diagnostics()
    assert diagnostics["relation_validated"] is False
    assert diagnostics["initial_discovery"]["outcome"] == "shared_relation_not_found"


def test_rejects_shared_distractor_that_only_recovers_exact_anchors() -> None:
    memory, _module, discovery, addresses = _fixture()
    distractor_value = 0x64000000
    distractor_offset = 0x180
    memory.allocate(distractor_value, 0x2000)
    memory.write_u32(int(addresses["player"]) + distractor_offset, distractor_value)
    for index, base in enumerate(addresses["asterions"]):
        memory.write_u32(int(base) + distractor_offset, distractor_value)
        memory.write_u32(distractor_value + 0x100 + index * 4, int(base))
    memory.write_u32(distractor_value + 0x20, int(addresses["player"]))

    result = discover_authoritative_actors(
        memory,
        discovery,
        selected_species_ids={944, 948},
        actor_stride=STRIDE,
        object_span=OBJECT_SPAN,
        maximum_address=0x7FFFFFFF,
        private_memory_only=True,
        chunk_size=1 << 20,
        coordinate_limit=100_000.0,
    )

    assert result.succeeded
    assert result.relation_offset == addresses["relation_offset"]
    assert len(result.relation_scans) >= 2
    distractor_scan = next(
        item for item in result.relation_scans if item.offset == distractor_offset
    )
    assert distractor_scan.exact_anchor_coverage == 3
    assert distractor_scan.valid_actor_bases == 3
    assert dict(result.species_counts) == {944: 3, 948: 4}


def test_preferred_validated_relation_stops_after_one_scan_when_species_absent() -> None:
    memory, _module, discovery, addresses = _fixture()
    relation_offset = int(addresses["relation_offset"])

    # Simulate starting recovery before any Dantalian is loaded. The saved
    # relation is already known-good, so absence of species 948 must not force
    # three additional full-process scans.
    for base in addresses["dantalians"]:
        memory.write_i32(int(base) + SPECIES_OFFSET, 944)
        memory.write_i32(int(base) + HP_OFFSET, 400236)

    result = discover_authoritative_actors(
        memory,
        discovery,
        selected_species_ids={944, 948},
        actor_stride=STRIDE,
        object_span=OBJECT_SPAN,
        maximum_address=0x7FFFFFFF,
        private_memory_only=True,
        chunk_size=1 << 20,
        coordinate_limit=100_000.0,
        preferred_relation_offsets=(relation_offset,),
    )

    assert result.succeeded
    assert result.relation_offset == relation_offset
    assert memory.find_calls == 1
    assert dict(result.species_counts) == {944: 7}


def test_reader_propagates_authoritative_scan_cancellation() -> None:
    memory, module, discovery, _addresses = _fixture()

    def cancelled_find(*_args, **_kwargs):
        raise MemorySearchCancelled("cancelled by user")

    memory.find_u32 = cancelled_find  # type: ignore[method-assign]

    with pytest.raises(MemorySearchCancelled):
        IndependentNativeReader(
            memory,
            module,
            discovery,
            configured_player_offset=0x586318,
            expected_full_hp_by_species={944: 400236},
            selected_species_ids={944, 948},
        )


def test_validated_active_field_is_diagnostic_and_never_hides_living_actor() -> None:
    memory, module, discovery, addresses = _fixture()
    reader = IndependentNativeReader(
        memory,
        module,
        discovery,
        configured_player_offset=0x586318,
        expected_full_hp_by_species={944: 400236},
        selected_species_ids={944, 948},
        slots_each_direction=31,
    )
    assert reader.active_species_validated is True

    actor = int(addresses["dantalians"][0])
    memory.write_i32(actor + ACTIVE_OFFSET, 0)
    memory.write_i32(actor + HP_OFFSET, 275000)

    snapshot = reader.snapshot(
        allowed_species={944, 948},
        vision_radius_native=1_000_000.0,
    )
    state = next(item for item in snapshot.actor_states if item.base == actor)
    assert state.state == "living"
    assert state.active_matches_species is False
    assert any(item.base == actor for item in snapshot.monsters)
