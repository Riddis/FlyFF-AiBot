from __future__ import annotations

import struct
from types import SimpleNamespace

from position.IndependentMonsterRediscovery import rediscover_known_layout_monsters
from position.IndependentNativeReader import IndependentNativeReader, infer_actor_stride
from position.NativeTraceTargets import (
    TraceMonsterTarget,
    TracePlayerTarget,
    TraceTargetDiscovery,
    TraceTargetEvidence,
)
from position.Win32ProcessMemory import ModuleInfo
from tools.test_native_independent_reader import (
    _derive_slot_lifecycle_events,
    _state_map,
)


PLAYER_HP_OFFSET = 0x81C
MONSTER_LIVE_HP_OFFSET = 0x81C
CONFIGURED_WRONG_HP_OFFSET = 0x814
CONFIGURED_ACTIVE_SPECIES_OFFSET = 0x1DBC
DISCOVERY_CROSS_SLOT_ACTIVE_OFFSET = 0x217C
SPECIES_OFFSET = 0x174
X_OFFSET = 0x160
Y_OFFSET = 0x164
Z_OFFSET = 0x168
SELF_OFFSET = 0x3C8
STRIDE = 0x2008


class Memory:
    def __init__(self) -> None:
        self.data: dict[int, bytes] = {}
        self.last_search_diagnostics = SimpleNamespace(
            bytes_read=0, regions_read=0
        )

    def write_u32(self, address: int, value: int) -> None:
        self.data[address] = struct.pack("<I", value)

    def write_i32(self, address: int, value: int) -> None:
        self.data[address] = struct.pack("<i", value)

    def write_f32(self, address: int, value: float) -> None:
        self.data[address] = struct.pack("<f", value)

    def read(self, address: int, size: int) -> bytes:
        if size == 4 and address in self.data:
            return self.data[address]
        raise OSError(f"unmapped 0x{address:X}")

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
        needle = struct.pack("<I", int(value))
        hits = tuple(
            sorted(
                address
                for address, data in self.data.items()
                if address <= maximum_address and data == needle
            )
        )
        self.last_search_diagnostics = SimpleNamespace(
            bytes_read=len(self.data) * 4, regions_read=1
        )
        return hits


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
        inferred_active_species_offset=DISCOVERY_CROSS_SLOT_ACTIVE_OFFSET,
        inferred_monster_hp_offset=MONSTER_LIVE_HP_OFFSET,
        inferred_x_offset=X_OFFSET,
        inferred_y_offset=Y_OFFSET,
        inferred_z_offset=Z_OFFSET,
        selected_player_hp_offset=PLAYER_HP_OFFSET,
    )


def _populate_actor(
    memory: Memory,
    base: int,
    *,
    species: int = 944,
    hp: int = 400236,
    active_species: int | None = None,
    configured_hp_junk: int = 85,
    x: float | None = None,
    z: float = 86.0,
) -> None:
    memory.write_u32(base + SELF_OFFSET, base)
    memory.write_i32(base + SPECIES_OFFSET, species)
    memory.write_i32(
        base + CONFIGURED_ACTIVE_SPECIES_OFFSET,
        species if active_species is None else active_species,
    )
    # The old +0x814 assumption is deliberately populated with index-like junk.
    memory.write_i32(base + CONFIGURED_WRONG_HP_OFFSET, configured_hp_junk)
    memory.write_i32(base + MONSTER_LIVE_HP_OFFSET, hp)
    # Do not write the discovery cross-slot candidate: for contiguous actors,
    # base+0x217C is naturally the next slot's base+0x174 species field.
    memory.write_f32(base + X_OFFSET, float(base & 0xFF) if x is None else x)
    memory.write_f32(base + Y_OFFSET, 100.0)
    memory.write_f32(base + Z_OFFSET, z)


def _monster_target(base: int, *, species: int = 944, hp: int = 400236) -> TraceMonsterTarget:
    return TraceMonsterTarget(
        base=base,
        species=species,
        hp=hp,
        x=float(base & 0xFF),
        y=100.0,
        z=86.0,
        self_pointer_offsets=(SELF_OFFSET,),
        species_offset=SPECIES_OFFSET,
        active_species_offset=DISCOVERY_CROSS_SLOT_ACTIVE_OFFSET,
        hp_offset=MONSTER_LIVE_HP_OFFSET,
        x_offset=X_OFFSET,
        y_offset=Y_OFFSET,
        z_offset=Z_OFFSET,
    )


def _player_target(player: int, slots: tuple[int, ...]) -> TracePlayerTarget:
    return TracePlayerTarget(
        base=player,
        hp=31513,
        x=253.0,
        y=100.0,
        z=86.0,
        self_pointer_offsets=(0x1EF0,),
        direct_module_slots=slots,
        hp_offset=PLAYER_HP_OFFSET,
        species_offset=SPECIES_OFFSET,
        active_species_offset=CONFIGURED_ACTIVE_SPECIES_OFFSET,
        x_offset=X_OFFSET,
        y_offset=Y_OFFSET,
        z_offset=Z_OFFSET,
    )


def _populate_player(memory: Memory, player: int, *, hp: int = 31513) -> None:
    memory.write_u32(player + 0x1EF0, player)
    memory.write_i32(player + CONFIGURED_WRONG_HP_OFFSET, 0)
    memory.write_i32(player + PLAYER_HP_OFFSET, hp)
    memory.write_f32(player + X_OFFSET, 253.0)
    memory.write_f32(player + Y_OFFSET, 100.0)
    memory.write_f32(player + Z_OFFSET, 86.0)


def _reader(
    memory: Memory,
    module: ModuleInfo,
    discovery: TraceTargetDiscovery,
    *,
    configured_player_offset: int | None = None,
    slots_each_direction: int = 0,
) -> IndependentNativeReader:
    return IndependentNativeReader(
        memory,
        module,
        discovery,
        configured_player_offset=configured_player_offset,
        monster_current_hp_offset=CONFIGURED_WRONG_HP_OFFSET,
        monster_active_species_offset=CONFIGURED_ACTIVE_SPECIES_OFFSET,
        expected_full_hp_by_species={944: 400236},
        slots_each_direction=slots_each_direction,
    )


def _fixture(
    *,
    active_species: int | None = None,
    slots_each_direction: int = 0,
) -> tuple[Memory, ModuleInfo, IndependentNativeReader, tuple[int, ...]]:
    memory = Memory()
    module = ModuleInfo("Neuz.exe", "Neuz.exe", 0x100000, 0x900000)
    player = 0x50000000
    slot_a = module.base_address + 0x586318
    slot_b = module.base_address + 0x5868D4
    memory.write_u32(slot_a, player)
    memory.write_u32(slot_b, player)
    _populate_player(memory, player)
    bases = (0x20000000, 0x20002008, 0x20004010)
    for base in bases:
        _populate_actor(memory, base, active_species=active_species)
    discovery = TraceTargetDiscovery(
        player=_player_target(player, (slot_a, slot_b)),
        monsters=tuple(_monster_target(base) for base in bases),
        evidence=_evidence(),
        outcome="success",
        message="ok",
    )
    return (
        memory,
        module,
        _reader(
            memory,
            module,
            discovery,
            configured_player_offset=0x5852B8,
            slots_each_direction=slots_each_direction,
        ),
        bases,
    )


def test_uses_dynamic_live_hp_and_expands_persistent_reusable_slots() -> None:
    memory, _module, reader, bases = _fixture(slots_each_direction=31)
    dead_slot = bases[-1] + STRIDE
    other_species_slot = dead_slot + STRIDE
    _populate_actor(memory, dead_slot, hp=0)
    _populate_actor(memory, other_species_slot, species=715, hp=120000)

    # Rebuild after the adjacent slots exist, as the production reader does at init.
    discovery = TraceTargetDiscovery(
        player=reader.player_target,
        monsters=reader.monster_targets,
        evidence=_evidence(),
        outcome="success",
        message="ok",
    )
    reader = _reader(
        memory,
        reader.module,
        discovery,
        configured_player_offset=0x5852B8,
        slots_each_direction=31,
    )

    assert infer_actor_stride(bases) == STRIDE
    assert reader.monster_hp_offset == MONSTER_LIVE_HP_OFFSET
    assert reader.configured_monster_hp_offset == CONFIGURED_WRONG_HP_OFFSET
    assert reader.configured_hp_matches == 0
    assert dead_slot in reader.actor_slots
    assert other_species_slot in reader.actor_slots

    snapshot = reader.snapshot(allowed_species={944})
    assert snapshot.player.hp == 31513
    assert snapshot.cached_actor_slots == 5
    assert snapshot.living_monsters == 3
    assert snapshot.zero_hp_monsters == 1
    assert snapshot.other_species_slots == 1
    assert snapshot.tracked_species_slots == 4
    assert set(snapshot.living_actor_bases) == set(bases)
    assert "actor_states" not in snapshot.to_dict()


def test_hp_zero_is_death_and_same_slot_can_respawn_or_change_species() -> None:
    memory, _module, reader, bases = _fixture()
    base = bases[0]
    memory.write_f32(base + X_OFFSET, 250.0)
    memory.write_f32(base + Z_OFFSET, 86.0)
    before = reader.snapshot(allowed_species={944})

    _populate_actor(memory, base, hp=0, x=250.0, z=86.0)
    dead = reader.snapshot(allowed_species={944})
    events = _derive_slot_lifecycle_events(
        _state_map(before), _state_map(dead), kill_event_radius=80.0
    )
    assert [event["base"] for event in events["deaths"]] == [base]
    assert events["deaths"][0]["probable_kill"] is True
    assert dead.zero_hp_monsters == 1
    assert base in reader.actor_slots

    _populate_actor(memory, base, species=715, hp=120000)
    reused = reader.snapshot(allowed_species={944})
    events = _derive_slot_lifecycle_events(
        _state_map(dead), _state_map(reused), kill_event_radius=80.0
    )
    assert events["deaths"] == []
    assert events["reuses"][0]["from_species"] == 944
    assert events["reuses"][0]["to_species"] == 715
    assert reused.other_species_slots == 1

    _populate_actor(memory, base, species=944, hp=400236)
    respawned = reader.snapshot(allowed_species={944})
    events = _derive_slot_lifecycle_events(
        _state_map(reused), _state_map(respawned), kill_event_radius=80.0
    )
    assert events["reuses"][0]["to_species"] == 944
    assert events["spawns"][0]["base"] == base
    assert base in respawned.living_actor_bases


def test_unproven_active_field_is_diagnostic_and_cannot_reject_every_actor() -> None:
    memory, _module, reader, bases = _fixture(active_species=0)
    assert reader.active_species_matches == 0
    assert reader.active_species_reliable is False

    snapshot = reader.snapshot(allowed_species={944})
    assert snapshot.living_monsters == len(bases)
    assert snapshot.active_field_mismatches == len(bases)
    assert snapshot.active_field_reliable is False


def test_reliable_active_field_can_filter_dormant_positive_hp_slot() -> None:
    memory, module, reader, bases = _fixture(slots_each_direction=31)
    dormant = bases[-1] + STRIDE
    _populate_actor(memory, dormant, hp=400236, active_species=0)
    discovery = TraceTargetDiscovery(
        player=reader.player_target,
        monsters=reader.monster_targets,
        evidence=_evidence(),
        outcome="success",
        message="ok",
    )
    reader = _reader(memory, module, discovery, slots_each_direction=31)
    assert reader.active_species_reliable is True

    snapshot = reader.snapshot(allowed_species={944})
    assert snapshot.cached_actor_slots == 4
    assert snapshot.living_monsters == 3
    assert snapshot.active_field_mismatches == 1
    assert next(state for state in snapshot.actor_states if state.base == dormant).state == "dormant"


def test_configured_0x814_junk_never_overrides_discovered_0x81c_hp() -> None:
    memory, _module, reader, bases = _fixture()
    base = bases[0]
    _populate_actor(memory, base, hp=123456, configured_hp_junk=327760)
    snapshot = reader.snapshot(allowed_species={944})
    monster = next(item for item in snapshot.monsters if item.base == base)
    assert monster.hp == 123456
    assert monster.full_hp == 400236
    assert monster.damaged is True
    assert snapshot.damaged_monsters == 1


def test_active_mismatch_or_species_change_without_hp_zero_is_not_a_kill() -> None:
    memory, _module, reader, bases = _fixture()
    before = reader.snapshot(allowed_species={944})
    base = bases[0]
    _populate_actor(memory, base, hp=400236, active_species=0)
    mismatch = reader.snapshot(allowed_species={944})
    events = _derive_slot_lifecycle_events(
        _state_map(before), _state_map(mismatch), kill_event_radius=80.0
    )
    assert events["deaths"] == []

    _populate_actor(memory, base, species=715, hp=120000)
    changed = reader.snapshot(allowed_species={944})
    events = _derive_slot_lifecycle_events(
        _state_map(mismatch), _state_map(changed), kill_event_radius=80.0
    )
    assert events["deaths"] == []
    assert len(events["reuses"]) == 1


def test_living_membership_is_not_limited_by_vision_radius() -> None:
    memory, _module, reader, bases = _fixture()
    near, far = bases[:2]
    memory.write_f32(near + X_OFFSET, 254.0)
    memory.write_f32(near + Z_OFFSET, 86.0)
    memory.write_f32(far + X_OFFSET, 1000.0)
    memory.write_f32(far + Z_OFFSET, 1000.0)
    snapshot = reader.snapshot(allowed_species={944}, vision_radius_native=250.0)
    assert snapshot.living_monsters == 3
    assert snapshot.visible_living_monsters == 2
    assert set(snapshot.living_actor_bases) == set(bases)
    assert far not in {monster.base for monster in snapshot.monsters}


def test_falls_back_to_second_player_alias() -> None:
    memory, _module, reader, _bases = _fixture()
    first, second = reader.player_slots
    memory.write_u32(first, 0)
    assert reader.read_player().pointer_slot == second


def test_background_known_layout_rediscovery_merges_new_slab_without_player_anchor(
) -> None:
    memory, _module, reader, bases = _fixture(slots_each_direction=31)
    new_bases = (0x30000000, 0x30002008, 0x30004010)
    for base in new_bases:
        _populate_actor(memory, base)

    # Player movement and HP changes do not affect monster-only rediscovery.
    _populate_player(memory, reader.player_target.base, hp=29000)
    memory.write_f32(reader.player_target.base + X_OFFSET, 500.0)
    memory.write_f32(reader.player_target.base + Z_OFFSET, 500.0)

    result = rediscover_known_layout_monsters(
        memory,
        template=reader.monster_targets[0],
        species_hp={944: 400236},
        maximum_address=0x7FFFFFFF,
        coordinate_limit=1_000_000.0,
    )
    merge = reader.merge_monster_targets(
        result.targets, slots_each_direction=31
    )

    assert set(new_bases).issubset(reader.actor_slots)
    assert merge.new_anchors == len(new_bases)
    assert merge.new_slots == len(new_bases)
    snapshot = reader.snapshot(allowed_species={944})
    assert snapshot.living_monsters == len(bases) + len(new_bases)


def test_reader_uses_the_single_hp_field_proven_by_exact_discovery() -> None:
    memory, _module, reader, bases = _fixture()

    assert reader.monster_hp_candidate_offsets == (MONSTER_LIVE_HP_OFFSET,)
    assert reader.monster_hp_offset == MONSTER_LIVE_HP_OFFSET
    assert reader.monster_hp_offset_validated is True

    killed_base = bases[0]
    memory.write_i32(killed_base + MONSTER_LIVE_HP_OFFSET, 0)
    after = reader.snapshot(allowed_species={944})

    assert after.monster_hp_offset == MONSTER_LIVE_HP_OFFSET
    assert after.monster_hp_offset_validated is True
    assert after.zero_hp_monsters == 1
    killed = next(state for state in after.actor_states if state.base == killed_base)
    assert killed.hp == 0
    assert killed.hp_offset == MONSTER_LIVE_HP_OFFSET
    assert killed.hp_candidates == ((MONSTER_LIVE_HP_OFFSET, 0),)

