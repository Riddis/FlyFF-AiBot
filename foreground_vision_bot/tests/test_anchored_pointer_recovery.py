from __future__ import annotations

import json
import struct
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from position.AnchoredPointerDiscovery import PointerRecoveryHints
from position.MonsterConfig import NativeMonsterConfig
from position.native_process_service import (
    NativePointerSnapshotError,
    NativeProcessService,
    NativeRecoveryOutcome,
)
from position.NativePointerRecovery import (
    PointerRecoveryState,
    recover_local_player_pointer,
)
from position.Win32ProcessMemory import ModuleInfo


@dataclass(frozen=True)
class Region:
    base_address: int
    size: int
    region_type: int
    protection: int = 0x04


class AnchoredMemory:
    def __init__(self) -> None:
        self.pid = 9911
        self.module_base_value = 0x00400000
        self.module = bytearray(0x10000)
        self.heap_base = 0x20000000
        self.heap = bytearray(0x10000)
        self.world_base = 0x24000000
        self.world = bytearray(0x1000)
        self.scalar_world_base = 0x3F800000
        self.scalar_world = bytearray(0x1000)
        self.player_base = self.heap_base + 0x6000
        self.player_slot_offset = 0x3000
        self.world_slot_offset = 0x3100
        self.new_world_offset = 0x180
        self.new_self_offset = 0x1EF0
        self.max_hp_offset = 0x818

    def module_info(self, module_name: str) -> ModuleInfo:
        return ModuleInfo(
            name=module_name,
            path=r"C:\FlyFF\Neuz.exe",
            base_address=self.module_base_value,
            size=len(self.module),
        )

    def module_base(self, module_name: str) -> int:
        return self.module_info(module_name).base_address

    def close(self) -> None:
        return None

    def readable_regions(self, *, maximum_address=0x7FFFFFFF, private_only=True):
        regions = (
            Region(self.module_base_value, len(self.module), 0x1000000),
            Region(self.heap_base, len(self.heap), 0x20000),
            Region(self.world_base, len(self.world), 0x20000),
            Region(self.scalar_world_base, len(self.scalar_world), 0x20000),
        )
        return tuple(
            region
            for region in regions
            if region.base_address <= maximum_address
            and (not private_only or region.region_type == 0x20000)
        )

    def read(self, address: int, size: int) -> bytes:
        for base, data in (
            (self.module_base_value, self.module),
            (self.heap_base, self.heap),
            (self.world_base, self.world),
            (self.scalar_world_base, self.scalar_world),
        ):
            if base <= address and address + size <= base + len(data):
                start = address - base
                return bytes(data[start : start + size])
        raise RuntimeError(f"unreadable 0x{address:X}+0x{size:X}")

    def find_u32(
        self,
        value: int,
        *,
        maximum_address: int,
        private_only: bool,
        chunk_size: int,
        cancellation=None,
        deadline=None,
    ) -> tuple[int, ...]:
        del chunk_size, cancellation, deadline
        needle = struct.pack("<I", value)
        matches: list[int] = []
        for region in self.readable_regions(
            maximum_address=maximum_address,
            private_only=private_only,
        ):
            data = self.read(region.base_address, region.size)
            cursor = 0
            while True:
                found = data.find(needle, cursor)
                if found < 0:
                    break
                address = region.base_address + found
                if address % 4 == 0:
                    matches.append(address)
                cursor = found + 1
        return tuple(matches)

    def u32(self, address: int, value: int) -> None:
        self._pack(address, "<I", value)

    def i32(self, address: int, value: int) -> None:
        self._pack(address, "<i", value)

    def f32(self, address: int, value: float) -> None:
        self._pack(address, "<f", value)

    def _pack(self, address: int, format_string: str, value: int | float) -> None:
        for base, data in (
            (self.module_base_value, self.module),
            (self.heap_base, self.heap),
            (self.world_base, self.world),
            (self.scalar_world_base, self.scalar_world),
        ):
            if base <= address < base + len(data):
                struct.pack_into(format_string, data, address - base, value)
                return
        raise RuntimeError(f"unwritable 0x{address:X}")


def _populate(memory: AnchoredMemory, config: NativeMonsterConfig) -> None:
    memory.u32(memory.module_base_value + 0x800, memory.module_base_value + 0x1000)
    memory.u32(memory.module_base_value + 0x804, memory.module_base_value + 0x1100)
    memory.u32(memory.module_base_value + 0x808, memory.module_base_value + 0x1200)
    memory.u32(memory.world_base, memory.module_base_value + 0x800)
    memory.u32(
        memory.module_base_value + memory.player_slot_offset,
        memory.player_base,
    )
    memory.u32(
        memory.module_base_value + memory.world_slot_offset,
        memory.world_base,
    )
    actors = (
        (memory.heap_base + 0x0000, 944, 1000, 240.0, 80.0),
        (memory.heap_base + 0x3000, 948, 1200, 260.0, 90.0),
    )
    for base, species, hp, x, z in actors:
        memory.u32(base + memory.new_self_offset, base)
        memory.u32(base + memory.new_world_offset, memory.world_base)
        memory.i32(base + config.species_offset, species)
        memory.i32(base + config.active_species_offset, species)
        memory.i32(base + config.hp_offset, hp)
        memory.f32(base + config.x_offset, x)
        memory.f32(base + config.y_offset, 12.0)
        memory.f32(base + config.z_offset, z)

    player = memory.player_base
    memory.u32(player + memory.new_self_offset, player)
    memory.u32(player + memory.new_world_offset, memory.world_base)
    memory.i32(player + config.species_offset, 0)
    memory.i32(player + config.active_species_offset, 0)
    memory.i32(player + config.hp_offset, 5000)
    memory.i32(player + memory.max_hp_offset, 6000)
    memory.f32(player + config.x_offset, 253.0)
    memory.f32(player + config.y_offset, 14.0)
    memory.f32(player + config.z_offset, 86.0)


def test_species_spawn_hp_candidate_requires_movement_before_recovery() -> None:
    memory = AnchoredMemory()
    config = NativeMonsterConfig(
        player_pointer_offset=0x1000,
        world_pointer_offset=0x1100,
        discovery_chunk_bytes=0x1000,
    )
    _populate(memory, config)
    hints = PointerRecoveryHints(
        known_species_ids=(944, 948),
        player_spawn_x=253.0,
        player_spawn_z=86.0,
        player_current_hp=5000,
        player_max_hp=6000,
    )
    state = PointerRecoveryState()

    first = recover_local_player_pointer(
        memory,
        module_base=memory.module_base_value,
        configured_player_pointer_offset=config.player_pointer_offset,
        monster_config=config,
        state=state,
        hints=hints,
        chunk_size=0x1000,
        timeout_seconds=2.0,
    )

    assert first is None
    first_metrics = state.metrics_for(memory.pid, memory.module_base_value)
    assert first_metrics is not None
    assert first_metrics.outcome == "movement_required"
    assert first_metrics.monster_candidates == 2
    assert first_metrics.inferred_world_offset == memory.new_world_offset
    assert first_metrics.inferred_self_offset == memory.new_self_offset
    assert first_metrics.spawn_player_matches == 1
    assert first_metrics.stable_spawn_candidates == 1

    memory.f32(memory.player_base + config.x_offset, 258.0)
    memory.i32(memory.player_base + config.hp_offset, 4900)
    moved_hints = PointerRecoveryHints(
        known_species_ids=(944, 948),
        player_spawn_x=253.0,
        player_spawn_z=86.0,
        player_current_hp=4900,
        player_max_hp=6000,
    )
    second = recover_local_player_pointer(
        memory,
        module_base=memory.module_base_value,
        configured_player_pointer_offset=config.player_pointer_offset,
        monster_config=config,
        state=state,
        hints=moved_hints,
        chunk_size=0x1000,
        timeout_seconds=2.0,
    )

    assert second is not None
    assert second.strategy == "anchored_movement"
    assert second.movement_validated
    assert second.player_pointer_offset == memory.player_slot_offset
    assert second.world_pointer_offset == memory.world_slot_offset
    assert second.world_field_offset == memory.new_world_offset
    assert second.world_vtable_offset == 0x800
    assert second.world_vtable_field_offset == 0
    assert second.self_pointer_offset == memory.new_self_offset


def test_stationary_second_sample_retains_pending_candidate() -> None:
    memory = AnchoredMemory()
    config = NativeMonsterConfig(
        player_pointer_offset=0x1000,
        world_pointer_offset=0x1100,
        discovery_chunk_bytes=0x1000,
    )
    _populate(memory, config)
    hints = PointerRecoveryHints(
        known_species_ids=(944, 948),
        player_spawn_x=253.0,
        player_spawn_z=86.0,
        player_current_hp=5000,
        player_max_hp=6000,
    )
    state = PointerRecoveryState()

    assert recover_local_player_pointer(
        memory,
        module_base=memory.module_base_value,
        configured_player_pointer_offset=config.player_pointer_offset,
        monster_config=config,
        state=state,
        hints=hints,
        chunk_size=0x1000,
        timeout_seconds=2.0,
    ) is None
    assert recover_local_player_pointer(
        memory,
        module_base=memory.module_base_value,
        configured_player_pointer_offset=config.player_pointer_offset,
        monster_config=config,
        state=state,
        hints=hints,
        chunk_size=0x1000,
        timeout_seconds=2.0,
    ) is None

    metrics = state.metrics_for(memory.pid, memory.module_base_value)
    assert metrics is not None
    assert metrics.outcome == "movement_not_observed"
    assert (memory.pid, memory.module_base_value) in state.pending_candidates


def test_species_anchor_infers_shifted_actor_field_family() -> None:
    memory = AnchoredMemory()
    stale = NativeMonsterConfig(
        player_pointer_offset=0x1000,
        world_pointer_offset=0x1100,
        discovery_chunk_bytes=0x1000,
    )
    actual = replace(
        stale,
        species_offset=stale.species_offset + 0x40,
        active_species_offset=stale.active_species_offset + 0x40,
        hp_offset=stale.hp_offset + 0x40,
        x_offset=stale.x_offset + 0x40,
        y_offset=stale.y_offset + 0x40,
        z_offset=stale.z_offset + 0x40,
    )
    _populate(memory, actual)
    hints = PointerRecoveryHints(
        known_species_ids=(944, 948),
        player_spawn_x=253.0,
        player_spawn_z=86.0,
        player_current_hp=5000,
        player_max_hp=6000,
    )
    state = PointerRecoveryState()

    assert recover_local_player_pointer(
        memory,
        module_base=memory.module_base_value,
        configured_player_pointer_offset=stale.player_pointer_offset,
        monster_config=stale,
        state=state,
        hints=hints,
        chunk_size=0x1000,
        timeout_seconds=2.0,
    ) is None
    metrics = state.metrics_for(memory.pid, memory.module_base_value)
    assert metrics is not None
    assert metrics.outcome == "movement_required"
    assert metrics.monster_base_hypotheses >= 2
    assert metrics.inferred_species_offset == actual.species_offset
    assert metrics.inferred_active_species_offset == actual.active_species_offset
    assert metrics.inferred_hp_offset == actual.hp_offset
    assert metrics.inferred_x_offset == actual.x_offset

    memory.f32(memory.player_base + actual.x_offset, 258.0)
    recovery = recover_local_player_pointer(
        memory,
        module_base=memory.module_base_value,
        configured_player_pointer_offset=stale.player_pointer_offset,
        monster_config=stale,
        state=state,
        hints=hints,
        chunk_size=0x1000,
        timeout_seconds=2.0,
    )

    assert recovery is not None
    assert recovery.species_offset == actual.species_offset
    assert recovery.active_species_offset == actual.active_species_offset
    assert recovery.hp_offset == actual.hp_offset
    assert recovery.x_offset == actual.x_offset
    assert recovery.y_offset == actual.y_offset
    assert recovery.z_offset == actual.z_offset


def test_shared_float_bits_cannot_be_inferred_as_world_object() -> None:
    memory = AnchoredMemory()
    config = NativeMonsterConfig(
        player_pointer_offset=0x1000,
        world_pointer_offset=0x1100,
        discovery_chunk_bytes=0x1000,
    )
    _populate(memory, config)
    false_world_offset = 0x14C
    for base in (
        memory.heap_base,
        memory.heap_base + 0x3000,
        memory.player_base,
    ):
        memory.u32(base + false_world_offset, memory.scalar_world_base)
    for slot_offset in range(0x3300, 0x3320, 4):
        memory.u32(
            memory.module_base_value + slot_offset,
            memory.scalar_world_base,
        )
    # A module-valued literal inside the readable scalar page is insufficient:
    # it must itself lead to a table of module-owned function pointers.
    memory.u32(
        memory.scalar_world_base + 0x20,
        memory.module_base_value + 0x900,
    )
    hints = PointerRecoveryHints(
        known_species_ids=(944, 948),
        player_spawn_x=253.0,
        player_spawn_z=86.0,
        player_current_hp=5000,
        player_max_hp=6000,
    )
    state = PointerRecoveryState()

    assert recover_local_player_pointer(
        memory,
        module_base=memory.module_base_value,
        configured_player_pointer_offset=config.player_pointer_offset,
        monster_config=config,
        state=state,
        hints=hints,
        chunk_size=0x1000,
        timeout_seconds=2.0,
    ) is None

    metrics = state.metrics_for(memory.pid, memory.module_base_value)
    assert metrics is not None
    assert metrics.outcome == "movement_required"
    assert metrics.world_object_rejections >= 1
    assert metrics.world_identity_vtable_misses >= 1
    assert metrics.world_identity_module_pointer_fields >= 1
    assert metrics.inferred_world_offset == memory.new_world_offset


def test_world_identity_can_use_a_displaced_vtable_field() -> None:
    memory = AnchoredMemory()
    config = NativeMonsterConfig(
        player_pointer_offset=0x1000,
        world_pointer_offset=0x1100,
        discovery_chunk_bytes=0x1000,
    )
    _populate(memory, config)
    displaced_field = 0x2C
    memory.u32(memory.world_base, 0)
    memory.u32(
        memory.world_base + displaced_field,
        memory.module_base_value + 0x800,
    )
    hints = PointerRecoveryHints(
        known_species_ids=(944, 948),
        player_spawn_x=253.0,
        player_spawn_z=86.0,
        player_current_hp=5000,
        player_max_hp=6000,
    )
    state = PointerRecoveryState()

    assert recover_local_player_pointer(
        memory,
        module_base=memory.module_base_value,
        configured_player_pointer_offset=config.player_pointer_offset,
        monster_config=config,
        state=state,
        hints=hints,
        chunk_size=0x1000,
        timeout_seconds=2.0,
    ) is None
    first_metrics = state.metrics_for(memory.pid, memory.module_base_value)
    assert first_metrics is not None
    assert first_metrics.outcome == "movement_required"
    assert first_metrics.inferred_world_vtable_field_offset == displaced_field

    memory.f32(memory.player_base + config.x_offset, 258.0)
    recovery = recover_local_player_pointer(
        memory,
        module_base=memory.module_base_value,
        configured_player_pointer_offset=config.player_pointer_offset,
        monster_config=config,
        state=state,
        hints=hints,
        chunk_size=0x1000,
        timeout_seconds=2.0,
    )

    assert recovery is not None
    assert recovery.world_vtable_offset == 0x800
    assert recovery.world_vtable_field_offset == displaced_field


def test_pointer_rich_world_can_use_a_stable_module_marker() -> None:
    memory = AnchoredMemory()
    config = NativeMonsterConfig(
        player_pointer_offset=0x1000,
        world_pointer_offset=0x1100,
        discovery_chunk_bytes=0x1000,
    )
    _populate(memory, config)
    marker_field = 0x2C
    memory.u32(memory.world_base, 0)
    memory.u32(
        memory.world_base + marker_field,
        memory.module_base_value + 0x900,
    )
    for index, pointer in enumerate(
        (memory.heap_base, memory.heap_base + 0x3000, memory.player_base)
    ):
        memory.u32(memory.world_base + 0x40 + index * 4, pointer)
    for index in range(4):
        memory.u32(memory.world_base + 0x60 + index * 4, index + 1)
    hints = PointerRecoveryHints(
        known_species_ids=(944, 948),
        player_spawn_x=253.0,
        player_spawn_z=86.0,
        player_current_hp=5000,
        player_max_hp=6000,
    )
    service = NativeProcessService(memory, config, owns_memory=False)

    first = service.recover_pointers(hints=hints, timeout_seconds=2.0)

    assert first.outcome is NativeRecoveryOutcome.MOVEMENT_REQUIRED
    assert first.metrics.inferred_world_identity_kind == "module_marker"
    assert first.metrics.inferred_world_readable_pointer_fields >= 3
    assert first.metrics.inferred_world_distinct_values >= 8
    assert first.metrics.world_identity_marker_accepts >= 1
    memory.f32(memory.player_base + config.x_offset, 258.0)

    second = service.recover_pointers(hints=hints, timeout_seconds=2.0)

    assert second.outcome is NativeRecoveryOutcome.SUCCESS
    assert second.recovery is not None
    assert second.recovery.world_identity_kind == "module_marker"
    assert service.world_identity_kind == "module_marker"
    assert service.world_vtable_field_offset == marker_field
    assert service.read_pointer_snapshot().world_base == memory.world_base


def test_spawn_player_selects_the_right_structural_world_hypothesis() -> None:
    memory = AnchoredMemory()
    config = NativeMonsterConfig(
        player_pointer_offset=0x1000,
        world_pointer_offset=0x1100,
        discovery_chunk_bytes=0x1000,
    )
    _populate(memory, config)
    decoy_world = memory.heap_base + 0x9000
    decoy_field = config.world_offset
    memory.u32(decoy_world + 0x2C, memory.module_base_value + 0x900)
    for index, pointer in enumerate(
        (memory.heap_base, memory.heap_base + 0x3000, memory.player_base)
    ):
        memory.u32(decoy_world + 0x40 + index * 4, pointer)
    for index in range(5):
        memory.u32(decoy_world + 0x60 + index * 4, index + 1)
    for index in range(8):
        memory.u32(memory.module_base_value + 0x4000 + index * 4, decoy_world)
    for actor in (memory.heap_base, memory.heap_base + 0x3000):
        memory.u32(actor + decoy_field, decoy_world)
    hints = PointerRecoveryHints(
        known_species_ids=(944, 948),
        player_spawn_x=253.0,
        player_spawn_z=86.0,
        player_current_hp=5000,
        player_max_hp=6000,
    )
    state = PointerRecoveryState()

    assert recover_local_player_pointer(
        memory,
        module_base=memory.module_base_value,
        configured_player_pointer_offset=config.player_pointer_offset,
        monster_config=config,
        state=state,
        hints=hints,
        chunk_size=0x1000,
        timeout_seconds=2.0,
    ) is None

    metrics = state.metrics_for(memory.pid, memory.module_base_value)
    assert metrics is not None
    assert metrics.outcome == "movement_required"
    assert metrics.structural_world_hypotheses >= 2
    assert metrics.spawn_world_hypothesis_matches >= 1
    assert metrics.inferred_world_offset == memory.new_world_offset
    assert metrics.inferred_world_vtable == memory.module_base_value + 0x800


def test_player_can_be_linked_from_world_without_carrying_monster_world_field() -> None:
    memory = AnchoredMemory()
    config = NativeMonsterConfig(
        player_pointer_offset=0x1000,
        world_pointer_offset=0x1100,
        discovery_chunk_bytes=0x1000,
    )
    _populate(memory, config)
    memory.u32(memory.player_base + memory.new_world_offset, 0)
    memory.u32(memory.world_base + 0x40, memory.player_base)
    hints = PointerRecoveryHints(
        known_species_ids=(944, 948),
        player_spawn_x=253.0,
        player_spawn_z=86.0,
        player_current_hp=5000,
        player_max_hp=6000,
    )
    state = PointerRecoveryState()

    assert recover_local_player_pointer(
        memory,
        module_base=memory.module_base_value,
        configured_player_pointer_offset=config.player_pointer_offset,
        monster_config=config,
        state=state,
        hints=hints,
        chunk_size=0x1000,
        timeout_seconds=2.0,
    ) is None

    metrics = state.metrics_for(memory.pid, memory.module_base_value)
    assert metrics is not None
    assert metrics.outcome == "movement_required"
    assert metrics.spawn_hp_matches == 1
    assert metrics.stable_spawn_candidates == 1
    assert metrics.spawn_world_matches == 0
    assert metrics.player_world_rooted_matches == 1

    memory.f32(memory.player_base + config.x_offset, 258.0)
    recovery = recover_local_player_pointer(
        memory,
        module_base=memory.module_base_value,
        configured_player_pointer_offset=config.player_pointer_offset,
        monster_config=config,
        state=state,
        hints=hints,
        chunk_size=0x1000,
        timeout_seconds=2.0,
    )

    assert recovery is not None
    assert recovery.player_pointer_offset == memory.world_slot_offset
    assert recovery.player_pointer_chain_offsets == (0x40,)
    assert recovery.world_pointer_offset == memory.world_slot_offset
    assert recovery.world_pointer_chain_offsets == ()


def test_same_target_world_field_aliases_are_not_player_ambiguity() -> None:
    memory = AnchoredMemory()
    config = NativeMonsterConfig(
        player_pointer_offset=0x1000,
        world_pointer_offset=0x1100,
        discovery_chunk_bytes=0x1000,
    )
    _populate(memory, config)
    alias_field = config.world_offset
    for actor in (
        memory.heap_base,
        memory.heap_base + 0x3000,
        memory.player_base,
    ):
        memory.u32(actor + alias_field, memory.world_base)
    hints = PointerRecoveryHints(
        known_species_ids=(944, 948),
        player_spawn_x=253.0,
        player_spawn_z=86.0,
        player_current_hp=5000,
        player_max_hp=6000,
    )
    state = PointerRecoveryState()

    assert recover_local_player_pointer(
        memory,
        module_base=memory.module_base_value,
        configured_player_pointer_offset=config.player_pointer_offset,
        monster_config=config,
        state=state,
        hints=hints,
        chunk_size=0x1000,
        timeout_seconds=2.0,
    ) is None

    metrics = state.metrics_for(memory.pid, memory.module_base_value)
    assert metrics is not None
    assert metrics.outcome == "movement_required"
    assert metrics.structural_world_hypotheses >= 2
    assert metrics.spawn_world_hypothesis_matches >= 2
    assert metrics.ambiguous_candidates == 0


def test_player_specific_hp_field_does_not_replace_monster_hp_layout() -> None:
    memory = AnchoredMemory()
    config = NativeMonsterConfig(
        player_pointer_offset=0x1000,
        world_pointer_offset=0x1100,
        discovery_chunk_bytes=0x1000,
    )
    _populate(memory, config)
    player_hp_offset = config.hp_offset + 8
    player_max_hp_offset = config.hp_offset + 12
    memory.i32(memory.player_base + config.hp_offset, 0)
    memory.i32(memory.player_base + player_hp_offset, 5000)
    memory.i32(memory.player_base + player_max_hp_offset, 6000)
    hints = PointerRecoveryHints(
        known_species_ids=(944, 948),
        player_spawn_x=253.0,
        player_spawn_z=86.0,
        player_current_hp=5000,
        player_max_hp=6000,
    )
    state = PointerRecoveryState()

    assert recover_local_player_pointer(
        memory,
        module_base=memory.module_base_value,
        configured_player_pointer_offset=config.player_pointer_offset,
        monster_config=config,
        state=state,
        hints=hints,
        chunk_size=0x1000,
        timeout_seconds=2.0,
    ) is None
    memory.f32(memory.player_base + config.x_offset, 258.0)
    recovery = recover_local_player_pointer(
        memory,
        module_base=memory.module_base_value,
        configured_player_pointer_offset=config.player_pointer_offset,
        monster_config=config,
        state=state,
        hints=hints,
        chunk_size=0x1000,
        timeout_seconds=2.0,
    )

    assert recovery is not None
    assert recovery.hp_offset == config.hp_offset


def test_unique_three_actor_single_species_layout_is_sufficient() -> None:
    memory = AnchoredMemory()
    config = NativeMonsterConfig(
        player_pointer_offset=0x1000,
        world_pointer_offset=0x1100,
        discovery_chunk_bytes=0x1000,
    )
    _populate(memory, config)
    second = memory.heap_base + 0x3000
    memory.i32(second + config.species_offset, 944)
    memory.i32(second + config.active_species_offset, 944)
    third = memory.heap_base + 0x9000
    memory.u32(third + memory.new_self_offset, third)
    memory.u32(third + memory.new_world_offset, memory.world_base)
    memory.i32(third + config.species_offset, 944)
    memory.i32(third + config.active_species_offset, 944)
    memory.i32(third + config.hp_offset, 900)
    memory.f32(third + config.x_offset, 270.0)
    memory.f32(third + config.y_offset, 12.0)
    memory.f32(third + config.z_offset, 95.0)
    hints = PointerRecoveryHints(
        known_species_ids=(944, 948),
        player_spawn_x=253.0,
        player_spawn_z=86.0,
        player_current_hp=5000,
        player_max_hp=6000,
    )
    state = PointerRecoveryState()

    assert recover_local_player_pointer(
        memory,
        module_base=memory.module_base_value,
        configured_player_pointer_offset=config.player_pointer_offset,
        monster_config=config,
        state=state,
        hints=hints,
        chunk_size=0x1000,
        timeout_seconds=2.0,
    ) is None

    metrics = state.metrics_for(memory.pid, memory.module_base_value)
    assert metrics is not None
    assert metrics.outcome == "movement_required"
    assert metrics.monster_candidates == 3
    assert metrics.monster_layout_species_support == 1
    assert metrics.monster_layout_ties == 0


def test_repeated_self_fields_are_one_layout_with_validated_aliases() -> None:
    memory = AnchoredMemory()
    config = NativeMonsterConfig(
        player_pointer_offset=0x1000,
        world_pointer_offset=0x1100,
        discovery_chunk_bytes=0x1000,
    )
    _populate(memory, config)
    aliases = (0x1E90, memory.new_self_offset, 0x1F10, 0x1F30)
    actor_bases = (
        memory.heap_base,
        memory.heap_base + 0x3000,
        memory.player_base,
    )
    for base in actor_bases:
        for offset in aliases:
            memory.u32(base + offset, base)
    hints = PointerRecoveryHints(
        known_species_ids=(944, 948),
        player_spawn_x=253.0,
        player_spawn_z=86.0,
        player_current_hp=5000,
        player_max_hp=6000,
    )
    state = PointerRecoveryState()

    assert recover_local_player_pointer(
        memory,
        module_base=memory.module_base_value,
        configured_player_pointer_offset=config.player_pointer_offset,
        monster_config=config,
        state=state,
        hints=hints,
        chunk_size=0x1000,
        timeout_seconds=2.0,
    ) is None

    metrics = state.metrics_for(memory.pid, memory.module_base_value)
    assert metrics is not None
    assert metrics.outcome == "movement_required"
    assert metrics.monster_layout_ties == 0
    assert metrics.monster_self_field_aliases == len(aliases)
    assert metrics.inferred_self_offset in aliases

    memory.f32(memory.player_base + config.x_offset, 258.0)
    recovery = recover_local_player_pointer(
        memory,
        module_base=memory.module_base_value,
        configured_player_pointer_offset=config.player_pointer_offset,
        monster_config=config,
        state=state,
        hints=hints,
        chunk_size=0x1000,
        timeout_seconds=2.0,
    )
    assert recovery is not None
    assert recovery.movement_validated
    assert recovery.self_pointer_offset in aliases


def test_distinct_actor_field_families_remain_ambiguous() -> None:
    memory = AnchoredMemory()
    config = NativeMonsterConfig(
        player_pointer_offset=0x1000,
        world_pointer_offset=0x1100,
        discovery_chunk_bytes=0x1000,
    )
    _populate(memory, config)
    shift = 0x40
    for base in (memory.heap_base, memory.heap_base + 0x3000):
        data = memory.read(base, 0x4000)
        memory.i32(base + config.species_offset + shift, struct.unpack_from(
            "<i", data, config.species_offset
        )[0])
        memory.i32(base + config.active_species_offset + shift, struct.unpack_from(
            "<i", data, config.active_species_offset
        )[0])
        memory.i32(base + config.hp_offset + shift, struct.unpack_from(
            "<i", data, config.hp_offset
        )[0])
        for offset in (config.x_offset, config.y_offset, config.z_offset):
            memory.f32(base + offset + shift, struct.unpack_from("<f", data, offset)[0])
    hints = PointerRecoveryHints(
        known_species_ids=(944, 948),
        player_spawn_x=253.0,
        player_spawn_z=86.0,
        player_current_hp=5000,
        player_max_hp=6000,
    )
    state = PointerRecoveryState()

    assert recover_local_player_pointer(
        memory,
        module_base=memory.module_base_value,
        configured_player_pointer_offset=config.player_pointer_offset,
        monster_config=config,
        state=state,
        hints=hints,
        chunk_size=0x1000,
        timeout_seconds=2.0,
    ) is None

    metrics = state.metrics_for(memory.pid, memory.module_base_value)
    assert metrics is not None
    assert metrics.outcome == "actor_layout_inconclusive"
    assert metrics.monster_layout_ties >= 1


def test_service_applies_one_level_player_chain_after_movement() -> None:
    memory = AnchoredMemory()
    config = NativeMonsterConfig(
        player_pointer_offset=0x1000,
        world_pointer_offset=0x1100,
        discovery_chunk_bytes=0x1000,
    )
    _populate(memory, config)
    holder = memory.heap_base + 0x9000
    memory.u32(
        memory.module_base_value + memory.player_slot_offset,
        holder,
    )
    memory.u32(holder + 0x20, memory.player_base)
    hints = PointerRecoveryHints(
        known_species_ids=(944, 948),
        player_spawn_x=253.0,
        player_spawn_z=86.0,
        player_current_hp=5000,
        player_max_hp=6000,
    )
    service = NativeProcessService(memory, config, owns_memory=False)

    first = service.recover_pointers(hints=hints, timeout_seconds=2.0)

    assert first.outcome is NativeRecoveryOutcome.MOVEMENT_REQUIRED
    assert not first.applied
    memory.f32(memory.player_base + config.z_offset, 91.0)

    second = service.recover_pointers(hints=hints, timeout_seconds=2.0)

    assert second.outcome is NativeRecoveryOutcome.SUCCESS
    assert second.applied
    assert second.recovery is not None
    assert second.recovery.player_pointer_chain_offsets == (0x20,)
    snapshot = service.read_pointer_snapshot()
    assert snapshot.player_base == memory.player_base
    assert snapshot.world_base == memory.world_base


def test_service_uses_confirmed_world_as_player_chain_root() -> None:
    memory = AnchoredMemory()
    config = NativeMonsterConfig(
        player_pointer_offset=0x1000,
        world_pointer_offset=0x1100,
        discovery_chunk_bytes=0x1000,
    )
    _populate(memory, config)
    player_world_offset = 0x280
    player_world_alias_offset = 0x2A0
    world_slot_alias_offset = 0x3200
    memory.u32(memory.module_base_value + memory.player_slot_offset, 0)
    memory.u32(
        memory.module_base_value + world_slot_alias_offset,
        memory.world_base,
    )
    memory.u32(memory.world_base + player_world_offset, memory.player_base)
    memory.u32(memory.world_base + player_world_alias_offset, memory.player_base)
    hints = PointerRecoveryHints(
        known_species_ids=(944, 948),
        player_spawn_x=253.0,
        player_spawn_z=86.0,
        player_current_hp=5000,
        player_max_hp=6000,
    )
    service = NativeProcessService(memory, config, owns_memory=False)

    first = service.recover_pointers(hints=hints, timeout_seconds=2.0)

    assert first.outcome is NativeRecoveryOutcome.MOVEMENT_REQUIRED
    first_metrics = first.metrics
    assert first_metrics.player_reference_matches >= 1
    assert first_metrics.player_world_chain_candidates == 2
    memory.f32(memory.player_base + config.x_offset, 258.0)

    second = service.recover_pointers(hints=hints, timeout_seconds=2.0)

    assert second.outcome is NativeRecoveryOutcome.SUCCESS
    assert second.applied
    assert second.recovery is not None
    assert second.recovery.player_pointer_offset == memory.world_slot_offset
    assert second.recovery.player_pointer_chain_offsets == (player_world_offset,)
    assert second.recovery.world_pointer_offset == memory.world_slot_offset
    snapshot = service.read_pointer_snapshot()
    assert snapshot.player_base == memory.player_base
    assert snapshot.world_base == memory.world_base


def test_service_rejects_changed_world_identity_after_anchored_recovery() -> None:
    memory = AnchoredMemory()
    config = NativeMonsterConfig(
        player_pointer_offset=0x1000,
        world_pointer_offset=0x1100,
        discovery_chunk_bytes=0x1000,
    )
    _populate(memory, config)
    displaced_field = 0x2C
    memory.u32(memory.world_base, 0)
    memory.u32(
        memory.world_base + displaced_field,
        memory.module_base_value + 0x800,
    )
    hints = PointerRecoveryHints(
        known_species_ids=(944, 948),
        player_spawn_x=253.0,
        player_spawn_z=86.0,
        player_current_hp=5000,
        player_max_hp=6000,
    )
    service = NativeProcessService(memory, config, owns_memory=False)

    first = service.recover_pointers(hints=hints, timeout_seconds=2.0)
    assert first.outcome is NativeRecoveryOutcome.MOVEMENT_REQUIRED
    memory.f32(memory.player_base + config.z_offset, 91.0)
    second = service.recover_pointers(hints=hints, timeout_seconds=2.0)
    assert second.outcome is NativeRecoveryOutcome.SUCCESS
    assert service.world_vtable_offset == 0x800
    assert service.world_vtable_field_offset == displaced_field
    assert service.read_pointer_snapshot().world_base == memory.world_base

    memory.u32(
        memory.world_base + displaced_field,
        memory.module_base_value + 0x804,
    )

    with pytest.raises(
        NativePointerSnapshotError,
        match="world object identity",
    ):
        service.read_pointer_snapshot()


def test_service_enforces_persisted_world_identity_after_restart() -> None:
    memory = AnchoredMemory()
    discovery_config = NativeMonsterConfig(discovery_chunk_bytes=0x1000)
    _populate(memory, discovery_config)
    displaced_field = 0x2C
    memory.u32(memory.world_base, 0)
    memory.u32(
        memory.world_base + displaced_field,
        memory.module_base_value + 0x800,
    )
    persisted_config = replace(
        discovery_config,
        player_pointer_offset=memory.player_slot_offset,
        world_pointer_offset=memory.world_slot_offset,
        world_offset=memory.new_world_offset,
        world_vtable_offset=0x800,
        world_vtable_field_offset=displaced_field,
        self_pointer_offset=memory.new_self_offset,
    )
    service = NativeProcessService(memory, persisted_config, owns_memory=False)

    assert service.read_pointer_snapshot().world_base == memory.world_base

    memory.u32(memory.world_base + displaced_field, memory.scalar_world_base)

    with pytest.raises(
        NativePointerSnapshotError,
        match="world object identity",
    ):
        service.read_pointer_snapshot()


def test_service_applies_inferred_shifted_actor_layout() -> None:
    memory = AnchoredMemory()
    stale = NativeMonsterConfig(
        player_pointer_offset=0x1000,
        world_pointer_offset=0x1100,
        discovery_chunk_bytes=0x1000,
    )
    actual = replace(
        stale,
        species_offset=stale.species_offset + 0x40,
        active_species_offset=stale.active_species_offset + 0x40,
        hp_offset=stale.hp_offset + 0x40,
        x_offset=stale.x_offset + 0x40,
        y_offset=stale.y_offset + 0x40,
        z_offset=stale.z_offset + 0x40,
    )
    _populate(memory, actual)
    hints = PointerRecoveryHints(
        known_species_ids=(944, 948),
        player_spawn_x=253.0,
        player_spawn_z=86.0,
        player_current_hp=5000,
        player_max_hp=6000,
    )
    service = NativeProcessService(memory, stale, owns_memory=False)

    first = service.recover_pointers(hints=hints, timeout_seconds=2.0)
    assert first.outcome is NativeRecoveryOutcome.MOVEMENT_REQUIRED
    memory.f32(memory.player_base + actual.z_offset, 91.0)
    second = service.recover_pointers(hints=hints, timeout_seconds=2.0)

    assert second.outcome is NativeRecoveryOutcome.SUCCESS
    assert service.species_offset == actual.species_offset
    assert service.active_species_offset == actual.active_species_offset
    assert service.hp_offset == actual.hp_offset
    assert (service.x_offset, service.y_offset, service.z_offset) == (
        actual.x_offset,
        actual.y_offset,
        actual.z_offset,
    )
    assert service.read_pointer_snapshot().player_base == memory.player_base


def test_explicit_anchor_does_not_persist_until_movement(
    tmp_path: Path,
) -> None:
    memory = AnchoredMemory()
    config = NativeMonsterConfig(
        player_pointer_offset=0x1000,
        world_pointer_offset=0x1100,
        discovery_chunk_bytes=0x1000,
    )
    _populate(memory, config)
    hints = PointerRecoveryHints(
        known_species_ids=(944, 948),
        player_spawn_x=253.0,
        player_spawn_z=86.0,
        player_current_hp=5000,
        player_max_hp=6000,
    )
    position_path = tmp_path / "native_position.json"
    monster_path = tmp_path / "native_monsters.json"
    original_position = {"pointer_offset": "0x1000"}
    original_monster = {
        "player_pointer_offset": "0x1000",
        "world_pointer_offset": "0x1100",
        "layout": {
            "world_offset": "0x16C",
            "self_pointer_offset": "0x1EE0",
        },
    }
    position_path.write_text(json.dumps(original_position), encoding="utf-8")
    monster_path.write_text(json.dumps(original_monster), encoding="utf-8")
    state = PointerRecoveryState()

    first = recover_local_player_pointer(
        memory,
        module_base=memory.module_base_value,
        configured_player_pointer_offset=config.player_pointer_offset,
        monster_config=config,
        state=state,
        hints=hints,
        chunk_size=0x1000,
        timeout_seconds=2.0,
        persist=True,
        position_config_path=position_path,
        monster_config_path=monster_path,
    )

    assert first is None
    assert json.loads(position_path.read_text()) == original_position
    assert json.loads(monster_path.read_text()) == original_monster

    memory.f32(memory.player_base + config.x_offset, 258.0)
    second = recover_local_player_pointer(
        memory,
        module_base=memory.module_base_value,
        configured_player_pointer_offset=config.player_pointer_offset,
        monster_config=config,
        state=state,
        hints=hints,
        chunk_size=0x1000,
        timeout_seconds=2.0,
        persist=True,
        position_config_path=position_path,
        monster_config_path=monster_path,
    )

    assert second is not None
    position = json.loads(position_path.read_text())
    assert position["pointer_offset"] == "0x3000"
    assert position["layout"]["x_offset"] == "0x160"
    monster = json.loads(monster_path.read_text())
    assert monster["world_pointer_offset"] == "0x3100"
    assert monster["layout"]["world_offset"] == "0x180"
    assert monster["layout"]["self_pointer_offset"] == "0x1EF0"
    assert monster["layout"]["species_offset"] == "0x174"
    assert monster["layout"]["active_species_offset"] == "0x1DBC"
    assert monster["layout"]["hp_offset"] == "0x814"
