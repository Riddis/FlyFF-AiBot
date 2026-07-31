from __future__ import annotations

import struct
from dataclasses import dataclass

from position.MonsterConfig import NativeMonsterConfig
from position.NativePointerRecovery import (
    PointerRecoveryState,
    recover_local_player_pointer,
)
from position.Win32ProcessMemory import ModuleInfo


@dataclass(frozen=True)
class Region:
    base_address: int
    size: int
    protection: int = 0x04
    region_type: int = 0x1000000


class RecoveryMemory:
    def __init__(self) -> None:
        self.pid = 1234
        self.module_base_value = 0x00400000
        self.module = bytearray(0x5000)
        self.actor = bytearray(0x3000)
        self.actor_base = 0x20000000
        self.world = bytearray(0x1000)
        self.world_base = 0x24000000
        self.extra_regions: dict[int, bytearray] = {}

    def module_info(self, module_name: str) -> ModuleInfo:
        assert module_name == "Neuz.exe"
        return ModuleInfo(
            name=module_name,
            path=r"C:\FlyFF\Neuz.exe",
            base_address=self.module_base_value,
            size=len(self.module),
        )

    def readable_regions(self, *, maximum_address=0x7FFFFFFF, private_only=True):
        del private_only
        regions = [
            Region(self.module_base_value, len(self.module)),
            Region(self.actor_base, len(self.actor), region_type=0x20000),
            Region(self.world_base, len(self.world), region_type=0x20000),
        ]
        regions.extend(
            Region(base, len(data), region_type=0x20000)
            for base, data in self.extra_regions.items()
        )
        return tuple(r for r in regions if r.base_address <= maximum_address)

    def read(self, address: int, size: int) -> bytes:
        if self.module_base_value <= address < self.module_base_value + len(self.module):
            start = address - self.module_base_value
            return bytes(self.module[start : start + size])
        if self.actor_base <= address < self.actor_base + len(self.actor):
            start = address - self.actor_base
            return bytes(self.actor[start : start + size])
        if self.world_base <= address < self.world_base + len(self.world):
            start = address - self.world_base
            return bytes(self.world[start : start + size])
        for base, data in self.extra_regions.items():
            if base <= address < base + len(data):
                start = address - base
                return bytes(data[start : start + size])
        raise RuntimeError(f"unreadable 0x{address:X}")

    def write_module_u32(self, offset: int, value: int) -> None:
        struct.pack_into("<I", self.module, offset, value)

    def write_actor_u32(self, offset: int, value: int) -> None:
        struct.pack_into("<I", self.actor, offset, value)

    def write_actor_i32(self, offset: int, value: int) -> None:
        struct.pack_into("<i", self.actor, offset, value)

    def write_actor_f32(self, offset: int, value: float) -> None:
        struct.pack_into("<f", self.actor, offset, value)


def test_recovers_equally_shifted_player_and_world_globals() -> None:
    memory = RecoveryMemory()
    config = NativeMonsterConfig(
        player_pointer_offset=0x1000,
        world_pointer_offset=0x1100,
        discovery_chunk_bytes=4096,
    )
    shift = 0x280
    player_slot = config.player_pointer_offset + shift
    world_slot = config.world_pointer_offset + shift
    player = memory.actor_base
    world = memory.world_base

    memory.write_module_u32(player_slot, player)
    memory.write_module_u32(world_slot, world)
    memory.write_actor_u32(config.self_pointer_offset, player)
    memory.write_actor_u32(config.world_offset, world)
    memory.write_actor_i32(config.species_offset, 1)
    memory.write_actor_i32(config.active_species_offset, 0)
    memory.write_actor_i32(config.hp_offset, 1000)
    memory.write_actor_f32(config.x_offset, 10.0)
    memory.write_actor_f32(config.y_offset, 20.0)
    memory.write_actor_f32(config.z_offset, 30.0)

    result = recover_local_player_pointer(
        memory,
        module_base=memory.module_base_value,
        configured_player_pointer_offset=config.player_pointer_offset,
        monster_config=config,
        search_radii=(0x400,),
        chunk_size=0x1000,
        persist=False,
    )

    assert result is not None
    assert result.player_pointer_offset == player_slot
    assert result.world_pointer_offset == world_slot
    assert result.player_base == player
    assert result.world_base == world
    assert result.strategy == "module_image"


def test_module_scan_recovers_independently_moved_player_and_world_slots() -> None:
    memory = RecoveryMemory()
    config = NativeMonsterConfig(
        player_pointer_offset=0x800,
        world_pointer_offset=0x900,
        discovery_chunk_bytes=4096,
    )
    player_slot = 0x2A80
    world_slot = 0x3B40
    player = memory.actor_base
    world = memory.world_base

    memory.write_module_u32(player_slot, player)
    memory.write_module_u32(world_slot, world)
    memory.write_actor_u32(config.self_pointer_offset, player)
    memory.write_actor_u32(config.world_offset, world)
    memory.write_actor_i32(config.species_offset, 1)
    memory.write_actor_i32(config.active_species_offset, 0)
    memory.write_actor_i32(config.hp_offset, 1000)
    memory.write_actor_f32(config.x_offset, 10.0)
    memory.write_actor_f32(config.y_offset, 20.0)
    memory.write_actor_f32(config.z_offset, 30.0)

    result = recover_local_player_pointer(
        memory,
        module_base=memory.module_base_value,
        configured_player_pointer_offset=config.player_pointer_offset,
        monster_config=config,
        search_radii=(0x1000,),
        chunk_size=0x1000,
        timeout_seconds=1.0,
    )

    assert result is not None
    assert result.player_pointer_offset == player_slot
    assert result.world_pointer_offset == world_slot
    assert result.strategy == "module_image"


def test_module_scan_rejects_equally_supported_player_candidates() -> None:
    memory = RecoveryMemory()
    config = NativeMonsterConfig(
        player_pointer_offset=0x800,
        world_pointer_offset=0x900,
        discovery_chunk_bytes=4096,
    )
    second_player = 0x21000000
    second_world = 0x25000000
    memory.extra_regions[second_player] = bytearray(0x3000)
    memory.extra_regions[second_world] = bytearray(0x1000)

    candidates = (
        (0x2A80, 0x3A80, memory.actor_base, memory.world_base, memory.actor),
        (
            0x2C80,
            0x3C80,
            second_player,
            second_world,
            memory.extra_regions[second_player],
        ),
    )
    for player_slot, world_slot, player, world, actor in candidates:
        memory.write_module_u32(player_slot, player)
        memory.write_module_u32(world_slot, world)
        struct.pack_into("<I", actor, config.self_pointer_offset, player)
        struct.pack_into("<I", actor, config.world_offset, world)
        struct.pack_into("<i", actor, config.species_offset, 1)
        struct.pack_into("<i", actor, config.active_species_offset, 0)
        struct.pack_into("<i", actor, config.hp_offset, 1000)
        struct.pack_into("<f", actor, config.x_offset, 10.0)
        struct.pack_into("<f", actor, config.y_offset, 20.0)
        struct.pack_into("<f", actor, config.z_offset, 30.0)

    state = PointerRecoveryState()
    result = recover_local_player_pointer(
        memory,
        module_base=memory.module_base_value,
        configured_player_pointer_offset=config.player_pointer_offset,
        monster_config=config,
        state=state,
        chunk_size=0x1000,
        timeout_seconds=1.0,
    )

    assert result is None
    metrics = state.metrics_for(memory.pid, memory.module_base_value)
    assert metrics is not None
    assert metrics.ambiguous_candidates == 2


def test_module_scan_rejects_actor_like_false_positive() -> None:
    memory = RecoveryMemory()
    config = NativeMonsterConfig(
        player_pointer_offset=0x800,
        world_pointer_offset=0x900,
        discovery_chunk_bytes=4096,
    )
    memory.write_module_u32(0x2A80, memory.actor_base)
    memory.write_module_u32(0x3B40, memory.world_base)
    memory.write_actor_u32(config.self_pointer_offset, memory.actor_base)
    memory.write_actor_u32(config.world_offset, memory.world_base)
    memory.write_actor_i32(config.species_offset, 77)
    memory.write_actor_i32(config.active_species_offset, 77)
    memory.write_actor_i32(config.hp_offset, 1000)
    memory.write_actor_f32(config.x_offset, 10.0)
    memory.write_actor_f32(config.y_offset, 20.0)
    memory.write_actor_f32(config.z_offset, 30.0)
    state = PointerRecoveryState()

    result = recover_local_player_pointer(
        memory,
        module_base=memory.module_base_value,
        configured_player_pointer_offset=config.player_pointer_offset,
        monster_config=config,
        state=state,
        chunk_size=0x1000,
        timeout_seconds=1.0,
    )

    assert result is None
    metrics = state.metrics_for(memory.pid, memory.module_base_value)
    assert metrics is not None
    assert metrics.non_player_rejections == 1


def test_self_mismatch_records_bounded_structural_near_match() -> None:
    memory = RecoveryMemory()
    config = NativeMonsterConfig(
        player_pointer_offset=0x800,
        world_pointer_offset=0x900,
        discovery_chunk_bytes=4096,
    )
    memory.write_module_u32(0x2A80, memory.actor_base)
    memory.write_module_u32(0x3B40, memory.world_base)
    # Deliberately leave the configured self field stale while preserving an
    # otherwise coherent player-shaped actor at the old layout.
    memory.write_actor_u32(config.self_pointer_offset, 0)
    memory.write_actor_u32(config.world_offset, memory.world_base)
    memory.write_actor_i32(config.species_offset, 1)
    memory.write_actor_i32(config.active_species_offset, 0)
    memory.write_actor_i32(config.hp_offset, 1000)
    memory.write_actor_f32(config.x_offset, 10.0)
    memory.write_actor_f32(config.y_offset, 20.0)
    memory.write_actor_f32(config.z_offset, 30.0)
    state = PointerRecoveryState()

    result = recover_local_player_pointer(
        memory,
        module_base=memory.module_base_value,
        configured_player_pointer_offset=config.player_pointer_offset,
        monster_config=config,
        state=state,
        chunk_size=0x1000,
        timeout_seconds=1.0,
    )

    assert result is None
    metrics = state.metrics_for(memory.pid, memory.module_base_value)
    assert metrics is not None
    assert metrics.self_mismatch_rejections == 1
    assert metrics.self_mismatch_near_probed == 1
    assert metrics.self_mismatch_world_nonzero == 1
    assert metrics.self_mismatch_world_module_ref == 1
    assert metrics.self_mismatch_coordinate_plausible == 1
    assert metrics.self_mismatch_hp_positive == 1
    assert metrics.self_mismatch_player_like == 1
    assert metrics.self_mismatch_full_near_matches == 1
