from __future__ import annotations

import struct
from dataclasses import dataclass

from position.MonsterConfig import NativeMonsterConfig
from position.NativePointerRecovery import recover_local_player_pointer


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

    def readable_regions(self, *, maximum_address=0x7FFFFFFF, private_only=True):
        del private_only
        regions = [
            Region(self.module_base_value, len(self.module)),
            Region(self.actor_base, len(self.actor), region_type=0x20000),
        ]
        return tuple(r for r in regions if r.base_address <= maximum_address)

    def read(self, address: int, size: int) -> bytes:
        if self.module_base_value <= address < self.module_base_value + len(self.module):
            start = address - self.module_base_value
            return bytes(self.module[start : start + size])
        if self.actor_base <= address < self.actor_base + len(self.actor):
            start = address - self.actor_base
            return bytes(self.actor[start : start + size])
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
    world = 0x24000000

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
