from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

from position.AnchoredPointerDiscovery import PointerRecoveryHints
from position.MonsterConfig import NativeMonsterConfig
from position.native_process_service import (
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
        ):
            if base <= address < base + len(data):
                struct.pack_into(format_string, data, address - base, value)
                return
        raise RuntimeError(f"unwritable 0x{address:X}")


def _populate(memory: AnchoredMemory, config: NativeMonsterConfig) -> None:
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
    assert json.loads(position_path.read_text())["pointer_offset"] == "0x3000"
    monster = json.loads(monster_path.read_text())
    assert monster["world_pointer_offset"] == "0x3100"
    assert monster["layout"]["world_offset"] == "0x180"
    assert monster["layout"]["self_pointer_offset"] == "0x1EF0"
