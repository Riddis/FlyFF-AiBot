from __future__ import annotations

import struct
from dataclasses import dataclass

from position.NativeTraceTargets import discover_trace_targets
from position.PointerScanWorkflow import ReadableRegionIndex
from position.Win32ProcessMemory import MemoryRegion, ModuleInfo


@dataclass
class _Segment:
    base: int
    data: bytearray


class _Memory:
    def __init__(self, segments: tuple[_Segment, ...]) -> None:
        self.segments = segments

    def read(self, address: int, size: int) -> bytes:
        for segment in self.segments:
            relative = address - segment.base
            if 0 <= relative and relative + size <= len(segment.data):
                return bytes(segment.data[relative : relative + size])
        raise RuntimeError(f"unmapped 0x{address:X}+0x{size:X}")


def _write_actor(
    heap: bytearray,
    heap_base: int,
    base: int,
    *,
    species: int,
    hp: int,
    hp_offset: int,
    x: float,
    z: float,
    self_offset: int = 0x1EE0,
) -> None:
    relative = base - heap_base
    struct.pack_into("<f", heap, relative + 0x160, x)
    struct.pack_into("<f", heap, relative + 0x164, 10.0)
    struct.pack_into("<f", heap, relative + 0x168, z)
    struct.pack_into("<i", heap, relative + 0x174, species)
    if species > 0:
        struct.pack_into("<i", heap, relative + 0x1DBC, species)
    struct.pack_into("<i", heap, relative + hp_offset, hp)
    struct.pack_into("<I", heap, relative + self_offset, base)


def _discover(*, monster_hp: int = 400236):
    heap_base = 0x20000
    monster_a = heap_base
    monster_b = heap_base + 0x5000
    player_base = heap_base + 0xA000
    module_base = 0x400000
    heap = bytearray(0x10000)
    # Deliberately differ from configured +0x814 to prove exact HP offset
    # discovery rather than fixed-layout acceptance.
    monster_hp_offset = 0xA20
    player_hp_offset = 0x920
    _write_actor(
        heap,
        heap_base,
        monster_a,
        species=944,
        hp=monster_hp,
        hp_offset=monster_hp_offset,
        x=260.0,
        z=90.0,
    )
    _write_actor(
        heap,
        heap_base,
        monster_b,
        species=944,
        hp=monster_hp,
        hp_offset=monster_hp_offset,
        x=245.0,
        z=82.0,
    )
    _write_actor(
        heap,
        heap_base,
        player_base,
        species=0,
        hp=38857,
        hp_offset=player_hp_offset,
        x=253.0,
        z=86.0,
        # Deliberately differ from every monster self offset. The player class
        # must discover its own self alias rather than inheriting actor layout.
        self_offset=0x2A0,
    )
    module = bytearray(0x2000)
    struct.pack_into("<I", module, 0x1234, player_base)
    memory = _Memory(
        (
            _Segment(heap_base, heap),
            _Segment(module_base, module),
        )
    )
    regions = (
        MemoryRegion(heap_base, len(heap), 0x04, 0x20000),
        MemoryRegion(module_base, len(module), 0x04, 0x1000000),
    )
    result = discover_trace_targets(
        memory,
        regions=regions,
        readable=ReadableRegionIndex.build(regions),
        module=ModuleInfo("Neuz.exe", "", module_base, len(module)),
        species_hp={944: 400236},
        spawn_x=253.0,
        spawn_z=86.0,
        player_hp=38857,
        species_offset=0x174,
        active_species_offset=0x1DBC,
        hp_offset=0x814,
        x_offset=0x160,
        y_offset=0x164,
        z_offset=0x168,
        self_pointer_offset=0x1EE0,
        coordinate_limit=10000.0,
        chunk_size=0x1000,
        maximum_scan_bytes=0x20000,
        stability_delay_seconds=0.0,
    )
    return result, player_base, monster_a, monster_b, module_base


def test_dynamic_trace_target_discovery_finds_player_monsters_and_module_alias() -> None:
    result, player_base, monster_a, monster_b, module_base = _discover()

    assert result.outcome == "success"
    assert result.player is not None
    assert result.player.base == player_base
    assert result.player.hp_offset == 0x920
    assert result.player.self_pointer_offsets[0] == 0x2A0
    assert result.player.direct_module_slots == (module_base + 0x1234,)
    assert tuple(item.base for item in result.monsters) == (monster_a, monster_b)
    assert {item.hp_offset for item in result.monsters} == {0xA20}
    assert result.evidence.inferred_monster_hp_offset == 0xA20


def test_exact_monster_hp_is_not_relaxed_during_dynamic_offset_search() -> None:
    result, *_ = _discover(monster_hp=399999)

    assert result.outcome == "monster_consensus_not_found"
    assert result.evidence.monster_candidates == 0
    assert result.evidence.monster_hp_rejections >= 2
