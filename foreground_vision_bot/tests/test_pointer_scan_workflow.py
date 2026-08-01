from __future__ import annotations

import struct
from dataclasses import replace

from position.PointerScanWorkflow import (
    MovementEvidence,
    PointerPath,
    PointerScanSnapshot,
    ReadableRegionIndex,
    rank_stable_paths,
    resolve_pointer_path,
    scan_module_pointer_index,
    scan_module_rooted_paths,
)
from position.Win32ProcessMemory import MemoryRegion, ModuleInfo


class FakeMemory:
    def __init__(self) -> None:
        self.pid = 77
        self.module_base = 0x100000
        self.module_size = 0x200
        self.manager = 0x200000
        self.player = 0x300000
        self._segments: dict[int, bytearray] = {
            self.module_base: bytearray(self.module_size),
            self.manager: bytearray(0x100),
            self.player: bytearray(0x100),
        }
        self.u32(self.module_base + 0x20, self.player)
        self.u32(self.module_base + 0x40, self.manager)
        self.u32(self.manager + 0x30, self.player)

    def u32(self, address: int, value: int) -> None:
        base, segment = self._segment(address, 4)
        struct.pack_into("<I", segment, address - base, value)

    def _segment(self, address: int, size: int) -> tuple[int, bytearray]:
        for base, data in self._segments.items():
            if base <= address and address + size <= base + len(data):
                return base, data
        raise OSError(f"unreadable 0x{address:X}+0x{size:X}")

    def read(self, address: int, size: int) -> bytes:
        base, segment = self._segment(address, size)
        start = address - base
        return bytes(segment[start : start + size])


def _regions(memory: FakeMemory) -> tuple[MemoryRegion, ...]:
    return (
        MemoryRegion(memory.module_base, memory.module_size, 4, 0x1000000),
        MemoryRegion(memory.manager, 0x100, 4, 0x20000),
        MemoryRegion(memory.player, 0x100, 4, 0x20000),
    )


def test_module_pointer_workflow_finds_direct_and_one_hop_paths() -> None:
    memory = FakeMemory()
    module = ModuleInfo(
        "Neuz.exe",
        "C:/Games/Neuz.exe",
        memory.module_base,
        memory.module_size,
    )
    readable = ReadableRegionIndex.build(_regions(memory))

    index = scan_module_pointer_index(
        memory,
        module,
        readable,
        chunk_size=0x1000,
    )
    paths = scan_module_rooted_paths(
        memory,
        module,
        readable,
        index,
        memory.player,
        maximum_depth=2,
        field_span=0x100,
        maximum_roots=100,
        maximum_nodes=100,
    )

    assert PointerPath(0x20, ()) in paths
    assert PointerPath(0x40, (0x30,)) in paths
    assert all(
        resolve_pointer_path(memory, memory.module_base, path) == memory.player
        for path in paths
    )


def _snapshot(*paths: PointerPath, confirmed: bool = True) -> PointerScanSnapshot:
    movement = MovementEvidence(
        confirmed=confirmed,
        distance_native=5.0,
        final_x=258.0,
        final_y=0.0,
        final_z=86.0,
        samples=3,
        message="ok",
    )
    return PointerScanSnapshot(
        schema_version=1,
        captured_at_utc="2026-08-01T00:00:00+00:00",
        pid=1,
        module_name="Neuz.exe",
        module_path="C:/Games/Neuz.exe",
        module_size=0x943000,
        module_base=0xB0000,
        player_base=0x300000,
        player_fields={},
        baseline={},
        movement={
            "confirmed": movement.confirmed,
            "distance_native": movement.distance_native,
        },
        direct_module_offsets=(),
        pointer_paths=tuple(
            {
                "root_module_offset": path.root_module_offset,
                "field_offsets": list(path.field_offsets),
                "signature": path.signature,
                "depth": path.depth,
            }
            for path in paths
        ),
        anchor_outcome="spawn_player_not_found",
        monster_hp_anchors=((944, 400236),),
    )


def test_history_ranking_normalizes_addresses_across_sessions() -> None:
    stable = PointerPath(0x512340, ())
    transient_one = PointerPath(0x612340, ())
    transient_two = PointerPath(0x712340, (0x20,))

    first = _snapshot(stable, transient_one)
    second = replace(
        _snapshot(stable, transient_two),
        pid=2,
        module_base=0x400000,
        player_base=0x700000,
    )

    ranked = rank_stable_paths((first, second))

    assert ranked[0].path == stable
    assert ranked[0].sessions_present == 2
    assert ranked[0].stable_across_all_sessions
    assert ranked[0].movement_confirmed_sessions == 2


def test_passive_player_proof_selects_historical_direct_alias_without_movement() -> None:
    from position.AnchoredPointerDiscovery import AnchoredPlayerObservation
    from position.AutonomousPointerSelection import (
        prove_player_and_rank_direct_slots,
    )

    memory = FakeMemory()
    memory.u32(memory.module_base + 0x24, memory.player)
    memory.u32(memory.player + 0x10, memory.player)
    memory.u32(memory.player + 0x20, 49603)
    memory.u32(memory.player + 0x24, 49603)
    memory.u32(memory.player + 0x28, 0)
    memory.u32(memory.player + 0x2C, 0)
    player_segment = memory._segments[memory.player]
    struct.pack_into("<fff", player_segment, 0x30, 253.0, 0.0, 86.0)

    observation = AnchoredPlayerObservation(
        player_base=memory.player,
        direct_module_slots=(
            memory.module_base + 0x20,
            memory.module_base + 0x24,
        ),
        self_pointer_offset=0x10,
        hp_offset=0x20,
        max_hp_offset=0x24,
        species_offset=0x28,
        active_species_offset=0x2C,
        x_offset=0x30,
        y_offset=0x34,
        z_offset=0x38,
        x=253.0,
        y=0.0,
        z=86.0,
        current_hp=49603,
        maximum_hp=49603,
    )
    module = ModuleInfo(
        "Neuz.exe",
        "C:/Games/Neuz.exe",
        memory.module_base,
        memory.module_size,
    )

    proof = prove_player_and_rank_direct_slots(
        memory,
        observation,
        module,
        _regions(memory),
        configured_player_offset=0x20,
        historical_sessions_by_offset={0x24: 3},
        samples=3,
        interval_seconds=0.0,
    )

    assert proof.accepted
    assert proof.coordinate_change_native == 0.0
    assert proof.selected_slot is not None
    assert proof.selected_slot.module_offset == 0x24
    assert proof.selected_slot.historical_sessions == 3


def test_passive_player_proof_rejects_monster_like_object() -> None:
    from position.AnchoredPointerDiscovery import AnchoredPlayerObservation
    from position.AutonomousPointerSelection import (
        prove_player_and_rank_direct_slots,
    )

    memory = FakeMemory()
    memory.u32(memory.player + 0x10, memory.player)
    memory.u32(memory.player + 0x20, 400236)
    memory.u32(memory.player + 0x24, 400236)
    memory.u32(memory.player + 0x28, 944)
    memory.u32(memory.player + 0x2C, 944)
    player_segment = memory._segments[memory.player]
    struct.pack_into("<fff", player_segment, 0x30, 253.0, 0.0, 86.0)
    observation = AnchoredPlayerObservation(
        player_base=memory.player,
        direct_module_slots=(memory.module_base + 0x20,),
        self_pointer_offset=0x10,
        hp_offset=0x20,
        max_hp_offset=0x24,
        species_offset=0x28,
        active_species_offset=0x2C,
        x_offset=0x30,
        y_offset=0x34,
        z_offset=0x38,
        x=253.0,
        y=0.0,
        z=86.0,
        current_hp=400236,
        maximum_hp=400236,
    )
    module = ModuleInfo(
        "Neuz.exe",
        "C:/Games/Neuz.exe",
        memory.module_base,
        memory.module_size,
    )

    proof = prove_player_and_rank_direct_slots(
        memory,
        observation,
        module,
        _regions(memory),
        configured_player_offset=0x20,
        samples=3,
        interval_seconds=0.0,
    )

    assert not proof.accepted
    assert proof.structural_samples == 0
