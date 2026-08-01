from __future__ import annotations

import json
import struct

from position.AggregateMonsterRootScan import scan_aggregate_monster_roots
from position.AnchoredPointerDiscovery import AnchoredMonsterObservation
from position.PointerScanWorkflow import ReadableRegionIndex
from position.Win32ProcessMemory import MemoryRegion, ModuleInfo


class AggregateFakeMemory:
    def __init__(self) -> None:
        self.pid = 88
        self.module = 0x100000
        self.world = 0x200000
        self.manager = 0x210000
        self.actors = (0x300000, 0x301000, 0x302000, 0x303000)
        self._segments: dict[int, bytearray] = {
            self.module: bytearray(0x100),
            self.world: bytearray(0x100),
            self.manager: bytearray(0x100),
            **{actor: bytearray(0x100) for actor in self.actors},
        }
        self.u32(self.module + 0x20, self.world)
        self.u32(self.module + 0x24, self.manager)
        for index, actor in enumerate(self.actors):
            self.u32(actor + 0x10, actor)
            self.i32(actor + 0x20, 944)
            self.i32(actor + 0x24, 944)
            self.i32(actor + 0x28, 400236)
            self.f32(actor + 0x30, 250.0 + index)
            self.f32(actor + 0x34, 0.0)
            self.f32(actor + 0x38, 85.0 + index)
            self.u32(actor + 0x40, self.world)
            self.u32(self.manager + index * 4, actor)

    def _segment(self, address: int, size: int) -> tuple[int, bytearray]:
        for base, data in self._segments.items():
            if base <= address and address + size <= base + len(data):
                return base, data
        raise OSError(f"unreadable 0x{address:X}+0x{size:X}")

    def read(self, address: int, size: int) -> bytes:
        base, data = self._segment(address, size)
        start = address - base
        return bytes(data[start : start + size])

    def u32(self, address: int, value: int) -> None:
        base, data = self._segment(address, 4)
        struct.pack_into("<I", data, address - base, value)

    def i32(self, address: int, value: int) -> None:
        base, data = self._segment(address, 4)
        struct.pack_into("<i", data, address - base, value)

    def f32(self, address: int, value: float) -> None:
        base, data = self._segment(address, 4)
        struct.pack_into("<f", data, address - base, value)


def _regions(memory: AggregateFakeMemory) -> tuple[MemoryRegion, ...]:
    return tuple(
        MemoryRegion(base, len(data), 4, 0x20000)
        for base, data in sorted(memory._segments.items())
    )


def _cohort(memory: AggregateFakeMemory) -> tuple[AnchoredMonsterObservation, ...]:
    return tuple(
        AnchoredMonsterObservation(
            base=actor,
            species=944,
            current_hp=400236,
            x=250.0 + index,
            y=0.0,
            z=85.0 + index,
            self_pointer_offsets=(0x10,),
            species_offset=0x20,
            active_species_offset=0x24,
            hp_offset=0x28,
            x_offset=0x30,
            y_offset=0x34,
            z_offset=0x38,
        )
        for index, actor in enumerate(memory.actors)
    )


def test_aggregate_scan_ranks_shared_world_and_holder_without_tracking_one_mob() -> None:
    memory = AggregateFakeMemory()
    regions = _regions(memory)
    module = ModuleInfo("Neuz.exe", "C:/Games/Neuz.exe", memory.module, 0x100)
    readable = ReadableRegionIndex.build(regions)
    module_refs = {
        memory.world: (memory.module + 0x20,),
        memory.manager: (memory.module + 0x24,),
    }

    def mutate(index: int) -> None:
        if index != 1:
            return
        memory.i32(memory.actors[0] + 0x28, 350000)
        memory.i32(memory.actors[1] + 0x24, 0)
        memory.i32(memory.actors[1] + 0x28, 0)

    report = scan_aggregate_monster_roots(
        memory,
        _cohort(memory),
        module,
        regions,
        readable,
        module_refs,
        duration_seconds=0.001,
        interval_seconds=0.001,
        object_span=0x100,
        holder_span=0x100,
        minimum_support=2,
        maximum_scan_bytes=sum(region.size for region in regions),
        maximum_depth=1,
        path_field_span=0x100,
        maximum_roots=20,
        maximum_nodes=20,
        path_candidate_limit=8,
        before_sample=mutate,
    )

    assert report.sample_count == 2
    assert report.changed_actor_count == 2
    assert report.transition_events == 2

    world = next(
        item
        for item in report.candidates
        if item.kind == "actor_field_target"
        and item.target_base == memory.world
        and item.actor_field_offset == 0x40
    )
    holder = next(
        item
        for item in report.candidates
        if item.kind == "module_rooted_holder"
        and item.target_base == memory.manager
    )

    assert world.recommended
    assert holder.recommended
    assert world.changed_actor_support == 2
    assert holder.changed_actor_support == 2
    assert world.pointer_paths[0].root_module_offset == 0x20
    assert holder.pointer_paths[0].root_module_offset == 0x24
    assert world.average_active_coverage == 1.0
    assert holder.average_active_coverage == 1.0

    payload = json.loads(report.to_json())
    assert payload["changed_actor_count"] == 2
    assert payload["candidates"][0]["pointer_paths"]
