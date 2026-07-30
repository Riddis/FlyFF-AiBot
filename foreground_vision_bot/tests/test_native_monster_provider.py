from __future__ import annotations

import struct

import pytest

from position.MonsterConfig import NativeMonsterConfig
from position.NativeFlyffMonsterProvider import (
    NativeFlyffMonsterProvider,
    NativeMonsterReadError,
)
from position.Win32ProcessMemory import MemorySearchDiagnostics


class FakeMemory:
    def __init__(self, module_base: int = 0x400000) -> None:
        self._module_base = module_base
        self.bytes: dict[int, int] = {}
        self.closed = False
        self.reads: list[tuple[int, int]] = []
        self.last_search_diagnostics = MemorySearchDiagnostics()

    def module_base(self, module_name: str) -> int:
        assert module_name == "Neuz.exe"
        return self._module_base

    def write(self, address: int, data: bytes) -> None:
        for index, value in enumerate(data):
            self.bytes[address + index] = value

    def u32(self, address: int, value: int) -> None:
        self.write(address, struct.pack("<I", value))

    def i32(self, address: int, value: int) -> None:
        self.write(address, struct.pack("<i", value))

    def f32(self, address: int, value: float) -> None:
        self.write(address, struct.pack("<f", value))

    def read(self, address: int, size: int) -> bytes:
        self.reads.append((address, size))
        try:
            return bytes(self.bytes[address + index] for index in range(size))
        except KeyError as error:
            raise RuntimeError(f"unreadable 0x{address:X}") from error

    def find_u32(
        self,
        value: int,
        *,
        maximum_address: int,
        private_only: bool,
        chunk_size: int,
    ) -> tuple[int, ...]:
        del private_only, chunk_size
        needle = struct.pack("<I", value)
        starts = sorted(
            address
            for address in self.bytes
            if address <= maximum_address
            and all(self.bytes.get(address + index) == needle[index] for index in range(4))
        )
        self.last_search_diagnostics = MemorySearchDiagnostics(
            regions_considered=3,
            regions_read=3,
            bytes_read=0xC0000,
            matches=len(starts),
        )
        return tuple(starts)

    def close(self) -> None:
        self.closed = True


def _config() -> NativeMonsterConfig:
    return NativeMonsterConfig(
        player_pointer_offset=0x100,
        world_pointer_offset=0x104,
        discovery_interval_seconds=10,
        vision_radius_native=160,
        discovery_chunk_bytes=4096,
    )


def _write_actor(
    memory: FakeMemory,
    config: NativeMonsterConfig,
    base: int,
    *,
    world: int,
    species: int,
    hp: int,
    x: float,
    z: float,
    active_species: int | None = None,
) -> None:
    memory.u32(base + config.self_pointer_offset, base)
    memory.u32(base + config.world_offset, world)
    memory.i32(base + config.species_offset, species)
    memory.i32(
        base + config.active_species_offset,
        species if active_species is None else active_species,
    )
    memory.i32(base + config.hp_offset, hp)
    memory.f32(base + config.x_offset, x)
    memory.f32(base + config.y_offset, 10.0)
    memory.f32(base + config.z_offset, z)


def _provider_fixture() -> tuple[NativeFlyffMonsterProvider, FakeMemory, dict[str, int]]:
    config = _config()
    memory = FakeMemory()
    player = 0x10000000
    world = 0x20000000
    addresses = {
        "player": player,
        # Deliberately unrelated allocations: discovery must not depend on the
        # player address or actor stride.
        "wrong_world": 0x26002000,
        "registered_near": 0x2EEF5C80,
        "dormant": 0x26C4E1B0,
        "unregistered": 0x37004000,
        "dead": 0x2EF03CB8,
        "registered_far": 0x3A005000,
        "world": world,
    }
    memory.u32(memory._module_base + config.player_pointer_offset, player)
    memory.u32(memory._module_base + config.world_pointer_offset, world)

    _write_actor(memory, config, player, world=world, species=11, hp=30000, x=0, z=0)
    _write_actor(
        memory,
        config,
        addresses["wrong_world"],
        world=0x22222222,
        species=874,
        hp=140000,
        x=5,
        z=5,
    )
    _write_actor(
        memory,
        config,
        addresses["registered_near"],
        world=world,
        species=874,
        hp=140000,
        x=30,
        z=40,
    )
    _write_actor(
        memory,
        config,
        addresses["dormant"],
        world=world,
        species=874,
        hp=140000,
        x=20,
        z=20,
        active_species=0,
    )
    _write_actor(
        memory,
        config,
        addresses["unregistered"],
        world=world,
        species=1848,
        hp=2297,
        x=10,
        z=10,
    )
    _write_actor(
        memory,
        config,
        addresses["dead"],
        world=world,
        species=874,
        hp=0,
        x=15,
        z=15,
    )
    _write_actor(
        memory,
        config,
        addresses["registered_far"],
        world=world,
        species=715,
        hp=120000,
        x=300,
        z=0,
    )

    # A false positive world pointer that does not have a self-valid actor base.
    memory.u32(0x3F000000, world)

    provider = NativeFlyffMonsterProvider(
        memory,
        config,
        clock=lambda: 1.0,
    )
    return provider, memory, addresses


def test_provider_discovers_unrelated_actor_slabs_globally() -> None:
    provider, _memory, addresses = _provider_fixture()

    slots = provider.discover_slots(force=True)

    assert addresses["player"] in slots
    assert addresses["registered_near"] in slots
    assert addresses["dormant"] in slots
    assert addresses["unregistered"] in slots
    assert addresses["dead"] in slots
    assert addresses["registered_far"] in slots
    assert addresses["wrong_world"] not in slots
    assert provider.last_diagnostics is not None
    assert provider.last_diagnostics.discovery_regions_read == 3
    assert provider.last_diagnostics.discovery_bytes_read == 0xC0000
    assert provider.last_diagnostics.rejected_invalid_actor >= 1


def test_provider_filters_world_active_species_hp_species_and_radius() -> None:
    provider, _memory, addresses = _provider_fixture()

    actors = provider.read_active_actors(allowed_species_ids={874, 715})

    assert [actor.base_address for actor in actors] == [addresses["registered_near"]]
    actor = actors[0]
    assert actor.species_id == 874
    assert actor.active_species_id == 874
    assert actor.hp == 140000
    assert actor.distance_native == pytest.approx(50.0)
    assert provider.last_diagnostics is not None
    assert provider.last_diagnostics.rejected_not_present == 1
    assert provider.last_diagnostics.rejected_dead == 1
    assert provider.last_diagnostics.rejected_species == 1
    assert provider.last_diagnostics.rejected_distance == 1


def test_provider_can_return_all_active_actor_species_for_diagnostics() -> None:
    provider, _memory, addresses = _provider_fixture()

    actors = provider.read_active_actors(allowed_species_ids=None)

    assert [actor.base_address for actor in actors] == [
        addresses["unregistered"],
        addresses["registered_near"],
    ]


def test_provider_captures_selected_actor_species() -> None:
    provider, memory, addresses = _provider_fixture()
    memory.u32(
        addresses["world"] + provider.config.selected_actor_offset,
        addresses["registered_near"],
    )

    actor = provider.capture_selected_actor()

    assert actor.base_address == addresses["registered_near"]
    assert actor.species_id == 874
    assert actor.active_species_id == 874
    assert actor.hp == 140000


def test_provider_rejects_dormant_selected_actor() -> None:
    provider, memory, addresses = _provider_fixture()
    memory.u32(
        addresses["world"] + provider.config.selected_actor_offset,
        addresses["dormant"],
    )

    with pytest.raises(NativeMonsterReadError, match="not currently instantiated"):
        provider.capture_selected_actor()


def test_provider_rejects_empty_selected_actor() -> None:
    provider, memory, addresses = _provider_fixture()
    memory.u32(addresses["world"] + provider.config.selected_actor_offset, 0)

    with pytest.raises(NativeMonsterReadError, match="No actor"):
        provider.capture_selected_actor()


def test_provider_closes_memory() -> None:
    provider, memory, _addresses = _provider_fixture()

    provider.close()

    assert memory.closed


def test_provider_can_be_created_from_process_id(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config()
    memory = FakeMemory()
    captured: dict[str, int] = {}

    class FakeProcessMemory:
        def __new__(cls, process_id: int, *, backend=None):
            captured["process_id"] = process_id
            return memory

    import importlib

    provider_module = importlib.import_module(
        "position.NativeFlyffMonsterProvider"
    )
    monkeypatch.setattr(
        provider_module,
        "Win32ProcessMemory",
        FakeProcessMemory,
    )

    provider = NativeFlyffMonsterProvider.from_process_id(26304, config)

    assert captured == {"process_id": 26304}
    assert provider.module_base == memory._module_base
