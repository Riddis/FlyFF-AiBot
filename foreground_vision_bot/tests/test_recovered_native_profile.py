from __future__ import annotations

import struct
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

from position.RecoveredNativeProfile import (
    PROFILE_VERSION,
    RecoveredNativeProfile,
    load_profile,
    restore_profile,
    save_profile,
)
from position.Win32ProcessMemory import ModuleInfo


class Memory:
    def __init__(self) -> None:
        self.data: dict[int, bytes] = {}
        self.last_search_diagnostics = SimpleNamespace(bytes_read=0, regions_read=0)

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
        del private_only, chunk_size, cancellation, deadline
        needle = struct.pack("<I", value)
        hits = tuple(
            sorted(
                address
                for address, data in self.data.items()
                if address <= maximum_address and data == needle
            )
        )
        self.last_search_diagnostics = SimpleNamespace(
            bytes_read=len(self.data) * 4,
            regions_read=1,
        )
        return hits


MODULE_BASE = 0x100000
PLAYER_SLOT_OFFSET = 0x2000
PLAYER_SELF = 0x3C8
MONSTER_SELF = 0x3C8
RELATION_OFFSET = 0x16C
SPECIES_OFFSET = 0x174
HP_OFFSET = 0x81C
X_OFFSET = 0x160
Y_OFFSET = 0x164
Z_OFFSET = 0x168
ACTIVE_OFFSET = 0x1DBC
RELATION_VALUE = 0x71000000


def profile() -> RecoveredNativeProfile:
    return RecoveredNativeProfile(
        version=PROFILE_VERSION,
        module_name="Neuz.exe",
        module_size=0x943000,
        module_filename="Neuz.exe",
        module_sha256="",
        player_slot_offsets=(PLAYER_SLOT_OFFSET,),
        player_self_offsets=(PLAYER_SELF,),
        monster_self_offsets=(MONSTER_SELF,),
        player_hp_offset=HP_OFFSET,
        species_offset=SPECIES_OFFSET,
        active_species_offset=ACTIVE_OFFSET,
        monster_hp_offset=HP_OFFSET,
        x_offset=X_OFFSET,
        y_offset=Y_OFFSET,
        z_offset=Z_OFFSET,
        actor_stride=0x2008,
        authoritative_relation_offset=RELATION_OFFSET,
        anchor_species=944,
        anchor_hp=400236,
        expected_full_hp_by_species=((944, 400236),),
        saved_at_utc="2026-08-02T00:00:00+00:00",
    )


def write_actor(memory: Memory, base: int, species: int, hp: int, x: float) -> None:
    memory.write_u32(base + MONSTER_SELF, base)
    memory.write_u32(base + RELATION_OFFSET, RELATION_VALUE)
    memory.write_i32(base + SPECIES_OFFSET, species)
    memory.write_i32(base + HP_OFFSET, hp)
    memory.write_f32(base + X_OFFSET, x)
    memory.write_f32(base + Y_OFFSET, 100.0)
    memory.write_f32(base + Z_OFFSET, 86.0)


def test_profile_round_trip(tmp_path) -> None:
    path = tmp_path / "native_profile.json"
    saved = save_profile(profile(), path)
    assert saved == path
    assert load_profile(path) == profile()


def test_restore_resolves_fresh_addresses_and_global_actor_relation() -> None:
    memory = Memory()
    module = ModuleInfo("Neuz.exe", r"F:\\Games\\Neuz.exe", MODULE_BASE, 0x943000)
    player = 0x44000000
    memory.write_u32(MODULE_BASE + PLAYER_SLOT_OFFSET, player)
    memory.write_u32(player + PLAYER_SELF, player)
    memory.write_u32(player + RELATION_OFFSET, RELATION_VALUE)
    memory.write_i32(player + HP_OFFSET, 26930)
    memory.write_f32(player + X_OFFSET, 253.0)
    memory.write_f32(player + Y_OFFSET, 100.0)
    memory.write_f32(player + Z_OFFSET, 86.0)
    memory.write_u32(RELATION_VALUE, 0x12345678)

    write_actor(memory, 0x50000000, 944, 400236, 250.0)
    write_actor(memory, 0x60000000, 948, 250000, 260.0)

    restored = restore_profile(
        memory,
        module,
        profile(),
        selected_species_ids=(944, 948),
        maximum_address=0x7FFFFFFF,
        private_memory_only=True,
        chunk_size=4096,
        coordinate_limit=100000.0,
    )

    assert restored.discovery.player is not None
    assert restored.discovery.player.base == player
    assert restored.relation_offset == RELATION_OFFSET
    assert restored.relation_value == RELATION_VALUE
    assert set(restored.authoritative.actor_bases) == {0x50000000, 0x60000000}
    assert dict(restored.authoritative.species_counts) == {944: 1, 948: 1}


def test_restore_rejects_another_client_build() -> None:
    memory = Memory()
    module = ModuleInfo("Neuz.exe", "Neuz.exe", MODULE_BASE, 0x123000)
    try:
        restore_profile(
            memory,
            module,
            profile(),
            selected_species_ids=(944,),
            maximum_address=0x7FFFFFFF,
            private_memory_only=True,
            chunk_size=4096,
            coordinate_limit=100000.0,
        )
    except ValueError as error:
        assert "client build" in str(error)
    else:
        raise AssertionError("mismatched module size was accepted")


def test_restore_rejects_changed_client_executable(tmp_path) -> None:
    executable = tmp_path / "Neuz.exe"
    executable.write_bytes(b"KNOWN-CLIENT-BUILD")
    expected_hash = sha256(executable.read_bytes()).hexdigest().upper()
    cached = replace(profile(), module_sha256=expected_hash)
    executable.write_bytes(b"CHANGED-CLIENT-BUILD")
    module = ModuleInfo(
        "Neuz.exe",
        str(executable),
        MODULE_BASE,
        cached.module_size,
    )

    try:
        restore_profile(
            Memory(),
            module,
            cached,
            selected_species_ids=(944,),
            maximum_address=0x7FFFFFFF,
            private_memory_only=True,
            chunk_size=4096,
            coordinate_limit=100000.0,
        )
    except ValueError as error:
        assert "client executable" in str(error)
    else:
        raise AssertionError("changed client executable was accepted")
