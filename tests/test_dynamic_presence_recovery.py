from __future__ import annotations

import struct
from types import SimpleNamespace

from position.AuthoritativeActorDiscovery import discover_authoritative_actors
from position.IndependentNativeReader import IndependentNativeReader
from position.profiling.presence_promotion import promote_validated_presence_offset


STRIDE = 0x2008
RELATION_OFFSET = 0x2A4
RELATION_VALUE = 0x61000000
SPECIES_OFFSET = 0x174
HP_OFFSET = 0x81C
X_OFFSET = 0x160
Y_OFFSET = 0x164
Z_OFFSET = 0x168
SELF_OFFSET = 0x3C8
MOVED_PRESENCE_OFFSET = 0x19A4
HISTORICAL_PRESENCE_OFFSET = 0x1DCC


class SparseMemory:
    def __init__(self) -> None:
        self.data: dict[int, bytes] = {}
        self.last_search_diagnostics = SimpleNamespace(bytes_read=0, regions_read=0)

    def write_u32(self, address: int, value: int) -> None:
        self.data[int(address)] = struct.pack("<I", int(value))

    def write_i32(self, address: int, value: int) -> None:
        self.data[int(address)] = struct.pack("<i", int(value))

    def write_f32(self, address: int, value: float) -> None:
        self.data[int(address)] = struct.pack("<f", float(value))

    def read(self, address: int, size: int) -> bytes:
        if size == 4 and int(address) in self.data:
            return self.data[int(address)]
        raise OSError(f"unmapped 0x{int(address):X}+0x{int(size):X}")

    def find_u32(self, value: int, **_kwargs) -> tuple[int, ...]:
        needle = struct.pack("<I", int(value))
        hits = tuple(sorted(address for address, data in self.data.items() if data == needle))
        self.last_search_diagnostics = SimpleNamespace(
            bytes_read=len(self.data) * 4,
            regions_read=1,
        )
        return hits


def write_actor(
    memory: SparseMemory,
    base: int,
    *,
    species: int,
    hp: int,
    presence: int,
) -> None:
    memory.write_u32(base + SELF_OFFSET, base)
    memory.write_u32(base + RELATION_OFFSET, RELATION_VALUE)
    memory.write_i32(base + SPECIES_OFFSET, species)
    memory.write_i32(base + HP_OFFSET, hp)
    memory.write_i32(base + MOVED_PRESENCE_OFFSET, presence)
    memory.write_f32(base + X_OFFSET, 250.0 if species else 0.0)
    memory.write_f32(base + Y_OFFSET, 100.0 if species else 0.0)
    memory.write_f32(base + Z_OFFSET, 86.0 if species else 0.0)


def test_recorder_discovers_and_consumes_moved_presence_field() -> None:
    memory = SparseMemory()
    memory.write_u32(RELATION_VALUE, 0x12345678)
    player = 0x50000000
    memory.write_u32(player + SELF_OFFSET, player)
    memory.write_u32(player + RELATION_OFFSET, RELATION_VALUE)
    selected_bases = (0x20000000, 0x21000000, 0x22000000, 0x23000000)
    for index, base in enumerate(selected_bases):
        species = 944 if index < 2 else 948
        write_actor(
            memory,
            base,
            species=species,
            hp=0 if index == 3 else 400236,
            presence=species,
        )
    for base in (0x24000000, 0x25000000):
        write_actor(memory, base, species=0, hp=0, presence=0)
    # A historical-offset coincidence on one actor is not layout evidence.
    memory.write_i32(selected_bases[0] + HISTORICAL_PRESENCE_OFFSET, 944)

    monsters = tuple(
        SimpleNamespace(
            base=base,
            species=944 if index < 2 else 948,
            hp=400236,
            x=250.0,
            y=100.0,
            z=86.0,
            self_pointer_offsets=(SELF_OFFSET,),
            species_offset=SPECIES_OFFSET,
            active_species_offset=0x1DBC,
            hp_offset=HP_OFFSET,
            x_offset=X_OFFSET,
            y_offset=Y_OFFSET,
            z_offset=Z_OFFSET,
        )
        for index, base in enumerate(selected_bases[:3])
    )
    discovery = SimpleNamespace(
        player=SimpleNamespace(base=player),
        monsters=monsters,
    )
    result = discover_authoritative_actors(
        memory,
        discovery,
        selected_species_ids={944, 948},
        actor_stride=STRIDE,
        object_span=STRIDE,
        maximum_address=0x7FFFFFFF,
        private_memory_only=True,
        chunk_size=4096,
        coordinate_limit=100000.0,
    )
    assert result.succeeded
    assert result.presence_species_offset == MOVED_PRESENCE_OFFSET
    assert result.presence_species_validated is True
    assert all(item.offset != HISTORICAL_PRESENCE_OFFSET for item in result.presence_candidates)

    reader = object.__new__(IndependentNativeReader)
    reader.actor_stride = STRIDE
    reader._recovered_presence_species_offset = result.presence_species_offset
    reader._presence_species_validated = result.presence_species_validated
    reader._presence_species_offset = None
    reader._presence_sampling_requested = False
    assert reader.enable_presence_optimized_sampling(
        selected_species_ids={944, 948}
    ) is True
    assert reader._presence_species_offset == MOVED_PRESENCE_OFFSET


def test_session_profiler_can_promote_presence_without_reappearance_gate() -> None:
    from threading import RLock

    memory = SparseMemory()
    bases = tuple(0x30000000 + index * 0x10000 for index in range(20))
    for index, base in enumerate(bases):
        species = 944 if index < 10 else 948
        memory.write_i32(base + SPECIES_OFFSET, species)
        memory.write_i32(base + MOVED_PRESENCE_OFFSET, species)

    reader = object.__new__(IndependentNativeReader)
    reader._memory = memory
    reader.monster_targets = (
        SimpleNamespace(
            species_offset=SPECIES_OFFSET,
            x_offset=X_OFFSET,
            y_offset=Y_OFFSET,
            z_offset=Z_OFFSET,
            self_pointer_offsets=(SELF_OFFSET,),
        ),
    )
    reader.monster_hp_offset = HP_OFFSET
    reader.actor_stride = STRIDE
    reader._authoritative_relation_offset = RELATION_OFFSET
    reader._authoritative_presence_candidates = ()
    reader._recovered_presence_species_offset = None
    reader._presence_species_validated = False
    reader._presence_validation_source = "none"
    reader._presence_sampling_requested = True
    reader._presence_species_offset = None
    reader._presence_selected_species = {944, 948}
    reader._selected_species_ids = {944, 948}
    reader._cache_lock = RLock()
    reader._actor_slots = bases

    evidence = {
        "offset": MOVED_PRESENCE_OFFSET,
        "live_samples": 40,
        "live_matches": 38,
        "zero_hp_samples": 12,
        "zero_hp_matches": 12,
        "dormant_samples": 20,
        "dormant_matches": 0,
        "inactive_samples": 8,
        "inactive_matches": 0,
        "reappearance_samples": 11,
        "reappearance_matches": 5,
        "unique_bases": 20,
        "species": [944, 948],
    }
    assert promote_validated_presence_offset(
        reader,
        memory,
        MOVED_PRESENCE_OFFSET,
        evidence=evidence,
        selected_species_ids={944, 948},
    ) is True
    assert reader._presence_species_offset == MOVED_PRESENCE_OFFSET
    assert reader._presence_species_validated is True
    assert reader._presence_validation_source == "session_longitudinal_profiler"
