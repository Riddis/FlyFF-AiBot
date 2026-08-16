from __future__ import annotations

import struct
from types import SimpleNamespace

from position.NativePointerRecovery import _verify_cached


class _Memory:
    def __init__(self, values: dict[int, bytes]) -> None:
        self._values = dict(values)

    def read(self, address: int, size: int) -> bytes:
        value = self._values[int(address)]
        if len(value) != int(size):
            raise AssertionError((address, size, len(value)))
        return value


def _u32(value: int) -> bytes:
    return struct.pack("<I", int(value))


def _i32(value: int) -> bytes:
    return struct.pack("<i", int(value))


def _f32(value: float) -> bytes:
    return struct.pack("<f", float(value))


def _cached(*, target_base: int, pointer_slot: int) -> SimpleNamespace:
    player = SimpleNamespace(
        base=int(target_base),
        self_pointer_offsets=(0x1EF0,),
        hp_offset=0x81C,
        x_offset=0x160,
        y_offset=0x164,
        z_offset=0x168,
    )
    discovery = SimpleNamespace(player=player)
    return SimpleNamespace(
        independent_discovery=discovery,
        player_pointer_address=int(pointer_slot),
    )


def _memory(*, pointer_slot: int, current_base: int) -> _Memory:
    return _Memory(
        {
            pointer_slot: _u32(current_base),
            current_base + 0x1EF0: _u32(current_base),
            current_base + 0x81C: _i32(28_087),
            current_base + 0x160: _f32(253.0),
            current_base + 0x164: _f32(86.0),
            current_base + 0x168: _f32(0.0),
        }
    )


def test_independent_cache_rejects_valid_new_player_at_stale_cached_base() -> None:
    pointer_slot = 0xB86318
    previous_base = 0x237C1000
    current_base = 0x237C8FD8

    assert not _verify_cached(
        _memory(pointer_slot=pointer_slot, current_base=current_base),
        _cached(target_base=previous_base, pointer_slot=pointer_slot),
        object(),
        0x610000,
    )


def test_independent_cache_accepts_same_validated_player_identity() -> None:
    pointer_slot = 0xB86318
    current_base = 0x237C8FD8

    assert _verify_cached(
        _memory(pointer_slot=pointer_slot, current_base=current_base),
        _cached(target_base=current_base, pointer_slot=pointer_slot),
        object(),
        0x610000,
    )
