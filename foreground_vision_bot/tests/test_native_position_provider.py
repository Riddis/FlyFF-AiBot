from __future__ import annotations

import math
import struct

import pytest
from position.NativeFlyffPositionProvider import (
    InvalidPlayerPoseError,
    NativeFlyffPositionProvider,
)
from position.PositionConfig import NativePositionConfig


class FakeMemory:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.reads: list[tuple[int, int]] = []
        self.closed = False

    def read(self, address: int, size: int) -> bytes:
        self.reads.append((address, size))
        return self.data[:size]

    def close(self) -> None:
        self.closed = True


def test_provider_reads_xyz_without_heading_in_one_block() -> None:
    memory = FakeMemory(struct.pack("<fff", 12.5, -3.25, 99.0))
    config = NativePositionConfig(
        enabled=True,
        transform_address=0x5000,
    )
    provider = NativeFlyffPositionProvider(
        memory,  # type: ignore[arg-type]
        config,
        clock=lambda: 42.5,
    )

    pose = provider.read_pose()

    assert pose.x == 12.5
    assert pose.y == -3.25
    assert pose.z == 99.0
    assert pose.heading_degrees is None
    assert pose.timestamp == 42.5
    assert memory.reads == [(0x5000, 12)]


def test_provider_converts_radian_heading_and_normalises_degrees() -> None:
    memory = FakeMemory(
        struct.pack("<ffff", 1.0, 2.0, 3.0, -math.pi / 2.0)
    )
    config = NativePositionConfig(
        enabled=True,
        transform_address=0x7000,
        heading_offset=12,
        heading_unit="radians",
    )
    provider = NativeFlyffPositionProvider(
        memory,  # type: ignore[arg-type]
        config,
    )

    pose = provider.read_pose()

    assert pose.heading_degrees == pytest.approx(270.0)


def test_provider_supports_nonzero_and_out_of_order_offsets() -> None:
    data = bytearray(24)
    struct.pack_into("<f", data, 12, 8.0)
    struct.pack_into("<f", data, 4, 6.0)
    struct.pack_into("<f", data, 20, 10.0)
    memory = FakeMemory(bytes(data[4:24]))
    config = NativePositionConfig(
        enabled=True,
        transform_address=0x9000,
        x_offset=12,
        y_offset=4,
        z_offset=20,
    )
    provider = NativeFlyffPositionProvider(
        memory,  # type: ignore[arg-type]
        config,
    )

    pose = provider.read_pose()

    assert (pose.x, pose.y, pose.z) == (8.0, 6.0, 10.0)
    assert memory.reads == [(0x9004, 20)]


def test_provider_rejects_non_finite_values() -> None:
    memory = FakeMemory(struct.pack("<fff", math.nan, 0.0, 0.0))
    config = NativePositionConfig(
        enabled=True,
        transform_address=0x5000,
    )
    provider = NativeFlyffPositionProvider(
        memory,  # type: ignore[arg-type]
        config,
    )

    with pytest.raises(InvalidPlayerPoseError, match="non-finite"):
        provider.read_pose()


def test_provider_closes_owned_memory() -> None:
    memory = FakeMemory(struct.pack("<fff", 1.0, 2.0, 3.0))
    config = NativePositionConfig(
        enabled=True,
        transform_address=0x5000,
    )
    provider = NativeFlyffPositionProvider(
        memory,  # type: ignore[arg-type]
        config,
    )

    provider.close()

    assert memory.closed


class AddressedMemory:
    def __init__(self, module_base: int, values: dict[int, bytes | Exception]) -> None:
        self._module_base = module_base
        self.values = values
        self.reads: list[tuple[int, int]] = []
        self.closed = False

    def module_base(self, module_name: str) -> int:
        assert module_name == "Neuz.exe"
        return self._module_base

    def read(self, address: int, size: int) -> bytes:
        self.reads.append((address, size))
        value = self.values[address]
        if isinstance(value, Exception):
            raise value
        return value[:size]

    def close(self) -> None:
        self.closed = True


def _module_config(**overrides: object) -> NativePositionConfig:
    payload: dict[str, object] = {
        "enabled": True,
        "resolver": "module_offsets",
        "module_name": "Neuz.exe",
        "transform_offsets": (0x10, 0x20, 0x30),
        "minimum_consensus_sources": 2,
        "consensus_tolerance": 0.05,
    }
    payload.update(overrides)
    return NativePositionConfig(**payload)  # type: ignore[arg-type]


def test_provider_resolves_module_offsets_and_uses_two_of_three_consensus() -> None:
    base = 0x10000000
    memory = AddressedMemory(
        base,
        {
            base + 0x10: struct.pack("<fff", 10.0, 20.0, 30.0),
            base + 0x20: struct.pack("<fff", 10.02, 20.01, 29.99),
            base + 0x30: struct.pack("<fff", 900.0, 800.0, 700.0),
        },
    )
    provider = NativeFlyffPositionProvider(
        memory,  # type: ignore[arg-type]
        _module_config(),
        clock=lambda: 5.0,
    )

    pose = provider.read_pose()

    assert provider.module_base == base
    assert provider.resolved_addresses == (base + 0x10, base + 0x20, base + 0x30)
    assert pose.x == pytest.approx(10.01)
    assert pose.y == pytest.approx(20.005)
    assert pose.z == pytest.approx(29.995)
    assert provider.last_diagnostics is not None
    assert provider.last_diagnostics.consensus_addresses == (
        base + 0x10,
        base + 0x20,
    )


def test_provider_tolerates_one_unreadable_mirror() -> None:
    base = 0x20000000
    memory = AddressedMemory(
        base,
        {
            base + 0x10: struct.pack("<fff", 1.0, 2.0, 3.0),
            base + 0x20: RuntimeError("unreadable"),
            base + 0x30: struct.pack("<fff", 1.0, 2.0, 3.0),
        },
    )
    provider = NativeFlyffPositionProvider(
        memory,  # type: ignore[arg-type]
        _module_config(),
    )

    pose = provider.read_pose()

    assert (pose.x, pose.y, pose.z) == (1.0, 2.0, 3.0)
    assert provider.last_diagnostics is not None
    assert provider.last_diagnostics.failed_addresses == (base + 0x20,)


def test_provider_rejects_three_way_disagreement() -> None:
    from position.NativeFlyffPositionProvider import PoseConsensusError

    base = 0x30000000
    memory = AddressedMemory(
        base,
        {
            base + 0x10: struct.pack("<fff", 1.0, 2.0, 3.0),
            base + 0x20: struct.pack("<fff", 4.0, 5.0, 6.0),
            base + 0x30: struct.pack("<fff", 7.0, 8.0, 9.0),
        },
    )
    provider = NativeFlyffPositionProvider(
        memory,  # type: ignore[arg-type]
        _module_config(),
    )

    with pytest.raises(PoseConsensusError, match="largest cluster"):
        provider.read_pose()


class DynamicPointerMemory:
    def __init__(
        self,
        module_base: int,
        pointer_offset: int,
        target: int,
        poses: dict[int, tuple[float, float, float]],
    ) -> None:
        self._module_base = module_base
        self.pointer_offset = pointer_offset
        self.target = target
        self.poses = poses
        self.reads: list[tuple[int, int]] = []
        self.closed = False

    @property
    def pointer_storage(self) -> int:
        return self._module_base + self.pointer_offset

    def module_base(self, module_name: str) -> int:
        assert module_name == "Neuz.exe"
        return self._module_base

    def read(self, address: int, size: int) -> bytes:
        self.reads.append((address, size))
        if address == self.pointer_storage:
            return struct.pack("<I", self.target)[:size]
        if address == self.target + 0x160:
            return struct.pack("<fff", *self.poses[self.target])[:size]
        raise RuntimeError(f"unexpected read 0x{address:X} size={size}")

    def close(self) -> None:
        self.closed = True


def _pointer_config() -> NativePositionConfig:
    return NativePositionConfig(
        enabled=True,
        resolver="module_pointer",
        module_name="Neuz.exe",
        pointer_offset=0x5852B8,
        x_offset=0x160,
        y_offset=0x164,
        z_offset=0x168,
    )


def test_provider_follows_module_pointer_to_player_transform() -> None:
    base = 0x00400000
    target = 0x35D0C158
    memory = DynamicPointerMemory(
        base,
        0x5852B8,
        target,
        {target: (123.5, 44.25, 3329.0)},
    )
    provider = NativeFlyffPositionProvider(
        memory,  # type: ignore[arg-type]
        _pointer_config(),
        clock=lambda: 17.0,
    )

    pose = provider.read_pose()

    assert provider.module_base == base
    assert provider.pointer_storage_address == base + 0x5852B8
    assert provider.resolved_addresses == (target,)
    assert (pose.x, pose.y, pose.z) == (123.5, 44.25, 3329.0)
    assert pose.timestamp == 17.0
    assert memory.reads == [
        (base + 0x5852B8, 4),
        (target + 0x160, 12),
    ]


def test_provider_rereads_player_pointer_on_every_pose() -> None:
    base = 0x00400000
    first = 0x26000000
    second = 0x37000000
    memory = DynamicPointerMemory(
        base,
        0x5852B8,
        first,
        {
            first: (1.0, 2.0, 3.0),
            second: (10.0, 20.0, 30.0),
        },
    )
    provider = NativeFlyffPositionProvider(
        memory,  # type: ignore[arg-type]
        _pointer_config(),
    )

    first_pose = provider.read_pose()
    memory.target = second
    second_pose = provider.read_pose()

    assert (first_pose.x, first_pose.y, first_pose.z) == (1.0, 2.0, 3.0)
    assert (second_pose.x, second_pose.y, second_pose.z) == (10.0, 20.0, 30.0)
    assert provider.resolved_addresses == (second,)
    assert memory.reads.count((base + 0x5852B8, 4)) == 2


def test_provider_rejects_null_player_pointer_with_diagnostics() -> None:
    from position.NativeFlyffPositionProvider import PointerResolutionError

    base = 0x00400000
    memory = DynamicPointerMemory(base, 0x5852B8, 0, {})
    provider = NativeFlyffPositionProvider(
        memory,  # type: ignore[arg-type]
        _pointer_config(),
    )

    with pytest.raises(PointerResolutionError, match="null"):
        provider.read_pose()

    assert provider.last_diagnostics is not None
    assert provider.last_diagnostics.failed_addresses == (
        base + 0x5852B8,
    )
