from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from threading import Event, Lock, Thread

import position.NativePointerRecovery as recovery_module
import pytest
from position.MonsterConfig import NativeMonsterConfig
from position.NativeFlyffMonsterProvider import (
    NativeFlyffMonsterProvider,
    NativeMonsterReadError,
)
from position.NativeFlyffPositionProvider import (
    InvalidPlayerPoseError,
    NativeFlyffPositionProvider,
    PointerResolutionError,
)
from position.NativePointerRecovery import (
    PointerRecoveryProgress,
    PointerRecoveryState,
    get_last_pointer_recovery_metrics,
    recover_local_player_pointer,
)
from position.PositionConfig import NativePositionConfig
from position.Win32ProcessMemory import MemorySearchDiagnostics


@dataclass(frozen=True)
class Region:
    base_address: int
    size: int
    protection: int = 0x04
    region_type: int = 0x20000


class RecoveryMemory:
    def __init__(self, *, pid: int) -> None:
        self.pid = pid
        self.module_base_value = 0x10000000
        self.actor_base = 0x30000000
        self.module = bytearray(0x6000)
        self.actor = bytearray(0x3000)
        self.readable_calls = 0
        self.read_calls: list[tuple[int, int]] = []
        self.enumeration_entered = Event()
        self.enumeration_release = Event()
        self.block_enumeration = False
        self.cancel_on_scan_read: Event | None = None
        self._lock = Lock()

    def readable_regions(self, *, maximum_address=0x7FFFFFFF, private_only=True):
        del private_only
        with self._lock:
            self.readable_calls += 1
        self.enumeration_entered.set()
        if self.block_enumeration:
            assert self.enumeration_release.wait(2.0)
        regions = (
            Region(self.module_base_value, len(self.module)),
            Region(self.actor_base, len(self.actor)),
        )
        return tuple(
            region
            for region in regions
            if region.base_address < maximum_address
        )

    def read(self, address: int, size: int) -> bytes:
        with self._lock:
            self.read_calls.append((address, size))
        if size > 4 and self.cancel_on_scan_read is not None:
            self.cancel_on_scan_read.set()
        if self.module_base_value <= address:
            relative = address - self.module_base_value
            if 0 <= relative and relative + size <= len(self.module):
                return bytes(self.module[relative : relative + size])
        if self.actor_base <= address:
            relative = address - self.actor_base
            if 0 <= relative and relative + size <= len(self.actor):
                return bytes(self.actor[relative : relative + size])
        raise RuntimeError(f"unreadable 0x{address:X}+0x{size:X}")

    def module_u32(self, offset: int, value: int) -> None:
        struct.pack_into("<I", self.module, offset, value)

    def actor_u32(self, offset: int, value: int) -> None:
        struct.pack_into("<I", self.actor, offset, value)

    def actor_i32(self, offset: int, value: int) -> None:
        struct.pack_into("<i", self.actor, offset, value)

    def actor_f32(self, offset: int, value: float) -> None:
        struct.pack_into("<f", self.actor, offset, value)


class CheapProviderMemory:
    def __init__(
        self,
        *,
        player_pointer: int,
        world_pointer: int = 0,
        pose: tuple[float, float, float] = (1.0, 2.0, 3.0),
        self_pointer_valid: bool = True,
        player_world: int | None = None,
    ) -> None:
        self.pid = 88001
        self.module_base_value = 0x00400000
        self.player_pointer = player_pointer
        self.world_pointer = world_pointer
        self.pose = pose
        self.self_pointer_valid = self_pointer_valid
        self.player_world = (
            world_pointer if player_world is None else int(player_world)
        )
        self.readable_calls = 0
        self.find_calls = 0
        self.reads: list[tuple[int, int]] = []
        self.closed = False
        self.last_search_diagnostics = MemorySearchDiagnostics()

    def module_base(self, module_name: str) -> int:
        assert module_name == "Neuz.exe"
        return self.module_base_value

    def read(self, address: int, size: int) -> bytes:
        self.reads.append((address, size))
        if address == self.module_base_value + 0x100:
            return struct.pack("<I", self.player_pointer)[:size]
        if address == self.module_base_value + 0x104:
            return struct.pack("<I", self.world_pointer)[:size]
        if address == self.player_pointer + 0x160:
            return struct.pack("<fff", *self.pose)[:size]
        if address == self.player_pointer + 0x168:
            return struct.pack("<f", self.pose[2])[:size]
        if address == self.player_pointer + 0x16C:
            return struct.pack("<I", self.player_world)[:size]
        if address == self.player_pointer + 0x1EE0:
            value = self.player_pointer if self.self_pointer_valid else 0
            return struct.pack("<I", value)[:size]
        raise RuntimeError(f"unreadable 0x{address:X}+0x{size:X}")

    def readable_regions(self, **_kwargs):
        self.readable_calls += 1
        return ()

    def find_u32(self, *_args, **_kwargs):
        self.find_calls += 1
        return ()

    def close(self) -> None:
        self.closed = True


_RECOVERY_STATE = PointerRecoveryState()


@pytest.fixture(autouse=True)
def _isolate_recovery_state() -> None:
    global _RECOVERY_STATE
    _RECOVERY_STATE = PointerRecoveryState()


def _monster_config() -> NativeMonsterConfig:
    return NativeMonsterConfig(
        player_pointer_offset=0x2000,
        world_pointer_offset=0x2100,
        discovery_chunk_bytes=4096,
    )


def _position_config() -> NativePositionConfig:
    return NativePositionConfig(
        enabled=True,
        resolver="module_pointer",
        module_name="Neuz.exe",
        pointer_offset=0x100,
        x_offset=0x160,
        y_offset=0x164,
        z_offset=0x168,
    )


def _cheap_monster_config() -> NativeMonsterConfig:
    return NativeMonsterConfig(
        player_pointer_offset=0x100,
        world_pointer_offset=0x104,
        discovery_chunk_bytes=4096,
    )


def _recover(
    memory: RecoveryMemory,
    *,
    status_callback=None,
    cancellation=None,
    timeout_seconds: float | None = 1.0,
    persist: bool = False,
    clock=None,
):
    config = _monster_config()
    options = {}
    if clock is not None:
        options["clock"] = clock
    return recover_local_player_pointer(
        memory,
        module_base=memory.module_base_value,
        configured_player_pointer_offset=config.player_pointer_offset,
        state=_RECOVERY_STATE,
        monster_config=config,
        search_radii=(0x1000,),
        chunk_size=0x1000,
        persist=persist,
        status_callback=status_callback,
        cancellation=cancellation,
        timeout_seconds=timeout_seconds,
        stability_delay_seconds=0.0,
        **options,
    )


def _populate_valid_shift(
    memory: RecoveryMemory,
) -> tuple[NativeMonsterConfig, int, int]:
    config = _monster_config()
    shift = 0x280
    player_slot = config.player_pointer_offset + shift
    world_slot = config.world_pointer_offset + shift
    player = memory.actor_base
    world = 0x34000000
    memory.module_u32(player_slot, player)
    memory.module_u32(world_slot, world)
    memory.actor_u32(config.self_pointer_offset, player)
    memory.actor_u32(config.world_offset, world)
    memory.actor_i32(config.species_offset, 1)
    memory.actor_i32(config.active_species_offset, 0)
    memory.actor_i32(config.hp_offset, 1000)
    memory.actor_f32(config.x_offset, 10.0)
    memory.actor_f32(config.y_offset, 20.0)
    memory.actor_f32(config.z_offset, 30.0)
    return config, player_slot, world_slot


def test_ordinary_null_position_read_is_cheap_and_never_scans() -> None:
    memory = CheapProviderMemory(player_pointer=0)
    provider = NativeFlyffPositionProvider(memory, _position_config())

    with pytest.raises(PointerResolutionError, match="explicit pointer recovery"):
        provider.read_pose()

    assert memory.readable_calls == 0
    assert memory.reads == [(memory.module_base_value + 0x100, 4)]


def test_ordinary_stale_nonzero_position_read_is_cheap_and_never_scans() -> None:
    memory = CheapProviderMemory(
        player_pointer=0x33000000,
        pose=(math.nan, 2.0, 3.0),
    )
    provider = NativeFlyffPositionProvider(memory, _position_config())

    with pytest.raises(InvalidPlayerPoseError, match="non-finite"):
        provider.read_pose()

    assert memory.readable_calls == 0
    assert len(memory.reads) == 2


def test_ordinary_null_monster_reads_are_cheap_and_never_scan() -> None:
    memory = CheapProviderMemory(player_pointer=0, world_pointer=0)
    provider = NativeFlyffMonsterProvider(memory, _cheap_monster_config())

    with pytest.raises(NativeMonsterReadError, match="explicit pointer recovery"):
        provider.read_player_base()
    with pytest.raises(NativeMonsterReadError, match="explicit pointer recovery"):
        provider.read_world_base()

    assert memory.readable_calls == 0
    assert memory.find_calls == 0
    assert memory.reads == [
        (memory.module_base_value + 0x100, 4),
        (memory.module_base_value + 0x104, 4),
    ]


def test_ordinary_stale_player_pointer_never_reaches_actor_discovery() -> None:
    memory = CheapProviderMemory(
        player_pointer=0x33000000,
        world_pointer=0x44000000,
        self_pointer_valid=False,
    )
    provider = NativeFlyffMonsterProvider(memory, _cheap_monster_config())

    with pytest.raises(NativeMonsterReadError, match="stale or unreadable"):
        provider.read_active_actors()

    assert memory.readable_calls == 0
    assert memory.find_calls == 0


def test_ordinary_stale_world_pointer_never_reaches_actor_discovery() -> None:
    memory = CheapProviderMemory(
        player_pointer=0x33000000,
        world_pointer=0x44000000,
        player_world=0x55000000,
    )
    provider = NativeFlyffMonsterProvider(memory, _cheap_monster_config())

    with pytest.raises(NativeMonsterReadError, match="stale or inconsistent"):
        provider.read_active_actors()

    assert memory.readable_calls == 0
    assert memory.find_calls == 0


def test_concurrent_requests_join_one_underlying_scan() -> None:
    memory = RecoveryMemory(pid=88002)
    memory.block_enumeration = True
    joined = Event()
    results: list[object] = []
    errors: list[BaseException] = []

    def owner() -> None:
        try:
            results.append(_recover(memory, timeout_seconds=2.0))
        except BaseException as error:  # pragma: no cover - asserted below.
            errors.append(error)

    def waiter_status(progress: PointerRecoveryProgress) -> None:
        if progress.phase == "waiting_for_inflight":
            joined.set()

    def waiter() -> None:
        try:
            results.append(
                _recover(
                    memory,
                    status_callback=waiter_status,
                    timeout_seconds=2.0,
                )
            )
        except BaseException as error:  # pragma: no cover - asserted below.
            errors.append(error)

    first = Thread(target=owner)
    first.start()
    assert memory.enumeration_entered.wait(1.0)
    second = Thread(target=waiter)
    second.start()
    assert joined.wait(1.0)
    memory.enumeration_release.set()
    first.join(2.0)
    second.join(2.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert results == [None, None]
    assert memory.readable_calls == 1


def test_joined_persistence_request_is_honoured_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = RecoveryMemory(pid=88008)
    _populate_valid_shift(memory)
    memory.block_enumeration = True
    joined = Event()
    results: list[object] = []
    persist_calls: list[object] = []
    monkeypatch.setattr(
        recovery_module,
        "persist_recovered_pointer_offsets",
        persist_calls.append,
    )

    first = Thread(
        target=lambda: results.append(
            _recover(memory, timeout_seconds=2.0, persist=False)
        )
    )
    first.start()
    assert memory.enumeration_entered.wait(1.0)
    second = Thread(
        target=lambda: results.append(
            _recover(
                memory,
                timeout_seconds=2.0,
                persist=True,
                status_callback=lambda progress: (
                    joined.set()
                    if progress.phase == "waiting_for_inflight"
                    else None
                ),
            )
        )
    )
    second.start()
    assert joined.wait(1.0)
    memory.enumeration_release.set()
    first.join(2.0)
    second.join(2.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert len(results) == 2
    assert results[0] is not None
    assert results[1] is results[0]
    assert memory.readable_calls == 1
    assert persist_calls == [results[0]]


def test_cancellation_stops_after_the_current_bounded_chunk() -> None:
    memory = RecoveryMemory(pid=88003)
    cancellation = Event()
    memory.cancel_on_scan_read = cancellation

    result = _recover(memory, cancellation=cancellation)

    assert result is None
    assert memory.readable_calls == 1
    metrics = get_last_pointer_recovery_metrics(
        memory.pid,
        memory.module_base_value,
        state=_RECOVERY_STATE,
    )
    assert metrics is not None
    assert metrics.outcome == "cancelled"
    assert metrics.chunks_read == 1


def test_zero_timeout_stops_before_region_enumeration() -> None:
    memory = RecoveryMemory(pid=88004)

    result = _recover(memory, timeout_seconds=0.0)

    assert result is None
    assert memory.readable_calls == 0
    metrics = get_last_pointer_recovery_metrics(
        memory.pid,
        memory.module_base_value,
        state=_RECOVERY_STATE,
    )
    assert metrics is not None
    assert metrics.outcome == "deadline"
    assert metrics.deadline_is_cooperative


def test_deadline_during_cache_verification_does_not_escape_private_error() -> None:
    memory = RecoveryMemory(pid=88007)
    _populate_valid_shift(memory)
    assert _recover(memory) is not None
    readable_calls = memory.readable_calls

    class ExpiringClock:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self) -> float:
            self.calls += 1
            return 1.0 if self.calls >= 6 else 0.0

    result = _recover(
        memory,
        timeout_seconds=0.5,
        clock=ExpiringClock(),
    )

    assert result is None
    assert memory.readable_calls == readable_calls
    metrics = get_last_pointer_recovery_metrics(
        memory.pid,
        memory.module_base_value,
        state=_RECOVERY_STATE,
    )
    assert metrics is not None
    assert metrics.outcome == "deadline"


def test_failed_attempt_enters_five_second_negative_cooldown() -> None:
    memory = RecoveryMemory(pid=88005)

    assert _recover(memory) is None
    first_readable_calls = memory.readable_calls
    assert _recover(memory) is None

    assert first_readable_calls == 1
    assert memory.readable_calls == 1
    metrics = get_last_pointer_recovery_metrics(
        memory.pid,
        memory.module_base_value,
        state=_RECOVERY_STATE,
    )
    assert metrics is not None
    assert metrics.outcome == "negative_cache"
    assert metrics.negative_cache_hit
    assert metrics.cooldown_remaining_seconds > 0.0


def test_successful_shift_is_cached_and_persistence_remains_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = RecoveryMemory(pid=88006)
    _config, player_slot, world_slot = _populate_valid_shift(memory)
    persist_calls: list[object] = []
    monkeypatch.setattr(
        recovery_module,
        "persist_recovered_pointer_offsets",
        persist_calls.append,
    )
    phases: list[str] = []

    first = _recover(
        memory,
        status_callback=lambda progress: phases.append(progress.phase),
    )
    reads_after_first = memory.readable_calls
    second = _recover(memory)
    assert persist_calls == []
    third = _recover(memory, persist=True)

    assert first is not None
    assert second is first
    assert third is first
    assert first.player_pointer_offset == player_slot
    assert first.world_pointer_offset == world_slot
    assert reads_after_first == 1
    assert memory.readable_calls == 1
    assert persist_calls == [first]
    assert "regions_indexed" in phases
    assert "success" in phases
    metrics = get_last_pointer_recovery_metrics(
        memory.pid,
        memory.module_base_value,
        state=_RECOVERY_STATE,
    )
    assert metrics is not None
    assert metrics.outcome == "cache_hit"
    assert metrics.cache_hit
