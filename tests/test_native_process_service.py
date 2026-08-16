from __future__ import annotations

import struct
from dataclasses import dataclass
from threading import Event, Lock, Thread

from position.AnchoredPointerDiscovery import PointerRecoveryHints
from position.MonsterConfig import NativeMonsterConfig
from position.native_process_service import (
    NativeProcessService,
    NativeRecoveryOutcome,
)
from position.NativeFlyffMonsterProvider import NativeFlyffMonsterProvider
from position.NativeFlyffPositionProvider import NativeFlyffPositionProvider
from position.PositionConfig import NativePositionConfig
from position.RecoveredNativeProfile import (
    PROFILE_VERSION,
    RecoveredNativeProfile,
    save_profile,
)
from position.Win32ProcessMemory import ModuleInfo
from position.policy import LIVE_ATTACH_POLICY


@dataclass(frozen=True)
class _Region:
    base_address: int
    size: int
    protection: int = 0x04
    region_type: int = 0x20000


class _ServiceMemory:
    def __init__(self) -> None:
        self.pid = 77123
        self.module_base_value = 0x10000000
        self.actor_base = 0x30000000
        self.module = bytearray(0x6000)
        self.actor = bytearray(0x10000)
        self.reads: list[tuple[int, int]] = []
        self.readable_calls = 0
        self.closed = False
        self.block_enumeration = False
        self.enumeration_entered = Event()
        self.enumeration_release = Event()
        self._lock = Lock()

    def module_base(self, module_name: str) -> int:
        assert module_name == "Neuz.exe"
        return self.module_base_value

    def module_info(self, module_name: str) -> ModuleInfo:
        assert module_name == "Neuz.exe"
        return ModuleInfo(
            name="Neuz.exe",
            path=r"F:\Games\Neuz.exe",
            base_address=self.module_base_value,
            size=len(self.module),
        )

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
        needle = struct.pack("<I", int(value))
        result: list[int] = []
        for base, data in (
            (self.module_base_value, self.module),
            (self.actor_base, self.actor),
        ):
            for offset in range(0, len(data) - 3, 4):
                address = base + offset
                if address <= maximum_address and data[offset : offset + 4] == needle:
                    result.append(address)
        return tuple(result)

    def read(self, address: int, size: int) -> bytes:
        with self._lock:
            self.reads.append((address, size))
            if (
                self.module_base_value
                <= address
                and address + size <= self.module_base_value + len(self.module)
            ):
                start = address - self.module_base_value
                return bytes(self.module[start : start + size])
            if (
                self.actor_base
                <= address
                and address + size <= self.actor_base + len(self.actor)
            ):
                start = address - self.actor_base
                return bytes(self.actor[start : start + size])
        raise RuntimeError(f"unreadable 0x{address:X}+0x{size:X}")

    def readable_regions(
        self,
        *,
        maximum_address: int = 0x7FFFFFFF,
        private_only: bool = True,
    ) -> tuple[_Region, ...]:
        del private_only
        with self._lock:
            self.readable_calls += 1
        self.enumeration_entered.set()
        if self.block_enumeration:
            assert self.enumeration_release.wait(2.0)
        regions = (
            _Region(self.module_base_value, len(self.module)),
            _Region(self.actor_base, len(self.actor)),
        )
        return tuple(
            region
            for region in regions
            if region.base_address < maximum_address
        )

    def module_u32(self, offset: int, value: int) -> None:
        with self._lock:
            struct.pack_into("<I", self.module, offset, value)

    def actor_u32(self, offset: int, value: int) -> None:
        with self._lock:
            struct.pack_into("<I", self.actor, offset, value)

    def actor_i32(self, offset: int, value: int) -> None:
        with self._lock:
            struct.pack_into("<i", self.actor, offset, value)

    def actor_f32(self, offset: int, value: float) -> None:
        with self._lock:
            struct.pack_into("<f", self.actor, offset, value)

    def close(self) -> None:
        self.closed = True


def _config() -> NativeMonsterConfig:
    return NativeMonsterConfig(
        player_pointer_offset=0x2000,
        world_pointer_offset=0x2100,
        discovery_chunk_bytes=4096,
    )


def _populate_actor(
    memory: _ServiceMemory,
    config: NativeMonsterConfig,
    *,
    player_slot: int,
    world_slot: int,
) -> tuple[int, int]:
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
    return player, world


def test_coherent_snapshot_uses_six_bounded_reads_and_never_scans() -> None:
    memory = _ServiceMemory()
    config = _config()
    player, world = _populate_actor(
        memory,
        config,
        player_slot=config.player_pointer_offset,
        world_slot=config.world_pointer_offset,
    )
    service = NativeProcessService(
        memory, config, clock=lambda: 42.5, attach_policy=LIVE_ATTACH_POLICY
    )

    snapshot = service.read_pointer_snapshot()

    player_pointer_address = (
        memory.module_base_value + config.player_pointer_offset
    )
    world_pointer_address = memory.module_base_value + config.world_pointer_offset
    assert snapshot.player_pointer_address == player_pointer_address
    assert snapshot.world_pointer_address == world_pointer_address
    assert snapshot.player_base == player
    assert snapshot.world_base == world
    assert snapshot.generation == 0
    assert snapshot.captured_at == 42.5
    assert memory.reads == [
        (player_pointer_address, 4),
        (world_pointer_address, 4),
        (player + config.self_pointer_offset, 4),
        (player + config.world_offset, 4),
        (player_pointer_address, 4),
        (world_pointer_address, 4),
    ]
    assert memory.readable_calls == 0


def test_coherent_snapshot_accepts_a_world_rooted_player_chain() -> None:
    memory = _ServiceMemory()
    config = NativeMonsterConfig(
        player_pointer_offset=0x2100,
        world_pointer_offset=0x2100,
        player_pointer_chain_offsets=(0x40,),
        discovery_chunk_bytes=4096,
    )
    player = memory.actor_base
    world = memory.actor_base + 0x2000
    memory.module_u32(config.world_pointer_offset, world)
    memory.actor_u32(0x2000 + 0x40, player)
    memory.actor_u32(config.self_pointer_offset, player)
    memory.actor_u32(config.world_offset, 0)
    service = NativeProcessService(memory, config, attach_policy=LIVE_ATTACH_POLICY)

    snapshot = service.read_pointer_snapshot()

    assert snapshot.player_base == player
    assert snapshot.world_base == world
    assert memory.readable_calls == 0


def test_explicit_recovery_returns_typed_metrics_and_applies_shifted_state() -> None:
    memory = _ServiceMemory()
    config = _config()
    shift = 0x280
    player_slot = config.player_pointer_offset + shift
    world_slot = config.world_pointer_offset + shift
    player, world = _populate_actor(
        memory,
        config,
        player_slot=player_slot,
        world_slot=world_slot,
    )
    service = NativeProcessService(memory, config, attach_policy=LIVE_ATTACH_POLICY)

    result = service.recover_pointers(timeout_seconds=1.0)

    assert result.outcome is NativeRecoveryOutcome.SUCCESS
    assert result.succeeded
    assert result.applied
    assert result.metrics.outcome == "success"
    assert result.recovery is not None
    assert result.recovery.player_pointer_offset == player_slot
    assert result.recovery.world_pointer_offset == world_slot
    assert service.player_pointer_address == memory.module_base_value + player_slot
    assert service.world_pointer_address == memory.module_base_value + world_slot

    snapshot = service.read_pointer_snapshot()
    assert snapshot.player_base == player
    assert snapshot.world_base == world
    assert snapshot.generation == 1


def test_both_providers_can_consume_one_shared_step_snapshot() -> None:
    memory = _ServiceMemory()
    config = _config()
    player, world = _populate_actor(
        memory,
        config,
        player_slot=config.player_pointer_offset,
        world_slot=config.world_pointer_offset,
    )
    position_config = NativePositionConfig(
        enabled=True,
        resolver="module_pointer",
        module_name="Neuz.exe",
        pointer_offset=config.player_pointer_offset,
        x_offset=config.x_offset,
        y_offset=config.y_offset,
        z_offset=config.z_offset,
    )
    service = NativeProcessService(
        memory,
        config,
        position_config=position_config,
        attach_policy=LIVE_ATTACH_POLICY,
    )
    position_provider = NativeFlyffPositionProvider.from_native_service(
        service,
        position_config,
    )
    monster_provider = NativeFlyffMonsterProvider.from_native_service(
        service,
        config,
    )
    snapshot = service.read_pointer_snapshot()
    memory.reads.clear()

    pose = position_provider.read_pose(pointer_snapshot=snapshot)
    observed_player = monster_provider.read_player_base(
        pointer_snapshot=snapshot
    )
    observed_world = monster_provider.read_world_base(
        pointer_snapshot=snapshot
    )

    assert (pose.x, pose.y, pose.z) == (10.0, 20.0, 30.0)
    assert observed_player == player
    assert observed_world == world
    assert memory.reads == [(player + config.x_offset, 12)]

    position_provider.close()
    monster_provider.close()
    assert not memory.closed
    service.close()
    assert memory.closed


def test_snapshot_is_not_blocked_by_inflight_recovery_enumeration() -> None:
    memory = _ServiceMemory()
    config = _config()
    service = NativeProcessService(memory, config, attach_policy=LIVE_ATTACH_POLICY)
    memory.block_enumeration = True
    recovery_errors: list[BaseException] = []
    snapshot_errors: list[BaseException] = []
    snapshot_result: list[object] = []
    snapshot_done = Event()

    def recover() -> None:
        try:
            service.recover_pointers(timeout_seconds=2.0)
        except BaseException as error:  # pragma: no cover - asserted below.
            recovery_errors.append(error)

    def read_snapshot() -> None:
        try:
            snapshot_result.append(service.read_pointer_snapshot())
        except BaseException as error:  # pragma: no cover - asserted below.
            snapshot_errors.append(error)
        finally:
            snapshot_done.set()

    recovery_thread = Thread(target=recover)
    snapshot_thread = Thread(target=read_snapshot)
    recovery_thread.start()
    assert memory.enumeration_entered.wait(1.0)
    assert service.recovery_active is True

    player, world = _populate_actor(
        memory,
        config,
        player_slot=config.player_pointer_offset,
        world_slot=config.world_pointer_offset,
    )
    snapshot_thread.start()
    snapshot_completed_while_recovery_was_blocked = snapshot_done.wait(0.25)
    recovery_was_still_blocked = recovery_thread.is_alive()

    memory.enumeration_release.set()
    recovery_thread.join(2.0)
    snapshot_thread.join(2.0)

    assert snapshot_completed_while_recovery_was_blocked
    assert recovery_was_still_blocked
    assert not recovery_thread.is_alive()
    assert not snapshot_thread.is_alive()
    assert recovery_errors == []
    assert snapshot_errors == []
    assert len(snapshot_result) == 1
    snapshot = snapshot_result[0]
    assert snapshot.player_base == player
    assert snapshot.world_base == world
    assert service.recovery_active is False


def test_close_defers_handle_release_until_blocked_recovery_exits() -> None:
    memory = _ServiceMemory()
    service = NativeProcessService(
        memory, _config(), attach_policy=LIVE_ATTACH_POLICY
    )
    memory.block_enumeration = True
    results: list[object] = []
    errors: list[BaseException] = []

    def recover() -> None:
        try:
            results.append(service.recover_pointers(timeout_seconds=2.0))
        except BaseException as error:  # pragma: no cover - asserted below.
            errors.append(error)

    recovery_thread = Thread(target=recover)
    recovery_thread.start()
    assert memory.enumeration_entered.wait(1.0)

    service.close()

    assert service.is_closed
    assert not memory.closed
    assert recovery_thread.is_alive()

    memory.enumeration_release.set()
    recovery_thread.join(2.0)

    assert not recovery_thread.is_alive()
    assert errors == []
    assert len(results) == 1
    assert not results[0].applied
    assert memory.closed


def test_persisted_independent_profile_is_validated_before_full_recovery(tmp_path) -> None:
    memory = _ServiceMemory()
    config = _config()
    player = memory.actor_base
    relation_value = memory.actor_base + 0xF000
    slot = memory.module_base_value + config.player_pointer_offset
    memory.module_u32(config.player_pointer_offset, player)
    memory.actor_u32(0x3C8, player)
    memory.actor_u32(0x16C, relation_value)
    memory.actor_i32(0x81C, 26930)
    memory.actor_f32(0x160, 253.0)
    memory.actor_f32(0x164, 100.0)
    memory.actor_f32(0x168, 86.0)
    memory.actor_u32(0xF000, 0x12345678)

    for relative, species, hp, x in (
        (0x3000, 944, 400236, 250.0),
        (0x6000, 948, 250000, 260.0),
    ):
        base = memory.actor_base + relative
        memory.actor_u32(relative + 0x3C8, base)
        memory.actor_u32(relative + 0x16C, relation_value)
        memory.actor_i32(relative + 0x174, species)
        memory.actor_i32(relative + 0x81C, hp)
        memory.actor_f32(relative + 0x160, x)
        memory.actor_f32(relative + 0x164, 100.0)
        memory.actor_f32(relative + 0x168, 86.0)

    profile_path = tmp_path / "native_profile.json"
    save_profile(
        RecoveredNativeProfile(
                version=PROFILE_VERSION,
            module_name="Neuz.exe",
            module_size=len(memory.module),
            module_filename="Neuz.exe",
            module_sha256="",
            player_slot_offsets=(config.player_pointer_offset,),
            player_self_offsets=(0x3C8,),
            monster_self_offsets=(0x3C8,),
            player_hp_offset=0x81C,
            species_offset=0x174,
            active_species_offset=0x1DBC,
            monster_hp_offset=0x81C,
            x_offset=0x160,
            y_offset=0x164,
            z_offset=0x168,
            actor_stride=0x3000,
            authoritative_relation_offset=0x16C,
            anchor_species=944,
            anchor_hp=400236,
            expected_full_hp_by_species=((944, 400236),),
            saved_at_utc="2026-08-02T00:00:00+00:00",
        ),
        profile_path,
    )
    service = NativeProcessService(
        memory,
        config,
        allow_independent_recovery=True,
        recovery_profile_path=profile_path,
        attach_policy=LIVE_ATTACH_POLICY,
    )

    restored = service.try_restore_persisted_profile(
        hints=PointerRecoveryHints(known_species_ids=(944, 948)),
    )

    assert restored is True
    snapshot = service.read_pointer_snapshot()
    assert snapshot.mode == "independent"
    assert snapshot.player_base == player
    assert service.independent_reader is not None
    assert dict(service.independent_reader.authoritative_species_counts) == {944: 1, 948: 1}
    assert memory.readable_calls == 0
