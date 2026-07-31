from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import position.attachment_factory as attachment_factory
import position.factory as position_factory
import position.monster_factory as monster_factory
import position.NativePointerRecovery as pointer_recovery
import pytest
from position.MonsterConfig import NativeMonsterConfig
from position.native_process_service import NativeProcessService
from position.PositionConfig import NativePositionConfig

POSITION_BYTES = (
    b'{\r\n  "enabled": true,\r\n  "pointer_offset": "0x2000",\r\n'
    b'  "note": "preserve me"\r\n}\r\n'
)
MONSTER_BYTES = (
    b'{"enabled":true,"player_pointer_offset":"0x2000",'
    b'"world_pointer_offset":"0x2100","note":"preserve me"}\n'
)


@dataclass(frozen=True)
class _Region:
    base_address: int
    size: int
    protection: int = 0x04
    region_type: int = 0x20000


class _RecoveryMemory:
    def __init__(self) -> None:
        self.pid = 99117
        self.module_base_value = 0x10000000
        self.actor_base = 0x30000000
        self.module = bytearray(0x6000)
        self.actor = bytearray(0x3000)
        self.reads: list[tuple[int, int]] = []
        self.closed = False

    def read(self, address: int, size: int) -> bytes:
        self.reads.append((address, size))
        if (
            self.module_base_value
            <= address
            and address + size <= self.module_base_value + len(self.module)
        ):
            offset = address - self.module_base_value
            return bytes(self.module[offset : offset + size])
        if (
            self.actor_base
            <= address
            and address + size <= self.actor_base + len(self.actor)
        ):
            offset = address - self.actor_base
            return bytes(self.actor[offset : offset + size])
        raise RuntimeError(f"unreadable 0x{address:X}+0x{size:X}")

    def readable_regions(
        self,
        *,
        maximum_address: int = 0x7FFFFFFF,
        private_only: bool = True,
    ) -> tuple[_Region, ...]:
        del private_only
        return tuple(
            region
            for region in (
                _Region(self.module_base_value, len(self.module)),
                _Region(self.actor_base, len(self.actor)),
            )
            if region.base_address < maximum_address
        )

    def module_base(self, module_name: str) -> int:
        assert module_name == "Neuz.exe"
        return self.module_base_value

    def module_u32(self, offset: int, value: int) -> None:
        struct.pack_into("<I", self.module, offset, value)

    def actor_u32(self, offset: int, value: int) -> None:
        struct.pack_into("<I", self.actor, offset, value)

    def actor_i32(self, offset: int, value: int) -> None:
        struct.pack_into("<i", self.actor, offset, value)

    def actor_f32(self, offset: int, value: float) -> None:
        struct.pack_into("<f", self.actor, offset, value)

    def close(self) -> None:
        self.closed = True


def _paths(tmp_path: Path) -> tuple[Path, Path]:
    position_path = tmp_path / "native_position.json"
    monster_path = tmp_path / "native_monsters.json"
    position_path.write_bytes(POSITION_BYTES)
    monster_path.write_bytes(MONSTER_BYTES)
    return position_path, monster_path


def _recovery() -> pointer_recovery.PlayerPointerRecovery:
    return pointer_recovery.PlayerPointerRecovery(
        player_pointer_address=0x10002280,
        player_pointer_offset=0x2280,
        player_base=0x30000000,
        world_base=0x34000000,
        world_pointer_address=0x10002380,
        world_pointer_offset=0x2380,
        configured_player_pointer_offset=0x2000,
        configured_world_pointer_offset=0x2100,
        search_radius=0x1000,
        validated_candidates=1,
    )


def _anchored_recovery() -> pointer_recovery.PlayerPointerRecovery:
    return pointer_recovery.PlayerPointerRecovery(
        player_pointer_address=0x10003000,
        player_pointer_offset=0x3000,
        player_base=0x30000000,
        world_base=0x34000000,
        world_pointer_address=0x10003100,
        world_pointer_offset=0x3100,
        configured_player_pointer_offset=0x2000,
        configured_world_pointer_offset=0x2100,
        search_radius=0,
        validated_candidates=2,
        strategy="anchored_movement",
        player_pointer_chain_offsets=(0x20,),
        world_field_offset=0x180,
        world_vtable_offset=0x800,
        self_pointer_offset=0x1EF0,
        movement_validated=True,
    )


def _backup(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".pre_pointer_recovery.bak")


def _transaction_artifacts(tmp_path: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in tmp_path.iterdir()
            if "pointer_recovery" in path.name
            and not path.name.endswith(".pre_pointer_recovery.bak")
        ),
        key=lambda path: path.name,
    )


def _monster_config() -> NativeMonsterConfig:
    return NativeMonsterConfig(
        player_pointer_offset=0x2000,
        world_pointer_offset=0x2100,
        discovery_chunk_bytes=4096,
    )


def _populated_memory(
    config: NativeMonsterConfig,
) -> tuple[_RecoveryMemory, int]:
    memory = _RecoveryMemory()
    shift = 0x280
    player_slot = config.player_pointer_offset + shift
    world_slot = config.world_pointer_offset + shift
    world = 0x34000000
    memory.module_u32(player_slot, memory.actor_base)
    memory.module_u32(world_slot, world)
    memory.actor_u32(config.self_pointer_offset, memory.actor_base)
    memory.actor_u32(config.world_offset, world)
    memory.actor_i32(config.species_offset, 1)
    memory.actor_i32(config.active_species_offset, 0)
    memory.actor_i32(config.hp_offset, 1000)
    memory.actor_f32(config.x_offset, 10.0)
    memory.actor_f32(config.y_offset, 20.0)
    memory.actor_f32(config.z_offset, 30.0)
    return memory, memory.module_base_value + player_slot


def _recover(
    memory: _RecoveryMemory,
    config: NativeMonsterConfig,
    *,
    persist: bool,
    cancellation: object | None = None,
    timeout_seconds: float = 1.0,
) -> pointer_recovery.PlayerPointerRecovery | None:
    return pointer_recovery.recover_local_player_pointer(
        memory,
        module_base=memory.module_base_value,
        configured_player_pointer_offset=config.player_pointer_offset,
        state=pointer_recovery.PointerRecoveryState(),
        monster_config=config,
        search_radii=(0x1000,),
        chunk_size=0x1000,
        persist=persist,
        cancellation=cancellation,
        timeout_seconds=timeout_seconds,
        stability_samples=1,
        stability_delay_seconds=0.0,
    )


def test_second_config_replace_failure_rolls_first_back_byte_for_byte(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    position_path, monster_path = _paths(tmp_path)
    real_replace = pointer_recovery._atomic_replace
    config_replaces: list[Path] = []
    failed = False

    def fail_second_replacement(source: Path, destination: Path) -> None:
        nonlocal failed
        if source.name.endswith(".pointer_recovery.replacement.tmp"):
            config_replaces.append(destination)
            if destination == monster_path and not failed:
                failed = True
                raise OSError("injected second replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(
        pointer_recovery,
        "_atomic_replace",
        fail_second_replacement,
    )

    with pytest.raises(
        pointer_recovery.PointerPersistenceError,
        match="rolled back",
    ):
        pointer_recovery.persist_recovered_pointer_offsets(
            _recovery(),
            position_config_path=position_path,
            monster_config_path=monster_path,
        )

    assert config_replaces == [position_path, monster_path]
    assert position_path.read_bytes() == POSITION_BYTES
    assert monster_path.read_bytes() == MONSTER_BYTES
    assert not _backup(position_path).exists()
    assert not _backup(monster_path).exists()
    assert _transaction_artifacts(tmp_path) == []


def test_successful_pair_update_keeps_exact_reversible_backups(
    tmp_path: Path,
) -> None:
    position_path, monster_path = _paths(tmp_path)

    pointer_recovery.persist_recovered_pointer_offsets(
        _recovery(),
        position_config_path=position_path,
        monster_config_path=monster_path,
    )

    position = json.loads(position_path.read_text(encoding="utf-8"))
    monster = json.loads(monster_path.read_text(encoding="utf-8"))
    assert position["pointer_offset"] == "0x2280"
    assert monster["player_pointer_offset"] == "0x2280"
    assert monster["world_pointer_offset"] == "0x2380"
    assert _backup(position_path).read_bytes() == POSITION_BYTES
    assert _backup(monster_path).read_bytes() == MONSTER_BYTES
    assert _transaction_artifacts(tmp_path) == []


def test_movement_validated_anchor_persists_chain_and_inferred_layout(
    tmp_path: Path,
) -> None:
    position_path, monster_path = _paths(tmp_path)

    pointer_recovery.persist_recovered_pointer_offsets(
        _anchored_recovery(),
        position_config_path=position_path,
        monster_config_path=monster_path,
    )

    position = json.loads(position_path.read_text(encoding="utf-8"))
    monster = json.loads(monster_path.read_text(encoding="utf-8"))
    assert position["pointer_offset"] == "0x3000"
    assert position["pointer_chain_offsets"] == ["0x20"]
    assert monster["player_pointer_offset"] == "0x3000"
    assert monster["world_pointer_offset"] == "0x3100"
    assert monster["player_pointer_chain_offsets"] == ["0x20"]
    assert monster["world_pointer_chain_offsets"] == []
    assert monster["layout"]["world_offset"] == "0x180"
    assert monster["layout"]["world_vtable_offset"] == "0x800"
    assert monster["layout"]["self_pointer_offset"] == "0x1EF0"
    assert _backup(position_path).read_bytes() == POSITION_BYTES
    assert _backup(monster_path).read_bytes() == MONSTER_BYTES


def test_missing_second_config_never_touches_first_or_creates_markers(
    tmp_path: Path,
) -> None:
    position_path = tmp_path / "native_position.json"
    monster_path = tmp_path / "native_monsters.json"
    position_path.write_bytes(POSITION_BYTES)

    with pytest.raises(
        pointer_recovery.PointerPersistenceError,
        match="does not exist",
    ):
        pointer_recovery.persist_recovered_pointer_offsets(
            _recovery(),
            position_config_path=position_path,
            monster_config_path=monster_path,
        )

    assert position_path.read_bytes() == POSITION_BYTES
    assert not monster_path.exists()
    assert not _backup(position_path).exists()
    assert _transaction_artifacts(tmp_path) == []


def test_interrupted_commit_is_recovered_from_retained_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    position_path, monster_path = _paths(tmp_path)
    real_replace = pointer_recovery._atomic_replace

    def fail_commit_and_immediate_rollback(
        source: Path,
        destination: Path,
    ) -> None:
        if (
            destination == monster_path
            and source.name.endswith(".pointer_recovery.replacement.tmp")
        ):
            raise OSError("injected commit interruption")
        if (
            destination == position_path
            and source.name.endswith(".pointer_recovery.restore.tmp")
        ):
            raise OSError("injected rollback interruption")
        real_replace(source, destination)

    monkeypatch.setattr(
        pointer_recovery,
        "_atomic_replace",
        fail_commit_and_immediate_rollback,
    )
    with pytest.raises(
        pointer_recovery.PointerPersistenceError,
        match="rollback is incomplete",
    ):
        pointer_recovery.persist_recovered_pointer_offsets(
            _recovery(),
            position_config_path=position_path,
            monster_config_path=monster_path,
        )
    assert _transaction_artifacts(tmp_path)

    monkeypatch.setattr(pointer_recovery, "_atomic_replace", real_replace)
    recovered = pointer_recovery.recover_interrupted_pointer_persistence(
        position_config_path=position_path,
        monster_config_path=monster_path,
    )

    assert recovered
    assert position_path.read_bytes() == POSITION_BYTES
    assert monster_path.read_bytes() == MONSTER_BYTES
    assert not _backup(position_path).exists()
    assert not _backup(monster_path).exists()
    assert _transaction_artifacts(tmp_path) == []


def test_persist_false_cancellation_and_timeout_leave_configs_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    position_path, monster_path = _paths(tmp_path)
    config = _monster_config()
    memory, _player_slot_address = _populated_memory(config)
    real_persist = pointer_recovery.persist_recovered_pointer_offsets
    persist_calls: list[pointer_recovery.PlayerPointerRecovery] = []

    def persist_into_test_configs(
        recovery: pointer_recovery.PlayerPointerRecovery,
    ) -> None:
        persist_calls.append(recovery)
        real_persist(
            recovery,
            position_config_path=position_path,
            monster_config_path=monster_path,
        )

    monkeypatch.setattr(
        pointer_recovery,
        "persist_recovered_pointer_offsets",
        persist_into_test_configs,
    )

    assert _recover(memory, config, persist=False) is not None
    cancellation = Event()
    cancellation.set()
    assert (
        _recover(
            memory,
            config,
            persist=True,
            cancellation=cancellation,
        )
        is None
    )
    assert (
        _recover(
            memory,
            config,
            persist=True,
            timeout_seconds=0.0,
        )
        is None
    )

    assert persist_calls == []
    assert position_path.read_bytes() == POSITION_BYTES
    assert monster_path.read_bytes() == MONSTER_BYTES
    assert _transaction_artifacts(tmp_path) == []


def test_opt_in_persistence_follows_at_least_three_stability_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    position_path, monster_path = _paths(tmp_path)
    config = _monster_config()
    memory, player_slot_address = _populated_memory(config)
    real_persist = pointer_recovery.persist_recovered_pointer_offsets

    monkeypatch.setattr(
        pointer_recovery,
        "persist_recovered_pointer_offsets",
        lambda recovery: real_persist(
            recovery,
            position_config_path=position_path,
            monster_config_path=monster_path,
        ),
    )

    recovery = _recover(memory, config, persist=True)

    assert recovery is not None
    stable_slot_reads = sum(
        address == player_slot_address and size == 4
        for address, size in memory.reads
    )
    assert stable_slot_reads >= 3
    assert json.loads(position_path.read_text())["pointer_offset"] == "0x2280"
    assert _transaction_artifacts(tmp_path) == []


def test_native_service_persists_to_its_injected_custom_config_paths(
    tmp_path: Path,
) -> None:
    position_path, monster_path = _paths(tmp_path)
    config = _monster_config()
    memory, _player_slot_address = _populated_memory(config)
    position_config = NativePositionConfig(
        enabled=True,
        resolver="module_pointer",
        module_name="Neuz.exe",
        pointer_offset=config.player_pointer_offset,
    )
    service = NativeProcessService(
        memory,
        config,
        position_config=position_config,
        position_config_path=position_path,
        monster_config_path=monster_path,
    )

    result = service.recover_pointers(
        persist=True,
        timeout_seconds=1.0,
    )

    assert result.succeeded
    assert result.applied
    assert json.loads(position_path.read_text())["pointer_offset"] == "0x2280"
    monster = json.loads(monster_path.read_text())
    assert monster["player_pointer_offset"] == "0x2280"
    assert monster["world_pointer_offset"] == "0x2380"
    assert _transaction_artifacts(tmp_path) == []


def test_native_factories_recover_transaction_before_loading_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    position_path = tmp_path / "position.json"
    monster_path = tmp_path / "monster.json"

    attachment_events: list[str] = []
    monkeypatch.setattr(
        attachment_factory,
        "recover_interrupted_pointer_persistence",
        lambda **_kwargs: attachment_events.append("recover"),
    )
    monkeypatch.setattr(
        attachment_factory,
        "load_native_position_config",
        lambda _path: (
            attachment_events.append("position"),
            NativePositionConfig(enabled=False),
        )[1],
    )
    monkeypatch.setattr(
        attachment_factory,
        "load_native_monster_config",
        lambda _path: (
            attachment_events.append("monster"),
            NativeMonsterConfig(enabled=False),
        )[1],
    )
    attachment_factory.create_native_provider_attachment(
        123,
        position_config_path=position_path,
        monster_config_path=monster_path,
    )
    assert attachment_events == ["recover", "position", "monster"]

    position_events: list[str] = []
    monkeypatch.setattr(
        position_factory,
        "recover_interrupted_pointer_persistence",
        lambda **_kwargs: position_events.append("recover"),
    )
    monkeypatch.setattr(
        position_factory,
        "load_native_position_config",
        lambda _path: (
            position_events.append("position"),
            NativePositionConfig(enabled=False),
        )[1],
    )
    assert (
        position_factory.create_native_position_provider(
            123,
            config_path=position_path,
            monster_config_path=monster_path,
        )
        is None
    )
    assert position_events == ["recover", "position"]

    monster_events: list[str] = []
    monkeypatch.setattr(
        monster_factory,
        "recover_interrupted_pointer_persistence",
        lambda **_kwargs: monster_events.append("recover"),
    )
    monkeypatch.setattr(
        monster_factory,
        "load_native_monster_config",
        lambda _path: (
            monster_events.append("monster"),
            NativeMonsterConfig(enabled=False),
        )[1],
    )
    assert (
        monster_factory.create_native_monster_provider(
            123,
            config_path=monster_path,
            position_config_path=position_path,
        )
        is None
    )
    assert monster_events == ["recover", "monster"]
