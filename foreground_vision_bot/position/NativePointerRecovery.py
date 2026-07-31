from __future__ import annotations

import base64
import json
import math
import os
import struct
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Event, RLock
from time import monotonic, sleep
from typing import Protocol, cast

from .MonsterConfig import (
    DEFAULT_MONSTER_CONFIG_PATH,
    NativeMonsterConfig,
    load_native_monster_config,
)
from .PositionConfig import DEFAULT_POSITION_CONFIG_PATH


class PointerRecoveryMemory(Protocol):
    pid: int

    def read(self, address: int, size: int) -> bytes: ...

    def readable_regions(
        self,
        *,
        maximum_address: int = 0x7FFFFFFF,
        private_only: bool = True,
    ) -> tuple[object, ...]: ...


@dataclass(frozen=True, slots=True)
class PlayerPointerRecovery:
    player_pointer_address: int
    player_pointer_offset: int
    player_base: int
    world_base: int
    world_pointer_address: int | None
    world_pointer_offset: int | None
    configured_player_pointer_offset: int
    configured_world_pointer_offset: int
    search_radius: int
    validated_candidates: int


@dataclass(frozen=True, slots=True)
class PointerRecoveryMetrics:
    """Bounded diagnostics for the latest explicit recovery request."""

    pid: int
    module_base: int
    outcome: str
    elapsed_seconds: float
    region_enumerations: int = 0
    region_count: int = 0
    radii_started: int = 0
    radii_completed: int = 0
    scan_intervals: int = 0
    chunks_read: int = 0
    bytes_scanned: int = 0
    read_failures: int = 0
    aligned_words_examined: int = 0
    pointer_like_words: int = 0
    containment_checks: int = 0
    candidate_targets: int = 0
    candidate_slots: int = 0
    candidates_validated: int = 0
    cache_hit: bool = False
    negative_cache_hit: bool = False
    joined_existing_attempt: bool = False
    cooldown_remaining_seconds: float = 0.0
    deadline_is_cooperative: bool = True


@dataclass(frozen=True, slots=True)
class PointerRecoveryProgress:
    """One status update emitted by an explicit recovery attempt."""

    phase: str
    message: str
    metrics: PointerRecoveryMetrics


PointerRecoveryStatusCallback = Callable[[PointerRecoveryProgress], None]


@dataclass(frozen=True, slots=True)
class _ValidatedCandidate:
    slot_address: int
    target_address: int
    world_base: int
    distance: int
    reference_count: int
    paired_world_slot: int | None
    pair_matches: bool
    configured_world_matches: bool
    player_like: bool


@dataclass(slots=True)
class _MetricsBuilder:
    pid: int
    module_base: int
    started_at: float
    region_enumerations: int = 0
    region_count: int = 0
    radii_started: int = 0
    radii_completed: int = 0
    scan_intervals: int = 0
    chunks_read: int = 0
    bytes_scanned: int = 0
    read_failures: int = 0
    aligned_words_examined: int = 0
    pointer_like_words: int = 0
    containment_checks: int = 0
    candidate_targets: int = 0
    candidate_slots: int = 0
    candidates_validated: int = 0

    def freeze(
        self,
        outcome: str,
        *,
        now: float,
        cache_hit: bool = False,
        negative_cache_hit: bool = False,
        joined_existing_attempt: bool = False,
        cooldown_remaining_seconds: float = 0.0,
    ) -> PointerRecoveryMetrics:
        return PointerRecoveryMetrics(
            pid=self.pid,
            module_base=self.module_base,
            outcome=outcome,
            elapsed_seconds=max(0.0, float(now) - self.started_at),
            region_enumerations=self.region_enumerations,
            region_count=self.region_count,
            radii_started=self.radii_started,
            radii_completed=self.radii_completed,
            scan_intervals=self.scan_intervals,
            chunks_read=self.chunks_read,
            bytes_scanned=self.bytes_scanned,
            read_failures=self.read_failures,
            aligned_words_examined=self.aligned_words_examined,
            pointer_like_words=self.pointer_like_words,
            containment_checks=self.containment_checks,
            candidate_targets=self.candidate_targets,
            candidate_slots=self.candidate_slots,
            candidates_validated=self.candidates_validated,
            cache_hit=cache_hit,
            negative_cache_hit=negative_cache_hit,
            joined_existing_attempt=joined_existing_attempt,
            cooldown_remaining_seconds=max(
                0.0,
                float(cooldown_remaining_seconds),
            ),
        )


@dataclass(slots=True)
class _RecoveryFlight:
    completed: Event = field(default_factory=Event)
    result: PlayerPointerRecovery | None = None
    metrics: PointerRecoveryMetrics | None = None
    persist_requested: bool = False


@dataclass(frozen=True, slots=True)
class _RegionIndex:
    """Merged readable intervals with logarithmic containment lookup."""

    starts: tuple[int, ...]
    stops: tuple[int, ...]

    @classmethod
    def build(cls, regions: tuple[object, ...]) -> _RegionIndex:
        bounds = sorted(
            (start, stop)
            for start, stop in (_region_bounds(region) for region in regions)
            if stop > start
        )
        merged: list[list[int]] = []
        for start, stop in bounds:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], stop)
            else:
                merged.append([start, stop])
        return cls(
            starts=tuple(item[0] for item in merged),
            stops=tuple(item[1] for item in merged),
        )

    def contains(self, address: int, size: int = 1) -> bool:
        if not self.starts:
            return False
        start = int(address)
        end = start + max(1, int(size))
        index = bisect_right(self.starts, start) - 1
        return index >= 0 and end <= self.stops[index]

    def intersections(self, start: int, stop: int) -> tuple[tuple[int, int], ...]:
        if stop <= start or not self.starts:
            return ()
        index = max(0, bisect_right(self.starts, int(start)) - 1)
        while index < len(self.starts) and self.stops[index] <= start:
            index += 1
        result: list[tuple[int, int]] = []
        while index < len(self.starts) and self.starts[index] < stop:
            clipped_start = max(int(start), self.starts[index])
            clipped_stop = min(int(stop), self.stops[index])
            if clipped_stop > clipped_start:
                result.append((clipped_start, clipped_stop))
            index += 1
        return tuple(result)


class _AttemptStopped(Exception):
    def __init__(self, outcome: str) -> None:
        super().__init__(outcome)
        self.outcome = outcome


@dataclass(frozen=True, slots=True)
class _AttemptControl:
    cancellation: object | None
    deadline: float
    clock: Callable[[], float]

    def check(self) -> None:
        if _cancellation_requested(self.cancellation):
            raise _AttemptStopped("cancelled")
        if self.clock() >= self.deadline:
            raise _AttemptStopped("deadline")

    def wait(self, seconds: float) -> None:
        wait_until = min(self.deadline, self.clock() + max(0.0, float(seconds)))
        while self.clock() < wait_until:
            self.check()
            sleep(min(0.01, max(0.0, wait_until - self.clock())))
        self.check()


class PointerRecoveryState:
    """Per-attachment cache, cooldown, and single-flight recovery state."""

    def __init__(self) -> None:
        self.lock = RLock()
        self.recovery_cache: dict[
            tuple[int, int],
            PlayerPointerRecovery,
        ] = {}
        self.negative_cache: dict[tuple[int, int], float] = {}
        self.inflight: dict[tuple[int, int], _RecoveryFlight] = {}
        self.last_metrics: dict[tuple[int, int], PointerRecoveryMetrics] = {}

    def clear(
        self,
        *,
        pid: int | None = None,
        module_base: int | None = None,
    ) -> None:
        with self.lock:
            if pid is None and module_base is None:
                self.recovery_cache.clear()
                self.negative_cache.clear()
                self.last_metrics.clear()
                return
            keys = (
                set(self.recovery_cache)
                | set(self.negative_cache)
                | set(self.last_metrics)
            )
            for key in keys:
                if pid is not None and key[0] != int(pid):
                    continue
                if module_base is not None and key[1] != int(module_base):
                    continue
                self.recovery_cache.pop(key, None)
                self.negative_cache.pop(key, None)
                self.last_metrics.pop(key, None)

    def metrics_for(
        self,
        pid: int,
        module_base: int,
    ) -> PointerRecoveryMetrics | None:
        with self.lock:
            return self.last_metrics.get((int(pid), int(module_base)))


_PERSIST_LOCK = RLock()
MINIMUM_NEGATIVE_COOLDOWN_SECONDS = 5.0
DEFAULT_RECOVERY_TIMEOUT_SECONDS = 0.5


def _u32(memory: PointerRecoveryMemory, address: int) -> int:
    return int(struct.unpack("<I", memory.read(address, 4))[0])


def _i32(memory: PointerRecoveryMemory, address: int) -> int:
    return int(struct.unpack("<i", memory.read(address, 4))[0])


def _f32(memory: PointerRecoveryMemory, address: int) -> float:
    return float(struct.unpack("<f", memory.read(address, 4))[0])


def _region_bounds(region: object) -> tuple[int, int]:
    start = int(getattr(region, "base_address"))
    size = int(getattr(region, "size"))
    return start, start + max(0, size)


def _contains(
    regions: tuple[object, ...] | _RegionIndex,
    address: int,
    size: int = 1,
) -> bool:
    """Compatibility wrapper; recovery builds one index and reuses it."""

    index = (
        regions
        if isinstance(regions, _RegionIndex)
        else _RegionIndex.build(regions)
    )
    return index.contains(address, size)


def _cancellation_requested(cancellation: object | None) -> bool:
    if cancellation is None:
        return False
    cancelled = getattr(cancellation, "cancelled", None)
    if cancelled is not None:
        return bool(cancelled() if callable(cancelled) else cancelled)
    is_set = getattr(cancellation, "is_set", None)
    if callable(is_set):
        return bool(is_set())
    return False


def _notify(
    callback: PointerRecoveryStatusCallback | None,
    phase: str,
    message: str,
    metrics: PointerRecoveryMetrics,
) -> None:
    if callback is None:
        return
    try:
        callback(
            PointerRecoveryProgress(
                phase=phase,
                message=message,
                metrics=metrics,
            )
        )
    except Exception:
        # Recovery diagnostics must never turn a read-only diagnostic into a
        # runtime failure.
        return


def clear_pointer_recovery_state(
    *,
    state: PointerRecoveryState | None = None,
    pid: int | None = None,
    module_base: int | None = None,
) -> None:
    """Invalidate one explicitly owned recovery state.

    Calls without ``state`` are retained as a compatibility no-op. New code
    owns a :class:`PointerRecoveryState` for the lifetime of one attachment.
    """

    if state is not None:
        state.clear(pid=pid, module_base=module_base)


def get_last_pointer_recovery_metrics(
    pid: int,
    module_base: int,
    *,
    state: PointerRecoveryState | None = None,
) -> PointerRecoveryMetrics | None:
    if state is None:
        return None
    return state.metrics_for(pid, module_base)


def _read_configured_world_hint(
    memory: PointerRecoveryMemory,
    module_base: int,
    config: NativeMonsterConfig,
) -> int | None:
    try:
        value = _u32(memory, module_base + config.world_pointer_offset)
    except Exception:
        return None
    return value if value > 0 else None


def _validate_candidate(
    memory: PointerRecoveryMemory,
    *,
    slot_address: int,
    target_address: int,
    reference_count: int,
    configured_slot_address: int,
    module_base: int,
    config: NativeMonsterConfig,
    configured_world_hint: int | None,
) -> _ValidatedCandidate | None:
    try:
        if _u32(memory, target_address + config.self_pointer_offset) != target_address:
            return None
        world = _u32(memory, target_address + config.world_offset)
        if world <= 0:
            return None
        x = _f32(memory, target_address + config.x_offset)
        y = _f32(memory, target_address + config.y_offset)
        z = _f32(memory, target_address + config.z_offset)
        if not all(math.isfinite(value) for value in (x, y, z)):
            return None
        limit = float(config.maximum_absolute_coordinate)
        if any(abs(value) > limit for value in (x, y, z)):
            return None

        species = _i32(memory, target_address + config.species_offset)
        active_species = _i32(memory, target_address + config.active_species_offset)
        hp = _i32(memory, target_address + config.hp_offset)
        if hp <= 0:
            return None
    except Exception:
        return None

    offset_delta = config.world_pointer_offset - config.player_pointer_offset
    paired_world_slot = slot_address + offset_delta
    pair_matches = False
    if paired_world_slot > module_base:
        try:
            pair_matches = _u32(memory, paired_world_slot) == world
        except Exception:
            pair_matches = False

    configured_world_matches = (
        configured_world_hint is not None and configured_world_hint == world
    )
    player_like = not (species > 0 and active_species == species)
    return _ValidatedCandidate(
        slot_address=slot_address,
        target_address=target_address,
        world_base=world,
        distance=abs(slot_address - configured_slot_address),
        reference_count=reference_count,
        paired_world_slot=paired_world_slot if pair_matches else None,
        pair_matches=pair_matches,
        configured_world_matches=configured_world_matches,
        player_like=player_like,
    )


def _stable_candidate(
    memory: PointerRecoveryMemory,
    candidate: _ValidatedCandidate,
    config: NativeMonsterConfig,
    *,
    samples: int = 3,
    delay_seconds: float = 0.03,
    control: _AttemptControl | None = None,
) -> bool:
    # A recovered global may be persisted later from the success cache, so
    # every cached result must meet the same minimum multi-sample standard.
    sample_count = max(3, int(samples))
    for index in range(sample_count):
        if control is not None:
            control.check()
        try:
            if _u32(memory, candidate.slot_address) != candidate.target_address:
                return False
            if (
                _u32(
                    memory,
                    candidate.target_address + config.self_pointer_offset,
                )
                != candidate.target_address
            ):
                return False
            if (
                _u32(memory, candidate.target_address + config.world_offset)
                != candidate.world_base
            ):
                return False
            coordinates = (
                _f32(memory, candidate.target_address + config.x_offset),
                _f32(memory, candidate.target_address + config.y_offset),
                _f32(memory, candidate.target_address + config.z_offset),
            )
            if not all(math.isfinite(value) for value in coordinates):
                return False
        except Exception:
            return False
        if index + 1 < sample_count and delay_seconds > 0.0:
            if control is None:
                sleep(delay_seconds)
            else:
                control.wait(delay_seconds)
    return True


class PointerPersistenceError(RuntimeError):
    """A two-config pointer update could not commit or recover safely."""


@dataclass(frozen=True, slots=True)
class _PersistenceTarget:
    path: Path
    original_existed: bool
    original_bytes: bytes
    replacement_bytes: bytes
    changed: bool
    backup_path: Path
    backup_existed: bool
    replacement_path: Path
    backup_stage_path: Path
    restore_path: Path


_POINTER_TRANSACTION_VERSION = 1


def _canonical_path(path: str | Path) -> Path:
    return Path(path).resolve(strict=False)


def _journal_path(position_config_path: Path) -> Path:
    return position_config_path.with_name(
        f".{position_config_path.name}.pointer_recovery.transaction.json"
    )


def _journal_stage_path(journal_path: Path) -> Path:
    return journal_path.with_name(f"{journal_path.name}.tmp")


def _target_paths(path: Path) -> tuple[Path, Path, Path, Path]:
    backup = path.with_suffix(path.suffix + ".pre_pointer_recovery.bak")
    replacement = path.with_name(
        f".{path.name}.pointer_recovery.replacement.tmp"
    )
    backup_stage = path.with_name(
        f".{path.name}.pointer_recovery.backup.tmp"
    )
    restore = path.with_name(f".{path.name}.pointer_recovery.restore.tmp")
    return backup, replacement, backup_stage, restore


def _sync_directory(directory: Path) -> None:
    """Best-effort metadata flush; opening directories is unsupported on Windows."""

    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _write_durable_file(path: Path, data: bytes) -> None:
    path.unlink(missing_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_replace(source: Path, destination: Path) -> None:
    source.replace(destination)
    _sync_directory(destination.parent)


def _remove_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _sync_directory(path.parent)


def _prepare_persistence_target(
    path: Path,
    updates: dict[str, str],
) -> _PersistenceTarget:
    if not path.parent.is_dir():
        raise PointerPersistenceError(
            f"Configuration directory does not exist: {path.parent}"
        )
    original_existed = path.is_file()
    if not original_existed:
        raise PointerPersistenceError(f"Configuration does not exist: {path}")
    original_bytes = path.read_bytes()
    try:
        payload = json.loads(original_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PointerPersistenceError(
            f"Configuration is not valid UTF-8 JSON: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise PointerPersistenceError(
            f"Configuration root must be an object: {path}"
        )

    changed = False
    for key, value in updates.items():
        if payload.get(key) != value:
            payload[key] = value
            changed = True
    replacement_bytes = (
        (json.dumps(payload, indent=2) + "\n").encode("utf-8")
        if changed
        else original_bytes
    )
    try:
        replacement_payload = json.loads(replacement_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PointerPersistenceError(
            f"Prepared replacement is not valid JSON: {path}"
        ) from error
    if not isinstance(replacement_payload, dict) or any(
        replacement_payload.get(key) != value for key, value in updates.items()
    ):
        raise PointerPersistenceError(
            f"Prepared replacement failed validation: {path}"
        )

    backup, replacement, backup_stage, restore = _target_paths(path)
    return _PersistenceTarget(
        path=path,
        original_existed=original_existed,
        original_bytes=original_bytes,
        replacement_bytes=replacement_bytes,
        changed=changed,
        backup_path=backup,
        backup_existed=backup.exists(),
        replacement_path=replacement,
        backup_stage_path=backup_stage,
        restore_path=restore,
    )


def _target_record(target: _PersistenceTarget) -> dict[str, object]:
    return {
        "path": str(target.path),
        "original_existed": target.original_existed,
        "original_base64": base64.b64encode(target.original_bytes).decode("ascii"),
        "changed": target.changed,
        "backup_existed": target.backup_existed,
    }


def _transaction_bytes(targets: tuple[_PersistenceTarget, ...]) -> bytes:
    payload = {
        "version": _POINTER_TRANSACTION_VERSION,
        "targets": [_target_record(target) for target in targets],
    }
    return (json.dumps(payload, indent=2) + "\n").encode("utf-8")


def _parse_transaction_target(value: object) -> _PersistenceTarget:
    if not isinstance(value, dict):
        raise PointerPersistenceError("Transaction target must be an object")
    path_value = value.get("path")
    original_existed = value.get("original_existed")
    original_base64 = value.get("original_base64")
    changed = value.get("changed")
    backup_existed = value.get("backup_existed")
    if (
        not isinstance(path_value, str)
        or not isinstance(original_existed, bool)
        or not isinstance(original_base64, str)
        or not isinstance(changed, bool)
        or not isinstance(backup_existed, bool)
    ):
        raise PointerPersistenceError("Transaction target fields are invalid")
    try:
        original_bytes = base64.b64decode(
            original_base64.encode("ascii"),
            validate=True,
        )
    except (UnicodeEncodeError, ValueError) as error:
        raise PointerPersistenceError(
            "Transaction target original bytes are invalid"
        ) from error
    path = _canonical_path(path_value)
    backup, replacement, backup_stage, restore = _target_paths(path)
    return _PersistenceTarget(
        path=path,
        original_existed=original_existed,
        original_bytes=original_bytes,
        replacement_bytes=b"",
        changed=changed,
        backup_path=backup,
        backup_existed=backup_existed,
        replacement_path=replacement,
        backup_stage_path=backup_stage,
        restore_path=restore,
    )


def _read_transaction(
    journal: Path,
    expected_paths: tuple[Path, Path],
) -> tuple[_PersistenceTarget, ...]:
    try:
        value = json.loads(journal.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PointerPersistenceError(
            f"Could not read pointer persistence journal: {journal}"
        ) from error
    if not isinstance(value, dict):
        raise PointerPersistenceError("Pointer persistence journal is invalid")
    if value.get("version") != _POINTER_TRANSACTION_VERSION:
        raise PointerPersistenceError(
            "Pointer persistence journal version is unsupported"
        )
    raw_targets = value.get("targets")
    if not isinstance(raw_targets, list) or len(raw_targets) != 2:
        raise PointerPersistenceError(
            "Pointer persistence journal must contain two targets"
        )
    targets = tuple(_parse_transaction_target(item) for item in raw_targets)
    if tuple(target.path for target in targets) != expected_paths:
        raise PointerPersistenceError(
            "Pointer persistence journal targets do not match this request"
        )
    return targets


def _cleanup_target_temps(targets: tuple[_PersistenceTarget, ...]) -> None:
    for target in targets:
        _remove_file(target.replacement_path)
        _remove_file(target.backup_stage_path)
        _remove_file(target.restore_path)


def _restore_persistence_target(target: _PersistenceTarget) -> None:
    if target.original_existed:
        current = target.path.read_bytes() if target.path.is_file() else None
        if current != target.original_bytes:
            _write_durable_file(target.restore_path, target.original_bytes)
            _atomic_replace(target.restore_path, target.path)
    elif target.path.exists():
        _remove_file(target.path)

    if target.changed and not target.backup_existed:
        _remove_file(target.backup_path)


def recover_interrupted_pointer_persistence(
    *,
    position_config_path: str | Path = DEFAULT_POSITION_CONFIG_PATH,
    monster_config_path: str | Path = DEFAULT_MONSTER_CONFIG_PATH,
) -> bool:
    """Rollback an interrupted two-config commit, if its journal is present."""

    position_path = _canonical_path(position_config_path)
    monster_path = _canonical_path(monster_config_path)
    if position_path == monster_path:
        raise PointerPersistenceError(
            "Position and monster configurations must be different files"
        )
    expected_paths = (position_path, monster_path)
    journal = _journal_path(position_path)
    journal_stage = _journal_stage_path(journal)
    with _PERSIST_LOCK:
        if not journal.exists():
            _remove_file(journal_stage)
            for path in expected_paths:
                backup, replacement, backup_stage, restore = _target_paths(path)
                del backup
                _remove_file(replacement)
                _remove_file(backup_stage)
                _remove_file(restore)
            return False

        targets = _read_transaction(journal, expected_paths)
        try:
            for target in targets:
                _restore_persistence_target(target)
            _cleanup_target_temps(targets)
            _remove_file(journal_stage)
            _remove_file(journal)
        except Exception as error:
            raise PointerPersistenceError(
                "Interrupted pointer persistence rollback is incomplete; "
                f"journal retained at {journal}"
            ) from error
        return True


def persist_recovered_pointer_offsets(
    recovery: PlayerPointerRecovery,
    *,
    position_config_path: str | Path = DEFAULT_POSITION_CONFIG_PATH,
    monster_config_path: str | Path = DEFAULT_MONSTER_CONFIG_PATH,
) -> None:
    with _PERSIST_LOCK:
        position_path = _canonical_path(position_config_path)
        monster_path = _canonical_path(monster_config_path)
        if position_path == monster_path:
            raise PointerPersistenceError(
                "Position and monster configurations must be different files"
            )
        recover_interrupted_pointer_persistence(
            position_config_path=position_path,
            monster_config_path=monster_path,
        )
        monster_updates = {
            "player_pointer_offset": f"0x{recovery.player_pointer_offset:X}",
        }
        if recovery.world_pointer_offset is not None:
            monster_updates["world_pointer_offset"] = (
                f"0x{recovery.world_pointer_offset:X}"
            )
        targets = (
            _prepare_persistence_target(
                position_path,
                {"pointer_offset": f"0x{recovery.player_pointer_offset:X}"},
            ),
            _prepare_persistence_target(monster_path, monster_updates),
        )
        if not any(target.changed for target in targets):
            return

        journal = _journal_path(position_path)
        journal_stage = _journal_stage_path(journal)
        try:
            for target in targets:
                if not target.changed:
                    continue
                _write_durable_file(
                    target.replacement_path,
                    target.replacement_bytes,
                )
                if not target.backup_existed:
                    _write_durable_file(
                        target.backup_stage_path,
                        target.original_bytes,
                    )

            # Re-read both sources only after every replacement has been
            # prepared and validated, but before the recovery journal or any
            # destination is touched.
            for target in targets:
                if (
                    target.path.is_file() != target.original_existed
                    or (
                        target.original_existed
                        and target.path.read_bytes() != target.original_bytes
                    )
                    or target.backup_path.exists() != target.backup_existed
                ):
                    raise PointerPersistenceError(
                        "Pointer configuration changed while preparing the "
                        f"transaction: {target.path}"
                    )

            _write_durable_file(journal_stage, _transaction_bytes(targets))
            _atomic_replace(journal_stage, journal)
        except Exception:
            _cleanup_target_temps(targets)
            _remove_file(journal_stage)
            raise

        try:
            for target in targets:
                if target.changed and not target.backup_existed:
                    _atomic_replace(
                        target.backup_stage_path,
                        target.backup_path,
                    )
            for target in targets:
                if target.changed:
                    _atomic_replace(target.replacement_path, target.path)
            _remove_file(journal)
        except Exception as commit_error:
            try:
                recover_interrupted_pointer_persistence(
                    position_config_path=position_path,
                    monster_config_path=monster_path,
                )
            except Exception as rollback_error:
                raise PointerPersistenceError(
                    "Pointer configuration commit failed and rollback is "
                    f"incomplete; recovery journal retained at {journal}"
                ) from rollback_error
            raise PointerPersistenceError(
                "Pointer configuration commit failed and was rolled back"
            ) from commit_error


def _persist_if_requested(
    recovery: PlayerPointerRecovery,
    *,
    requested: bool,
    position_config_path: str | Path | None = None,
    monster_config_path: str | Path | None = None,
) -> None:
    if not requested:
        return
    try:
        if position_config_path is None and monster_config_path is None:
            persist_recovered_pointer_offsets(recovery)
        elif position_config_path is None or monster_config_path is None:
            raise ValueError(
                "Both persistence config paths must be provided together"
            )
        else:
            persist_recovered_pointer_offsets(
                recovery,
                position_config_path=position_config_path,
                monster_config_path=monster_config_path,
            )
    except Exception as error:
        print(
            "Native pointer recovery succeeded but config persistence "
            f"failed: {type(error).__name__}: {error}"
        )


def _verify_cached(
    memory: PointerRecoveryMemory,
    cached: PlayerPointerRecovery,
    config: NativeMonsterConfig,
) -> bool:
    try:
        return (
            _u32(memory, cached.player_pointer_address) == cached.player_base
            and _u32(memory, cached.player_base + config.self_pointer_offset)
            == cached.player_base
            and _u32(memory, cached.player_base + config.world_offset)
            == cached.world_base
        )
    except Exception:
        return False


def _progress_metrics(
    builder: _MetricsBuilder,
    control: _AttemptControl,
) -> PointerRecoveryMetrics:
    return builder.freeze("running", now=control.clock())


def _scan_new_band(
    memory: PointerRecoveryMemory,
    *,
    start: int,
    stop: int,
    chunk_size: int,
    max_actor_span: int,
    region_index: _RegionIndex,
    slot_refs: dict[int, list[int]],
    metrics: _MetricsBuilder,
    control: _AttemptControl,
) -> None:
    intervals = region_index.intersections(start, stop)
    metrics.scan_intervals += len(intervals)
    for interval_start, interval_stop in intervals:
        cursor = interval_start
        while cursor < interval_stop:
            control.check()
            amount = min(max(0x1000, int(chunk_size)), interval_stop - cursor)
            try:
                data = memory.read(cursor, amount)
            except Exception:
                metrics.read_failures += 1
                cursor += amount
                continue
            metrics.chunks_read += 1
            metrics.bytes_scanned += len(data)
            control.check()

            first = (-cursor) % 4
            for relative in range(first, len(data) - 3, 4):
                if (relative - first) % 0x400 == 0:
                    control.check()
                metrics.aligned_words_examined += 1
                target = int(struct.unpack_from("<I", data, relative)[0])
                if target <= 0x10000 or target % 4 != 0:
                    continue
                metrics.pointer_like_words += 1
                metrics.containment_checks += 1
                if not region_index.contains(target, max_actor_span):
                    continue
                slot = cursor + relative
                refs = slot_refs.setdefault(target, [])
                if len(refs) < 8:
                    refs.append(slot)
            cursor += amount


def _perform_recovery_attempt(
    memory: PointerRecoveryMemory,
    *,
    module_base: int,
    configured_player_pointer_offset: int,
    config: NativeMonsterConfig,
    search_radii: tuple[int, ...],
    chunk_size: int,
    maximum_candidates: int,
    stability_samples: int,
    stability_delay_seconds: float,
    metrics: _MetricsBuilder,
    control: _AttemptControl,
    status_callback: PointerRecoveryStatusCallback | None,
) -> tuple[PlayerPointerRecovery | None, str]:
    readable_regions_fn = cast(
        Callable[..., tuple[object, ...]] | None,
        getattr(memory, "readable_regions", None),
    )
    if not callable(readable_regions_fn):
        return None, "unavailable"

    control.check()
    configured_slot = int(module_base) + int(configured_player_pointer_offset)
    configured_world_hint = _read_configured_world_hint(memory, module_base, config)
    control.check()

    metrics.region_enumerations = 1
    try:
        regions = tuple(
            readable_regions_fn(
                maximum_address=config.maximum_scan_address,
                private_only=False,
            )
        )
    except Exception:
        return None, "region_error"
    control.check()

    metrics.region_count = len(regions)
    region_index = _RegionIndex.build(regions)
    _notify(
        status_callback,
        "regions_indexed",
        f"Indexed {len(region_index.starts)} merged readable intervals.",
        _progress_metrics(metrics, control),
    )

    max_actor_span = max(
        config.self_pointer_offset,
        config.active_species_offset,
        config.hp_offset,
        config.z_offset,
    ) + 4
    radii = tuple(
        sorted({max(0x1000, int(value)) for value in search_radii})
    )
    slot_refs: dict[int, list[int]] = {}
    validated: list[_ValidatedCandidate] = []
    inspected_pairs: set[tuple[int, int]] = set()
    previous_start: int | None = None
    previous_stop: int | None = None

    for radius in radii:
        control.check()
        metrics.radii_started += 1
        scan_start = max(int(module_base), configured_slot - radius)
        scan_stop = configured_slot + radius + 4
        _notify(
            status_callback,
            "scanning",
            f"Scanning new memory bands for radius 0x{radius:X}.",
            _progress_metrics(metrics, control),
        )

        if previous_start is None or previous_stop is None:
            bands = ((scan_start, scan_stop),)
        else:
            pending: list[tuple[int, int]] = []
            if scan_start < previous_start:
                pending.append((scan_start, previous_start))
            if scan_stop > previous_stop:
                pending.append((previous_stop, scan_stop))
            bands = tuple(pending)

        for band_start, band_stop in bands:
            _scan_new_band(
                memory,
                start=band_start,
                stop=band_stop,
                chunk_size=chunk_size,
                max_actor_span=max_actor_span,
                region_index=region_index,
                slot_refs=slot_refs,
                metrics=metrics,
                control=control,
            )
        previous_start = scan_start
        previous_stop = scan_stop
        metrics.candidate_targets = len(slot_refs)
        metrics.candidate_slots = sum(len(slots) for slots in slot_refs.values())

        ordered_slots: list[tuple[int, int, int]] = []
        for target, slots in slot_refs.items():
            for slot in slots:
                ordered_slots.append(
                    (abs(slot - configured_slot), slot, target)
                )
        ordered_slots.sort(key=lambda item: item[0])

        for _distance, slot, target in ordered_slots[: max(1, maximum_candidates)]:
            control.check()
            pair = (slot, target)
            if pair in inspected_pairs:
                continue
            inspected_pairs.add(pair)
            metrics.candidates_validated += 1
            candidate = _validate_candidate(
                memory,
                slot_address=slot,
                target_address=target,
                reference_count=len(slot_refs.get(target, ())),
                configured_slot_address=configured_slot,
                module_base=module_base,
                config=config,
                configured_world_hint=configured_world_hint,
            )
            if candidate is not None:
                validated.append(candidate)

        strong = [
            candidate
            for candidate in validated
            if candidate.pair_matches or candidate.configured_world_matches
        ]
        strong.sort(
            key=lambda candidate: (
                int(candidate.pair_matches),
                int(candidate.configured_world_matches),
                int(candidate.player_like),
                min(candidate.reference_count, 8),
                -candidate.distance,
            ),
            reverse=True,
        )
        metrics.radii_completed += 1
        if not strong:
            _notify(
                status_callback,
                "radius_complete",
                f"No validated replacement inside radius 0x{radius:X}.",
                _progress_metrics(metrics, control),
            )
            continue

        selected = strong[0]
        if not _stable_candidate(
            memory,
            selected,
            config,
            samples=stability_samples,
            delay_seconds=stability_delay_seconds,
            control=control,
        ):
            _notify(
                status_callback,
                "radius_complete",
                f"Candidate inside radius 0x{radius:X} was not stable.",
                _progress_metrics(metrics, control),
            )
            continue

        player_offset = selected.slot_address - int(module_base)
        world_slot = selected.paired_world_slot
        world_offset = (
            None if world_slot is None else world_slot - int(module_base)
        )
        return (
            PlayerPointerRecovery(
                player_pointer_address=selected.slot_address,
                player_pointer_offset=player_offset,
                player_base=selected.target_address,
                world_base=selected.world_base,
                world_pointer_address=world_slot,
                world_pointer_offset=world_offset,
                configured_player_pointer_offset=int(
                    configured_player_pointer_offset
                ),
                configured_world_pointer_offset=int(config.world_pointer_offset),
                search_radius=radius,
                validated_candidates=len(validated),
            ),
            "success",
        )

    return None, "not_found"


def recover_local_player_pointer(
    memory: PointerRecoveryMemory,
    *,
    module_base: int,
    configured_player_pointer_offset: int,
    state: PointerRecoveryState | None = None,
    monster_config: NativeMonsterConfig | None = None,
    search_radii: tuple[int, ...] = (0x20000, 0x100000, 0x400000, 0x800000),
    chunk_size: int = 0x10000,
    maximum_candidates: int = 4096,
    persist: bool = False,
    position_config_path: str | Path | None = None,
    monster_config_path: str | Path | None = None,
    cancellation: object | None = None,
    deadline: float | None = None,
    timeout_seconds: float = DEFAULT_RECOVERY_TIMEOUT_SECONDS,
    negative_cooldown_seconds: float = MINIMUM_NEGATIVE_COOLDOWN_SECONDS,
    status_callback: PointerRecoveryStatusCallback | None = None,
    clock: Callable[[], float] = monotonic,
    stability_samples: int = 3,
    stability_delay_seconds: float = 0.03,
) -> PlayerPointerRecovery | None:
    """Explicitly recover a shifted Neuz local-player global.

    Ordinary position and monster reads never call this function. Explicit
    attempts sharing a :class:`PointerRecoveryState` are single-flight per
    ``(pid, module_base)``, enumerate and index readable regions once, scan each
    expanding memory band at most once, and stop cooperatively at the deadline
    or cancellation boundary. Failed complete attempts enter an
    attachment-scoped cooldown to prevent retry storms.

    Deadline checks surround each synchronous backend call and run throughout
    CPU scanning. Python cannot preempt a backend call already in progress, so
    ``PointerRecoveryMetrics.deadline_is_cooperative`` is explicit and callers
    must keep this diagnostic behind a bounded lifecycle owner.
    """

    recovery_state = state or PointerRecoveryState()
    config = monster_config or load_native_monster_config()
    pid = int(getattr(memory, "pid", 0))
    cache_key = (pid, int(module_base))
    started_at = clock()
    bounded_deadline = started_at + max(0.0, float(timeout_seconds))
    if deadline is not None:
        bounded_deadline = min(bounded_deadline, float(deadline))
    control = _AttemptControl(
        cancellation=cancellation,
        deadline=bounded_deadline,
        clock=clock,
    )
    builder = _MetricsBuilder(
        pid=pid,
        module_base=int(module_base),
        started_at=started_at,
    )
    minimum_cooldown = max(
        MINIMUM_NEGATIVE_COOLDOWN_SECONDS,
        float(negative_cooldown_seconds),
    )

    _notify(
        status_callback,
        "started",
        "Explicit native pointer recovery started.",
        _progress_metrics(builder, control),
    )

    while True:
        try:
            control.check()
        except _AttemptStopped as stopped:
            metrics = builder.freeze(stopped.outcome, now=clock())
            with recovery_state.lock:
                recovery_state.last_metrics[cache_key] = metrics
            _notify(
                status_callback,
                stopped.outcome,
                f"Pointer recovery stopped: {stopped.outcome}.",
                metrics,
            )
            return None

        now = clock()
        with recovery_state.lock:
            cached = recovery_state.recovery_cache.get(cache_key)
            cooldown_until = recovery_state.negative_cache.get(cache_key)
            flight = recovery_state.inflight.get(cache_key)
            if cooldown_until is not None and cooldown_until <= now:
                recovery_state.negative_cache.pop(cache_key, None)
                cooldown_until = None
            if cached is None and cooldown_until is None and flight is None:
                flight = _RecoveryFlight(persist_requested=bool(persist))
                recovery_state.inflight[cache_key] = flight
                owner = True
            else:
                owner = False
                if (
                    persist
                    and cached is None
                    and cooldown_until is None
                    and flight is not None
                ):
                    flight.persist_requested = True

        if cached is not None:
            try:
                control.check()
                cached_is_valid = _verify_cached(memory, cached, config)
                control.check()
            except _AttemptStopped as stopped:
                metrics = builder.freeze(stopped.outcome, now=clock())
                with recovery_state.lock:
                    recovery_state.last_metrics[cache_key] = metrics
                _notify(
                    status_callback,
                    stopped.outcome,
                    f"Cached pointer verification stopped: {stopped.outcome}.",
                    metrics,
                )
                return None
            if cached_is_valid:
                _persist_if_requested(
                    cached,
                    requested=bool(persist),
                    position_config_path=position_config_path,
                    monster_config_path=monster_config_path,
                )
                metrics = builder.freeze(
                    "cache_hit",
                    now=clock(),
                    cache_hit=True,
                )
                with recovery_state.lock:
                    recovery_state.last_metrics[cache_key] = metrics
                _notify(
                    status_callback,
                    "cache_hit",
                    "Verified cached native pointer recovery.",
                    metrics,
                )
                return cached
            with recovery_state.lock:
                if recovery_state.recovery_cache.get(cache_key) is cached:
                    recovery_state.recovery_cache.pop(cache_key, None)
            continue

        if cooldown_until is not None:
            remaining = max(0.0, cooldown_until - now)
            metrics = builder.freeze(
                "negative_cache",
                now=clock(),
                negative_cache_hit=True,
                cooldown_remaining_seconds=remaining,
            )
            with recovery_state.lock:
                recovery_state.last_metrics[cache_key] = metrics
            _notify(
                status_callback,
                "negative_cache",
                f"Recovery cooldown active for {remaining:.3f}s.",
                metrics,
            )
            return None

        assert flight is not None
        if owner:
            break

        _notify(
            status_callback,
            "waiting_for_inflight",
            "Waiting for the in-flight recovery for this process/module.",
            _progress_metrics(builder, control),
        )
        try:
            while not flight.completed.is_set():
                control.check()
                remaining = max(0.0, control.deadline - clock())
                flight.completed.wait(min(0.02, remaining))
        except _AttemptStopped as stopped:
            metrics = builder.freeze(
                stopped.outcome,
                now=clock(),
                joined_existing_attempt=True,
            )
            with recovery_state.lock:
                recovery_state.last_metrics[cache_key] = metrics
            _notify(
                status_callback,
                stopped.outcome,
                f"Stopped waiting for recovery: {stopped.outcome}.",
                metrics,
            )
            return None

        shared = flight.metrics
        if shared is None:
            shared = builder.freeze("shared_unknown", now=clock())
        metrics = replace(
            shared,
            elapsed_seconds=max(0.0, clock() - started_at),
            joined_existing_attempt=True,
        )
        with recovery_state.lock:
            recovery_state.last_metrics[cache_key] = metrics
        _notify(
            status_callback,
            "inflight_complete",
            f"In-flight recovery completed with outcome {metrics.outcome}.",
            metrics,
        )
        return flight.result

    recovery: PlayerPointerRecovery | None = None
    outcome = "not_found"
    try:
        recovery, outcome = _perform_recovery_attempt(
            memory,
            module_base=int(module_base),
            configured_player_pointer_offset=int(
                configured_player_pointer_offset
            ),
            config=config,
            search_radii=search_radii,
            chunk_size=chunk_size,
            maximum_candidates=maximum_candidates,
            stability_samples=stability_samples,
            stability_delay_seconds=stability_delay_seconds,
            metrics=builder,
            control=control,
            status_callback=status_callback,
        )
    except _AttemptStopped as stopped:
        outcome = stopped.outcome
    except Exception:
        outcome = "error"

    completed_at = clock()
    metrics = builder.freeze(outcome, now=completed_at)
    with recovery_state.lock:
        if recovery is not None:
            recovery_state.recovery_cache[cache_key] = recovery
            recovery_state.negative_cache.pop(cache_key, None)
        elif outcome in {"not_found", "deadline"}:
            recovery_state.negative_cache[cache_key] = (
                completed_at + minimum_cooldown
            )
        flight.result = recovery
        flight.metrics = metrics
        recovery_state.last_metrics[cache_key] = metrics
        if recovery_state.inflight.get(cache_key) is flight:
            recovery_state.inflight.pop(cache_key, None)
        persist_for_flight = bool(
            recovery is not None and flight.persist_requested
        )

    if recovery is None:
        flight.completed.set()
        _notify(
            status_callback,
            outcome,
            f"Pointer recovery completed with outcome {outcome}.",
            metrics,
        )
        return None

    _persist_if_requested(
        recovery,
        requested=persist_for_flight,
        position_config_path=position_config_path,
        monster_config_path=monster_config_path,
    )
    flight.completed.set()
    _notify(
        status_callback,
        "success",
        "Native player pointer recovered and strongly validated.",
        metrics,
    )
    print(
        "Native player pointer recovered | "
        f"player=0x{configured_player_pointer_offset:X}->"
        f"0x{recovery.player_pointer_offset:X} "
        + (
            "world=unchanged/unknown "
            if recovery.world_pointer_offset is None
            else (
                f"world=0x{config.world_pointer_offset:X}->"
                f"0x{recovery.world_pointer_offset:X} "
            )
        )
        + (
            f"radius=0x{recovery.search_radius:X} "
            f"candidates={recovery.validated_candidates}"
        )
    )
    return recovery
