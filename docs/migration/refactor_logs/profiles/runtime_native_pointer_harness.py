"""Bounded fake-memory profile for the current native pointer recovery path.

This audit helper deliberately imports the production recovery/provider code,
but it never opens or attaches to a process.  Every successful recovery passes
``persist=False``.  The native JSON files are hashed before and after the run
and a mismatch fails the script.

Run from the repository root:

    .venv\\Scripts\\python.exe refactor_logs\\profiles\\runtime_native_pointer_harness.py
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import platform
import struct
import sys
from dataclasses import asdict, replace
from pathlib import Path
from threading import Barrier, Event, Lock, Thread
from time import perf_counter, sleep
from typing import Callable


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPOSITORY_ROOT / "foreground_vision_bot"
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT))

from foreground_vision_bot.position.MonsterConfig import (  # noqa: E402
    NativeMonsterConfig,
    load_native_monster_config,
)
from foreground_vision_bot.position.NativeFlyffMonsterProvider import (  # noqa: E402
    NativeFlyffMonsterProvider,
)
from foreground_vision_bot.position.NativeFlyffPositionProvider import (  # noqa: E402
    NativeFlyffPositionProvider,
)
from foreground_vision_bot.position.NativePointerRecovery import (  # noqa: E402
    _CACHE_LOCK,
    _RECOVERY_CACHE,
    _contains,
    recover_local_player_pointer,
)
from foreground_vision_bot.position.PositionConfig import (  # noqa: E402
    NativePositionConfig,
    load_native_position_config,
)
from foreground_vision_bot.position.Win32ProcessMemory import (  # noqa: E402
    MemoryRegion,
    MemorySearchDiagnostics,
    ProcessMemoryError,
)
from runtime_bus import RuntimeBus  # noqa: E402
from worker_manager import WorkerKind, WorkerManager  # noqa: E402


MODULE_BASE = 0x10000000
PAGE_READWRITE = 0x04
MEM_PRIVATE = 0x20000
CONFIG_PATHS = (
    PACKAGE_ROOT / "position" / "native_position.json",
    PACKAGE_ROOT / "position" / "native_monsters.json",
)


class FakeProcessMemory:
    """Sparse, counted, thread-safe process memory used only by this profile."""

    def __init__(
        self,
        *,
        pid: int,
        module_base: int = MODULE_BASE,
        segments: tuple[tuple[int, bytearray], ...],
        extra_regions: tuple[MemoryRegion, ...] = (),
        read_delay_seconds: float = 0.0,
    ) -> None:
        self.pid = int(pid)
        self._module_base = int(module_base)
        self._segments = tuple(segments)
        segment_regions = tuple(
            MemoryRegion(
                base_address=base,
                size=len(data),
                protection=PAGE_READWRITE,
                region_type=MEM_PRIVATE,
            )
            for base, data in self._segments
        )
        self._regions = segment_regions + tuple(extra_regions)
        self._read_delay_seconds = max(0.0, float(read_delay_seconds))
        self._lock = Lock()
        self._closed = False
        self.read_calls = 0
        self.bytes_read = 0
        self.read_failures = 0
        self.readable_region_calls = 0
        self.module_base_calls = 0
        self.close_calls = 0
        self.active_reads = 0
        self.peak_active_reads = 0
        self.first_read = Event()
        self.last_search_diagnostics = MemorySearchDiagnostics()

    def module_base(self, _module_name: str) -> int:
        with self._lock:
            self.module_base_calls += 1
            if self._closed:
                raise ProcessMemoryError("fake process memory is closed")
        return self._module_base

    def readable_regions(
        self,
        *,
        maximum_address: int = 0x7FFFFFFF,
        private_only: bool = True,
    ) -> tuple[MemoryRegion, ...]:
        del private_only
        with self._lock:
            self.readable_region_calls += 1
            if self._closed:
                raise ProcessMemoryError("fake process memory is closed")
        return tuple(
            region
            for region in self._regions
            if region.base_address < int(maximum_address)
        )

    def read(self, address: int, size: int) -> bytes:
        address = int(address)
        size = int(size)
        self.first_read.set()
        with self._lock:
            self.read_calls += 1
            self.active_reads += 1
            self.peak_active_reads = max(self.peak_active_reads, self.active_reads)
            closed = self._closed
        try:
            if self._read_delay_seconds:
                sleep(self._read_delay_seconds)
            with self._lock:
                closed = closed or self._closed
            if closed:
                raise ProcessMemoryError("fake process memory is closed")
            for base, data in self._segments:
                relative = address - base
                if relative >= 0 and relative + size <= len(data):
                    result = bytes(data[relative : relative + size])
                    with self._lock:
                        self.bytes_read += len(result)
                    return result
            raise ProcessMemoryError(
                f"fake address range is unreadable: 0x{address:X}+0x{size:X}"
            )
        except Exception:
            with self._lock:
                self.read_failures += 1
            raise
        finally:
            with self._lock:
                self.active_reads -= 1

    def find_u32(
        self,
        _value: int,
        *,
        maximum_address: int,
        private_only: bool,
        chunk_size: int,
    ) -> tuple[int, ...]:
        del maximum_address, private_only, chunk_size
        return ()

    def close(self) -> None:
        with self._lock:
            self.close_calls += 1
            self._closed = True

    def counters(self) -> dict[str, int]:
        with self._lock:
            return {
                "read_calls": self.read_calls,
                "bytes_read": self.bytes_read,
                "read_failures": self.read_failures,
                "readable_region_calls": self.readable_region_calls,
                "module_base_calls": self.module_base_calls,
                "close_calls": self.close_calls,
                "peak_active_reads": self.peak_active_reads,
            }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clear_recovery_cache() -> None:
    with _CACHE_LOCK:
        _RECOVERY_CACHE.clear()


def _time_call(call: Callable[[], object]) -> tuple[float, object]:
    started = perf_counter()
    result = call()
    return (perf_counter() - started) * 1000.0, result


def _blank_module(
    *,
    pid: int,
    size: int,
    read_delay_seconds: float = 0.0,
) -> FakeProcessMemory:
    return FakeProcessMemory(
        pid=pid,
        segments=((MODULE_BASE, bytearray(size)),),
        read_delay_seconds=read_delay_seconds,
    )


def _write_u32(data: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<I", data, offset, int(value))


def _write_i32(data: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<i", data, offset, int(value))


def _write_f32(data: bytearray, offset: int, value: float) -> None:
    struct.pack_into("<f", data, offset, float(value))


def _recover(
    memory: FakeProcessMemory,
    *,
    config: NativeMonsterConfig,
    configured_offset: int,
    search_radii: tuple[int, ...] | None = None,
) -> object:
    kwargs: dict[str, object] = {
        "module_base": MODULE_BASE,
        "configured_player_pointer_offset": configured_offset,
        "monster_config": config,
        "persist": False,
    }
    if search_radii is not None:
        kwargs["search_radii"] = search_radii
    return recover_local_player_pointer(memory, **kwargs)


def profile_default_miss(config: NativeMonsterConfig) -> dict[str, object]:
    """Exercise the production four-radius default on an all-zero module."""

    _clear_recovery_cache()
    size = config.player_pointer_offset + 0x800000 + 4
    memory = _blank_module(pid=91001, size=size)
    elapsed_ms, recovery = _time_call(
        lambda: _recover(
            memory,
            config=config,
            configured_offset=config.player_pointer_offset,
        )
    )
    return {
        "elapsed_ms": round(elapsed_ms, 3),
        "result": None if recovery is None else "unexpected recovery",
        "search_radii_hex": ["0x20000", "0x100000", "0x400000", "0x800000"],
        "module_bytes_allocated": size,
        "counters": memory.counters(),
    }


def profile_provider_attach_construction(
    monster_config: NativeMonsterConfig,
    position_config: NativePositionConfig,
) -> dict[str, object]:
    """Separate provider construction cost from the first live native read."""

    module_size = max(
        int(position_config.pointer_offset or 0),
        monster_config.player_pointer_offset,
        monster_config.world_pointer_offset,
    ) + 4
    memory = _blank_module(pid=91000, size=module_size)
    started = perf_counter()
    position_provider = NativeFlyffPositionProvider(memory, position_config)
    monster_provider = NativeFlyffMonsterProvider(memory, monster_config)
    elapsed_ms = (perf_counter() - started) * 1000.0
    return {
        "elapsed_ms": round(elapsed_ms, 3),
        "position_pointer_address": position_provider.pointer_storage_address,
        "monster_player_pointer_address": monster_provider.player_pointer_address,
        "counters": memory.counters(),
        "note": "constructors resolve module bases but do not dereference pointers",
    }


def profile_position_null(
    monster_config: NativeMonsterConfig,
    position_config: NativePositionConfig,
) -> dict[str, object]:
    """Show the cost inherited by one ordinary module-pointer pose read."""

    _clear_recovery_cache()
    size = int(position_config.pointer_offset or 0) + 0x800000 + 4
    memory = _blank_module(pid=91002, size=size)
    provider = NativeFlyffPositionProvider(memory, position_config)
    started = perf_counter()
    error: Exception | None = None
    try:
        provider.read_pose()
    except Exception as caught:  # Expected null-pointer failure.
        error = caught
    elapsed_ms = (perf_counter() - started) * 1000.0
    assert error is not None
    return {
        "elapsed_ms": round(elapsed_ms, 3),
        "error_type": type(error).__name__,
        "error": str(error),
        "last_pointer_recovery": provider.last_pointer_recovery is not None,
        "counters": memory.counters(),
        "monster_config_offset_matches_position": (
            monster_config.player_pointer_offset == position_config.pointer_offset
        ),
    }


def profile_stale_nonzero(position_config: NativePositionConfig) -> dict[str, object]:
    """A stale nonzero target takes the validation-error path, not recovery."""

    _clear_recovery_cache()
    pointer_offset = int(position_config.pointer_offset or 0)
    actor_base = 0x30000000
    module = bytearray(pointer_offset + 4)
    actor = bytearray(0x400)
    _write_u32(module, pointer_offset, actor_base)
    _write_f32(actor, position_config.x_offset, math.nan)
    _write_f32(actor, position_config.y_offset, 1.0)
    _write_f32(actor, position_config.z_offset, 2.0)
    memory = FakeProcessMemory(
        pid=91003,
        segments=((MODULE_BASE, module), (actor_base, actor)),
    )
    provider = NativeFlyffPositionProvider(memory, position_config)
    started = perf_counter()
    error: Exception | None = None
    try:
        provider.read_pose()
    except Exception as caught:
        error = caught
    elapsed_ms = (perf_counter() - started) * 1000.0
    assert error is not None
    return {
        "elapsed_ms": round(elapsed_ms, 3),
        "error_type": type(error).__name__,
        "error": str(error),
        "last_pointer_recovery": provider.last_pointer_recovery is not None,
        "counters": memory.counters(),
    }


def profile_repeated_misses(config: NativeMonsterConfig) -> dict[str, object]:
    """Confirm misses are not cached and every ordinary retry scans again."""

    _clear_recovery_cache()
    player_offset = 0x80000
    world_offset = 0x90000
    local_config = replace(
        config,
        player_pointer_offset=player_offset,
        world_pointer_offset=world_offset,
    )
    memory = _blank_module(pid=91004, size=0x120000)
    attempts: list[dict[str, object]] = []
    previous = memory.counters()
    for index in range(3):
        elapsed_ms, recovery = _time_call(
            lambda: _recover(
                memory,
                config=local_config,
                configured_offset=player_offset,
                search_radii=(0x20000,),
            )
        )
        current = memory.counters()
        delta = {key: current[key] - previous[key] for key in current}
        previous = current
        attempts.append(
            {
                "attempt": index + 1,
                "elapsed_ms": round(elapsed_ms, 3),
                "result": None if recovery is None else "unexpected recovery",
                "counter_delta": delta,
            }
        )
    with _CACHE_LOCK:
        cache_entries = len(_RECOVERY_CACHE)
    return {
        "attempts": attempts,
        "total_counters": memory.counters(),
        "cache_entries_after_failures": cache_entries,
    }


def profile_valid_shift_and_cache(config: NativeMonsterConfig) -> dict[str, object]:
    """Recover one shifted paired player/world global and then verify its cache."""

    _clear_recovery_cache()
    old_player_offset = 0x80000
    old_world_offset = 0x90000
    shift = 0x4000
    new_player_offset = old_player_offset + shift
    new_world_offset = old_world_offset + shift
    actor_base = 0x31000000
    world_base = 0x32000000
    module = bytearray(0x120000)
    actor = bytearray(0x3000)
    _write_u32(module, new_player_offset, actor_base)
    _write_u32(module, new_world_offset, world_base)
    _write_u32(actor, config.self_pointer_offset, actor_base)
    _write_u32(actor, config.world_offset, world_base)
    _write_f32(actor, config.x_offset, 10.0)
    _write_f32(actor, config.y_offset, 20.0)
    _write_f32(actor, config.z_offset, 30.0)
    _write_i32(actor, config.species_offset, 0)
    _write_i32(actor, config.active_species_offset, 0)
    _write_i32(actor, config.hp_offset, 100)
    local_config = replace(
        config,
        player_pointer_offset=old_player_offset,
        world_pointer_offset=old_world_offset,
    )
    memory = FakeProcessMemory(
        pid=91005,
        segments=((MODULE_BASE, module), (actor_base, actor)),
    )
    captured_stdout = io.StringIO()
    with contextlib.redirect_stdout(captured_stdout):
        first_ms, first = _time_call(
            lambda: _recover(
                memory,
                config=local_config,
                configured_offset=old_player_offset,
                search_radii=(0x10000,),
            )
        )
        counters_after_first = memory.counters()
        second_ms, second = _time_call(
            lambda: _recover(
                memory,
                config=local_config,
                configured_offset=old_player_offset,
                search_radii=(0x10000,),
            )
        )
    assert first is not None and second is first
    second_delta = {
        key: memory.counters()[key] - counters_after_first[key]
        for key in memory.counters()
    }
    return {
        "first_elapsed_ms": round(first_ms, 3),
        "cache_hit_elapsed_ms": round(second_ms, 3),
        "recovery": asdict(first),
        "first_counters": counters_after_first,
        "cache_hit_counter_delta": second_delta,
        "stdout": captured_stdout.getvalue().strip().splitlines(),
        "persist_was_enabled": False,
    }


def profile_candidate_pressure(config: NativeMonsterConfig) -> dict[str, object]:
    """Measure linear region containment when scan words look pointer-like."""

    _clear_recovery_cache()
    player_offset = 0x40000
    world_offset = 0x50000
    module = bytearray(0x80000)
    decoy_count = 256
    decoy_regions = tuple(
        MemoryRegion(
            base_address=0x40000000 + index * 0x4000,
            size=0x3000,
            protection=PAGE_READWRITE,
            region_type=MEM_PRIVATE,
        )
        for index in range(decoy_count)
    )
    target = decoy_regions[-1].base_address
    actor = bytearray(0x3000)  # Deliberately fails the self-pointer validation.
    scan_start = player_offset - 0x20000
    scan_stop = player_offset + 0x20000
    planted = 0
    for offset in range(scan_start, scan_stop, 0x100):
        _write_u32(module, offset, target)
        planted += 1
    local_config = replace(
        config,
        player_pointer_offset=player_offset,
        world_pointer_offset=world_offset,
    )
    memory = FakeProcessMemory(
        pid=91006,
        segments=((MODULE_BASE, module), (target, actor)),
        extra_regions=decoy_regions[:-1],
    )
    elapsed_ms, recovery = _time_call(
        lambda: _recover(
            memory,
            config=local_config,
            configured_offset=player_offset,
            search_radii=(0x20000,),
        )
    )
    return {
        "elapsed_ms": round(elapsed_ms, 3),
        "result": None if recovery is None else "unexpected recovery",
        "planted_pointer_like_words": planted,
        "readable_region_count": len(memory._regions),
        "target_region_position": "last",
        "counters": memory.counters(),
    }


def profile_contains_scaling() -> dict[str, object]:
    """Microprofile the production tuple-linear ``_contains`` helper."""

    calls = 3000
    result: dict[str, object] = {"calls_per_size": calls, "measurements": []}
    measurements: list[dict[str, object]] = []
    for region_count in (8, 128, 512):
        regions = tuple(
            MemoryRegion(
                base_address=0x50000000 + index * 0x4000,
                size=0x3000,
                protection=PAGE_READWRITE,
                region_type=MEM_PRIVATE,
            )
            for index in range(region_count)
        )
        target = regions[-1].base_address
        started = perf_counter()
        hits = sum(1 for _ in range(calls) if _contains(regions, target, 0x1EE4))
        elapsed_ms = (perf_counter() - started) * 1000.0
        measurements.append(
            {
                "regions": region_count,
                "elapsed_ms": round(elapsed_ms, 3),
                "microseconds_per_lookup": round(elapsed_ms * 1000.0 / calls, 3),
                "hits": hits,
            }
        )
    result["measurements"] = measurements
    return result


def profile_concurrent_misses(config: NativeMonsterConfig) -> dict[str, object]:
    """Launch simultaneous position-like and monster-like resolver requests."""

    _clear_recovery_cache()
    player_offset = 0xC0000
    world_offset = 0xD0000
    local_config = replace(
        config,
        player_pointer_offset=player_offset,
        world_pointer_offset=world_offset,
    )
    memory = _blank_module(
        pid=91007,
        size=0x180000,
        read_delay_seconds=0.001,
    )
    barrier = Barrier(3)
    timings: dict[str, float] = {}
    results: dict[str, object] = {}
    result_lock = Lock()

    def caller(label: str) -> None:
        barrier.wait()
        elapsed_ms, recovery = _time_call(
            lambda: _recover(
                memory,
                config=local_config,
                configured_offset=player_offset,
                search_radii=(0x40000,),
            )
        )
        with result_lock:
            timings[label] = round(elapsed_ms, 3)
            results[label] = None if recovery is None else "unexpected recovery"

    threads = [
        Thread(target=caller, args=("position_like",), daemon=False),
        Thread(target=caller, args=("monster_like",), daemon=False),
    ]
    for thread in threads:
        thread.start()
    wall_started = perf_counter()
    barrier.wait()
    for thread in threads:
        thread.join(10.0)
    wall_ms = (perf_counter() - wall_started) * 1000.0
    assert not any(thread.is_alive() for thread in threads)
    return {
        "wall_elapsed_ms": round(wall_ms, 3),
        "caller_elapsed_ms": timings,
        "results": results,
        "counters": memory.counters(),
        "expected_readable_region_calls_if_single_flight": 1,
        "actual_readable_region_calls": memory.readable_region_calls,
    }


def profile_worker_cancellation(config: NativeMonsterConfig) -> dict[str, object]:
    """Show WorkerManager cancellation cannot interrupt the recovery function."""

    _clear_recovery_cache()
    player_offset = 0x100000
    world_offset = 0x110000
    local_config = replace(
        config,
        player_pointer_offset=player_offset,
        world_pointer_offset=world_offset,
    )
    memory = _blank_module(
        pid=91008,
        size=0x200000,
        read_delay_seconds=0.004,
    )
    bus = RuntimeBus(max_logs=20)
    manager = WorkerManager(bus)
    observed: dict[str, object] = {}

    def target(token: object) -> None:
        observed["token_cancelled_before_recovery"] = bool(
            getattr(token, "cancelled")
        )
        _recover(
            memory,
            config=local_config,
            configured_offset=player_offset,
            search_radii=(0x80000,),
        )
        observed["token_cancelled_at_return"] = bool(getattr(token, "cancelled"))

    manager.start(
        name="audit-preview-native-recovery",
        kind=WorkerKind.PREVIEW,
        target=target,
    )
    assert memory.first_read.wait(2.0)
    cancellation_started = perf_counter()
    joined_with_short_budget = manager.stop_and_join(
        WorkerKind.PREVIEW,
        timeout=0.015,
    )
    short_return_ms = (perf_counter() - cancellation_started) * 1000.0
    snapshot_after_short_join = manager.snapshot(WorkerKind.PREVIEW)
    final_joined = manager.join(WorkerKind.PREVIEW, timeout=10.0)
    cancellation_to_finish_ms = (perf_counter() - cancellation_started) * 1000.0
    final_snapshot = manager.snapshot(WorkerKind.PREVIEW)
    assert final_joined
    bus.close()
    return {
        "short_join_budget_ms": 15.0,
        "short_join_returned_in_ms": round(short_return_ms, 3),
        "short_join_succeeded": joined_with_short_budget,
        "alive_after_short_join": bool(
            snapshot_after_short_join and snapshot_after_short_join.alive
        ),
        "state_after_short_join": (
            None
            if snapshot_after_short_join is None
            else snapshot_after_short_join.state.value
        ),
        "cancellation_to_worker_finish_ms": round(cancellation_to_finish_ms, 3),
        "final_join_succeeded": final_joined,
        "final_state": (
            None if final_snapshot is None else final_snapshot.state.value
        ),
        "observed": observed,
        "counters": memory.counters(),
        "recovery_accepts_cancellation_token": False,
    }


def profile_closed_process(
    monster_config: NativeMonsterConfig,
    position_config: NativePositionConfig,
) -> dict[str, object]:
    """Compare position and monster provider errors after the fake process exits."""

    pointer_offset = int(position_config.pointer_offset or 0)
    module_size = max(
        pointer_offset,
        monster_config.player_pointer_offset,
        monster_config.world_pointer_offset,
    ) + 4

    position_memory = _blank_module(pid=91009, size=module_size)
    position_provider = NativeFlyffPositionProvider(
        position_memory,
        position_config,
    )
    position_memory.close()
    position_error: Exception | None = None
    position_started = perf_counter()
    try:
        position_provider.read_pose()
    except Exception as caught:
        position_error = caught
    position_ms = (perf_counter() - position_started) * 1000.0

    monster_memory = _blank_module(pid=91010, size=module_size)
    monster_provider = NativeFlyffMonsterProvider(
        monster_memory,
        monster_config,
    )
    monster_memory.close()
    monster_error: Exception | None = None
    monster_started = perf_counter()
    try:
        monster_provider.read_active_actors()
    except Exception as caught:
        monster_error = caught
    monster_ms = (perf_counter() - monster_started) * 1000.0

    assert position_error is not None and monster_error is not None
    return {
        "position": {
            "elapsed_ms": round(position_ms, 3),
            "error_type": type(position_error).__name__,
            "error": str(position_error),
            "counters": position_memory.counters(),
        },
        "monster": {
            "elapsed_ms": round(monster_ms, 3),
            "error_type": type(monster_error).__name__,
            "error": str(monster_error),
            "counters": monster_memory.counters(),
        },
    }


def main() -> int:
    config_hashes_before = {str(path): _sha256(path) for path in CONFIG_PATHS}
    monster_config = load_native_monster_config()
    position_config = load_native_position_config()
    if (
        not position_config.enabled
        or position_config.resolver != "module_pointer"
        or position_config.pointer_offset is None
    ):
        raise RuntimeError("audit expects the checked-in module-pointer config")

    suite_started = perf_counter()
    results = {
        "audit_constraints": {
            "backend": "FakeProcessMemory only",
            "real_process_opened": False,
            "successful_recovery_persist": False,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "provider_attach_construction": profile_provider_attach_construction(
            monster_config,
            position_config,
        ),
        "default_failed_recovery": profile_default_miss(monster_config),
        "ordinary_position_null_read": profile_position_null(
            monster_config,
            position_config,
        ),
        "stale_nonzero_position_read": profile_stale_nonzero(position_config),
        "repeated_failed_recovery": profile_repeated_misses(monster_config),
        "valid_shift_and_success_cache": profile_valid_shift_and_cache(
            monster_config
        ),
        "candidate_pressure": profile_candidate_pressure(monster_config),
        "region_containment_scaling": profile_contains_scaling(),
        "concurrent_position_monster_like_misses": profile_concurrent_misses(
            monster_config
        ),
        "worker_cancellation": profile_worker_cancellation(monster_config),
        "closed_process": profile_closed_process(
            monster_config,
            position_config,
        ),
    }
    results["suite_elapsed_ms"] = round(
        (perf_counter() - suite_started) * 1000.0,
        3,
    )
    config_hashes_after = {str(path): _sha256(path) for path in CONFIG_PATHS}
    results["config_integrity"] = {
        "before": config_hashes_before,
        "after": config_hashes_after,
        "unchanged": config_hashes_before == config_hashes_after,
    }
    if config_hashes_before != config_hashes_after:
        raise RuntimeError("native config changed during fake recovery profiling")

    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
