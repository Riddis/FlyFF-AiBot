from __future__ import annotations

import json
import math
import struct
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Event, RLock
from time import monotonic, sleep
from typing import Protocol

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


_CACHE_LOCK = RLock()
_RECOVERY_CACHE: dict[tuple[int, int], PlayerPointerRecovery] = {}
_NEGATIVE_CACHE: dict[tuple[int, int], float] = {}
_INFLIGHT: dict[tuple[int, int], _RecoveryFlight] = {}
_LAST_METRICS: dict[tuple[int, int], PointerRecoveryMetrics] = {}
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
    pid: int | None = None,
    module_base: int | None = None,
) -> None:
    """Invalidate cached recovery state, primarily on detach/reattach or in tests."""

    with _CACHE_LOCK:
        if pid is None and module_base is None:
            _RECOVERY_CACHE.clear()
            _NEGATIVE_CACHE.clear()
            _LAST_METRICS.clear()
            return
        keys = set(_RECOVERY_CACHE) | set(_NEGATIVE_CACHE) | set(_LAST_METRICS)
        for key in keys:
            if pid is not None and key[0] != int(pid):
                continue
            if module_base is not None and key[1] != int(module_base):
                continue
            _RECOVERY_CACHE.pop(key, None)
            _NEGATIVE_CACHE.pop(key, None)
            _LAST_METRICS.pop(key, None)


def get_last_pointer_recovery_metrics(
    pid: int,
    module_base: int,
) -> PointerRecoveryMetrics | None:
    with _CACHE_LOCK:
        return _LAST_METRICS.get((int(pid), int(module_base)))


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


def _atomic_json_update(path: Path, updates: dict[str, str]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration root must be an object: {path}")
    changed = False
    for key, value in updates.items():
        if payload.get(key) != value:
            payload[key] = value
            changed = True
    if not changed:
        return

    backup = path.with_suffix(path.suffix + ".pre_pointer_recovery.bak")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def persist_recovered_pointer_offsets(
    recovery: PlayerPointerRecovery,
    *,
    position_config_path: str | Path = DEFAULT_POSITION_CONFIG_PATH,
    monster_config_path: str | Path = DEFAULT_MONSTER_CONFIG_PATH,
) -> None:
    with _PERSIST_LOCK:
        _atomic_json_update(
            Path(position_config_path),
            {"pointer_offset": f"0x{recovery.player_pointer_offset:X}"},
        )
        monster_updates = {
            "player_pointer_offset": f"0x{recovery.player_pointer_offset:X}",
        }
        if recovery.world_pointer_offset is not None:
            monster_updates["world_pointer_offset"] = (
                f"0x{recovery.world_pointer_offset:X}"
            )
        _atomic_json_update(Path(monster_config_path), monster_updates)


def _persist_if_requested(
    recovery: PlayerPointerRecovery,
    *,
    requested: bool,
) -> None:
    if not requested:
        return
    try:
        persist_recovered_pointer_offsets(recovery)
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
    readable_regions_fn = getattr(memory, "readable_regions", None)
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
    monster_config: NativeMonsterConfig | None = None,
    search_radii: tuple[int, ...] = (0x20000, 0x100000, 0x400000, 0x800000),
    chunk_size: int = 0x10000,
    maximum_candidates: int = 4096,
    persist: bool = False,
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
    attempts are single-flight per ``(pid, module_base)``, enumerate and index
    readable regions once, scan each expanding memory band at most once, and
    stop cooperatively at the deadline or cancellation boundary. Failed
    complete attempts enter a process-scoped cooldown to prevent retry storms.

    Deadline checks surround each synchronous backend call and run throughout
    CPU scanning. Python cannot preempt a backend call already in progress, so
    ``PointerRecoveryMetrics.deadline_is_cooperative`` is explicit and callers
    must keep this diagnostic behind a bounded lifecycle owner.
    """

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
            with _CACHE_LOCK:
                _LAST_METRICS[cache_key] = metrics
            _notify(
                status_callback,
                stopped.outcome,
                f"Pointer recovery stopped: {stopped.outcome}.",
                metrics,
            )
            return None

        now = clock()
        with _CACHE_LOCK:
            cached = _RECOVERY_CACHE.get(cache_key)
            cooldown_until = _NEGATIVE_CACHE.get(cache_key)
            flight = _INFLIGHT.get(cache_key)
            if cooldown_until is not None and cooldown_until <= now:
                _NEGATIVE_CACHE.pop(cache_key, None)
                cooldown_until = None
            if cached is None and cooldown_until is None and flight is None:
                flight = _RecoveryFlight(persist_requested=bool(persist))
                _INFLIGHT[cache_key] = flight
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
                with _CACHE_LOCK:
                    _LAST_METRICS[cache_key] = metrics
                _notify(
                    status_callback,
                    stopped.outcome,
                    f"Cached pointer verification stopped: {stopped.outcome}.",
                    metrics,
                )
                return None
            if cached_is_valid:
                _persist_if_requested(cached, requested=bool(persist))
                metrics = builder.freeze(
                    "cache_hit",
                    now=clock(),
                    cache_hit=True,
                )
                with _CACHE_LOCK:
                    _LAST_METRICS[cache_key] = metrics
                _notify(
                    status_callback,
                    "cache_hit",
                    "Verified cached native pointer recovery.",
                    metrics,
                )
                return cached
            with _CACHE_LOCK:
                if _RECOVERY_CACHE.get(cache_key) is cached:
                    _RECOVERY_CACHE.pop(cache_key, None)
            continue

        if cooldown_until is not None:
            remaining = max(0.0, cooldown_until - now)
            metrics = builder.freeze(
                "negative_cache",
                now=clock(),
                negative_cache_hit=True,
                cooldown_remaining_seconds=remaining,
            )
            with _CACHE_LOCK:
                _LAST_METRICS[cache_key] = metrics
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
            with _CACHE_LOCK:
                _LAST_METRICS[cache_key] = metrics
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
        with _CACHE_LOCK:
            _LAST_METRICS[cache_key] = metrics
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
    with _CACHE_LOCK:
        if recovery is not None:
            _RECOVERY_CACHE[cache_key] = recovery
            _NEGATIVE_CACHE.pop(cache_key, None)
        elif outcome in {"not_found", "deadline"}:
            _NEGATIVE_CACHE[cache_key] = completed_at + minimum_cooldown
        flight.result = recovery
        flight.metrics = metrics
        _LAST_METRICS[cache_key] = metrics
        if _INFLIGHT.get(cache_key) is flight:
            _INFLIGHT.pop(cache_key, None)
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

    _persist_if_requested(recovery, requested=persist_for_flight)
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
