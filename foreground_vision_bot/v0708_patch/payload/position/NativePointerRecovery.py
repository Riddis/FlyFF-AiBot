from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from time import sleep
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


_CACHE_LOCK = RLock()
_RECOVERY_CACHE: dict[tuple[int, int], PlayerPointerRecovery] = {}
_PERSIST_LOCK = RLock()


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


def _contains(regions: tuple[object, ...], address: int, size: int = 1) -> bool:
    end = int(address) + max(1, int(size))
    for region in regions:
        start, stop = _region_bounds(region)
        if start <= address and end <= stop:
            return True
    return False


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
) -> bool:
    for index in range(max(1, int(samples))):
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
        if index + 1 < samples and delay_seconds > 0.0:
            sleep(delay_seconds)
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


def recover_local_player_pointer(
    memory: PointerRecoveryMemory,
    *,
    module_base: int,
    configured_player_pointer_offset: int,
    monster_config: NativeMonsterConfig | None = None,
    search_radii: tuple[int, ...] = (0x20000, 0x100000, 0x400000, 0x800000),
    chunk_size: int = 0x10000,
    maximum_candidates: int = 4096,
    persist: bool = True,
) -> PlayerPointerRecovery | None:
    """Recover a shifted Neuz local-player global after a client update.

    The scan is deliberately restricted to module memory near the previously
    configured slot. A candidate must point to a self-valid actor with finite
    coordinates and positive HP. High confidence comes from either the old
    world pointer still matching the actor or the player/world globals having
    shifted by the same module-relative delta.
    """

    readable_regions_fn = getattr(memory, "readable_regions", None)
    if not callable(readable_regions_fn):
        return None

    config = monster_config or load_native_monster_config()
    pid = int(getattr(memory, "pid", 0))
    cache_key = (pid, int(module_base))
    with _CACHE_LOCK:
        cached = _RECOVERY_CACHE.get(cache_key)
    if cached is not None and _verify_cached(memory, cached, config):
        return cached

    configured_slot = int(module_base) + int(configured_player_pointer_offset)
    configured_world_hint = _read_configured_world_hint(memory, module_base, config)
    max_actor_span = max(
        config.self_pointer_offset,
        config.active_species_offset,
        config.hp_offset,
        config.z_offset,
    ) + 4

    for radius in tuple(sorted({max(0x1000, int(value)) for value in search_radii})):
        scan_start = max(int(module_base), configured_slot - radius)
        scan_end = configured_slot + radius + 4
        try:
            regions = tuple(
                readable_regions_fn(
                    maximum_address=config.maximum_scan_address,
                    private_only=False,
                )
            )
        except Exception:
            return None

        candidate_regions = tuple(
            region
            for region in regions
            if _region_bounds(region)[1] > scan_start
            and _region_bounds(region)[0] < scan_end
        )
        all_regions = regions
        slot_refs: dict[int, list[int]] = {}

        for region in candidate_regions:
            region_start, region_end = _region_bounds(region)
            cursor = max(scan_start, region_start)
            stop = min(scan_end, region_end)
            while cursor < stop:
                amount = min(max(0x1000, int(chunk_size)), stop - cursor)
                try:
                    data = memory.read(cursor, amount)
                except Exception:
                    cursor += amount
                    continue

                first = (-cursor) % 4
                for relative in range(first, len(data) - 3, 4):
                    target = int(struct.unpack_from("<I", data, relative)[0])
                    if target <= 0x10000 or target % 4 != 0:
                        continue
                    if not _contains(all_regions, target, max_actor_span):
                        continue
                    slot = cursor + relative
                    refs = slot_refs.setdefault(target, [])
                    if len(refs) < 8:
                        refs.append(slot)
                cursor += amount

        ordered_slots: list[tuple[int, int, int]] = []
        for target, slots in slot_refs.items():
            reference_count = len(slots)
            for slot in slots:
                ordered_slots.append(
                    (abs(slot - configured_slot), slot, target)
                )
        ordered_slots.sort(key=lambda item: item[0])

        validated: list[_ValidatedCandidate] = []
        seen_pairs: set[tuple[int, int]] = set()
        for _distance, slot, target in ordered_slots[: max(1, maximum_candidates)]:
            key = (slot, target)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
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
        if not strong:
            continue

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
        selected = strong[0]
        if not _stable_candidate(memory, selected, config):
            continue

        player_offset = selected.slot_address - int(module_base)
        world_slot = selected.paired_world_slot
        world_offset = (
            None if world_slot is None else world_slot - int(module_base)
        )
        recovery = PlayerPointerRecovery(
            player_pointer_address=selected.slot_address,
            player_pointer_offset=player_offset,
            player_base=selected.target_address,
            world_base=selected.world_base,
            world_pointer_address=world_slot,
            world_pointer_offset=world_offset,
            configured_player_pointer_offset=int(configured_player_pointer_offset),
            configured_world_pointer_offset=int(config.world_pointer_offset),
            search_radius=radius,
            validated_candidates=len(validated),
        )
        with _CACHE_LOCK:
            _RECOVERY_CACHE[cache_key] = recovery
        if persist:
            try:
                persist_recovered_pointer_offsets(recovery)
            except Exception as error:
                print(
                    "Native pointer recovery succeeded but config persistence "
                    f"failed: {type(error).__name__}: {error}"
                )
        print(
            "Native player pointer recovered | "
            f"player=0x{configured_player_pointer_offset:X}->"
            f"0x{player_offset:X} "
            + (
                "world=unchanged/unknown "
                if world_offset is None
                else f"world=0x{config.world_pointer_offset:X}->0x{world_offset:X} "
            )
            + f"radius=0x{radius:X} candidates={len(validated)}"
        )
        return recovery

    return None
