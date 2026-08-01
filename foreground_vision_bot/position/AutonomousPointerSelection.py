from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from collections.abc import Callable, Mapping

from .AnchoredPointerDiscovery import AnchoredPlayerObservation
from .PointerScanWorkflow import (
    PointerScanSnapshot,
    PointerWorkflowMemory,
    read_player_observation,
)
from .Win32ProcessMemory import MemoryRegion, ModuleInfo

# Writable Win32 page protections. The low byte contains the base protection;
# modifiers such as PAGE_NOCACHE may be ORed into the full value.
_PAGE_WRITABLE = frozenset((0x04, 0x08, 0x40, 0x80))


@dataclass(frozen=True, slots=True)
class DirectPlayerSlotEvidence:
    slot_address: int
    module_offset: int
    stable_samples: int
    total_samples: int
    writable: bool
    historical_sessions: int
    configured_distance: int

    @property
    def stable(self) -> bool:
        return self.total_samples > 0 and self.stable_samples == self.total_samples


@dataclass(frozen=True, slots=True)
class PassivePlayerProof:
    accepted: bool
    player_base: int
    structural_samples: int
    total_samples: int
    coordinate_change_native: float
    hp_changes: int
    selected_slot: DirectPlayerSlotEvidence | None
    fallback_slots: tuple[DirectPlayerSlotEvidence, ...]
    message: str


def _region_for(address: int, regions: tuple[object, ...]) -> object | None:
    for region in regions:
        start = int(getattr(region, "base_address"))
        stop = start + int(getattr(region, "size"))
        if start <= address < stop:
            return region
    return None


def _is_writable(region: object | None) -> bool:
    if region is None:
        return False
    protection = int(getattr(region, "protection", 0))
    return (protection & 0xFF) in _PAGE_WRITABLE


def _read_player_like(
    memory: PointerWorkflowMemory,
    observation: AnchoredPlayerObservation,
    *,
    maximum_absolute_coordinate: float,
) -> tuple[bool, float, float, int]:
    x, _y, z, hp, maximum_hp, self_pointer = read_player_observation(
        memory,
        observation,
    )
    if self_pointer != observation.player_base:
        return False, x, z, hp
    if not math.isfinite(x) or not math.isfinite(z):
        return False, x, z, hp
    if abs(x) > maximum_absolute_coordinate or abs(z) > maximum_absolute_coordinate:
        return False, x, z, hp
    if hp <= 0 or maximum_hp <= 0:
        return False, x, z, hp
    if observation.max_hp_offset != observation.hp_offset and hp > maximum_hp:
        return False, x, z, hp

    # The monster cohort taught us that active monsters duplicate species in a
    # second field. The local player should not satisfy that actor predicate.
    try:
        span = max(observation.species_offset, observation.active_species_offset) + 4
        data = memory.read(observation.player_base, span)
        species = int.from_bytes(
            data[observation.species_offset : observation.species_offset + 4],
            "little",
            signed=True,
        )
        active = int.from_bytes(
            data[
                observation.active_species_offset : observation.active_species_offset
                + 4
            ],
            "little",
            signed=True,
        )
    except Exception:
        return False, x, z, hp
    if species > 0 and active == species:
        return False, x, z, hp
    return True, x, z, hp


def prove_player_and_rank_direct_slots(
    memory: PointerWorkflowMemory,
    observation: AnchoredPlayerObservation,
    module: ModuleInfo,
    regions: tuple[object, ...],
    *,
    configured_player_offset: int,
    historical_sessions_by_offset: Mapping[int, int] | None = None,
    samples: int = 24,
    interval_seconds: float = 0.10,
    maximum_absolute_coordinate: float = 1_000_000.0,
    check: Callable[[], None] | None = None,
) -> PassivePlayerProof:
    """Passively prove one player object and rank direct module aliases.

    No movement or combat transition is required. Coordinates and HP are
    allowed to change; the proof requires only that the exact object preserves
    its self identity, player-like actor semantics, plausible pose, and valid HP
    relationship while module slots continue to resolve to it.
    """

    sample_count = max(3, int(samples))
    history = historical_sessions_by_offset or {}
    module_start = int(module.base_address)
    module_stop = module_start + int(module.size)
    slots = tuple(
        sorted(
            set(
                int(slot)
                for slot in observation.direct_module_slots
                if module_start <= int(slot) < module_stop
            )
        )
    )
    stable_counts = {slot: 0 for slot in slots}
    structural = 0
    baseline_x = observation.x
    baseline_z = observation.z
    greatest_distance = 0.0
    previous_hp: int | None = None
    hp_changes = 0

    for index in range(sample_count):
        if check is not None:
            check()
        try:
            valid, x, z, hp = _read_player_like(
                memory,
                observation,
                maximum_absolute_coordinate=maximum_absolute_coordinate,
            )
        except Exception:
            valid, x, z, hp = False, baseline_x, baseline_z, 0
        if valid:
            structural += 1
            greatest_distance = max(
                greatest_distance,
                math.hypot(x - baseline_x, z - baseline_z),
            )
            if previous_hp is not None and hp != previous_hp:
                hp_changes += 1
            previous_hp = hp
        for slot in slots:
            try:
                value = int.from_bytes(memory.read(slot, 4), "little")
            except Exception:
                continue
            if value == observation.player_base:
                stable_counts[slot] += 1
        if index + 1 < sample_count:
            sleep(max(0.0, float(interval_seconds)))

    evidence = tuple(
        DirectPlayerSlotEvidence(
            slot_address=slot,
            module_offset=slot - module_start,
            stable_samples=stable_counts[slot],
            total_samples=sample_count,
            writable=_is_writable(_region_for(slot, regions)),
            historical_sessions=max(0, int(history.get(slot - module_start, 0))),
            configured_distance=abs(
                (slot - module_start) - int(configured_player_offset)
            ),
        )
        for slot in slots
    )
    stable = tuple(item for item in evidence if item.stable)
    writable_stable = tuple(item for item in stable if item.writable)
    eligible = writable_stable or stable
    ranked = tuple(
        sorted(
            eligible,
            key=lambda item: (
                -item.historical_sessions,
                int(not item.writable),
                item.configured_distance,
                item.module_offset,
            ),
        )
    )
    selected = ranked[0] if ranked else None
    accepted = structural == sample_count and selected is not None
    if accepted:
        movement_note = (
            f"; natural coordinate change={greatest_distance:.3f}"
            if greatest_distance > 0.0
            else ""
        )
        hp_note = f"; HP changes={hp_changes}" if hp_changes else ""
        message = (
            f"Passively proved one player object for {sample_count}/{sample_count} "
            f"samples and selected {module.name}+0x{selected.module_offset:X} "
            f"from {len(stable)} stable direct alias(es){movement_note}{hp_note}."
        )
    elif structural != sample_count:
        message = (
            f"Player structure passed {structural}/{sample_count} passive samples; "
            "no pointer is accepted."
        )
    else:
        message = (
            "The player object was passively stable, but no direct module slot "
            "remained pointed at it for every sample."
        )
    return PassivePlayerProof(
        accepted=accepted,
        player_base=observation.player_base,
        structural_samples=structural,
        total_samples=sample_count,
        coordinate_change_native=greatest_distance,
        hp_changes=hp_changes,
        selected_slot=selected,
        fallback_slots=ranked[1:],
        message=message,
    )


def historical_direct_offset_counts(
    snapshots: tuple[object, ...],
) -> dict[int, int]:
    counts: dict[int, int] = {}
    for snapshot in snapshots:
        seen = {
            int(value)
            for value in getattr(snapshot, "direct_module_offsets", ())
        }
        for offset in seen:
            counts[offset] = counts.get(offset, 0) + 1
    return counts


def load_matching_snapshot_history(
    directory: str | Path,
    module: ModuleInfo,
) -> tuple[PointerScanSnapshot, ...]:
    from pathlib import Path

    snapshots: list[PointerScanSnapshot] = []
    wanted_path = Path(module.path).as_posix().casefold() if module.path else ""
    for path in sorted(Path(directory).glob("pointer_scan_*.json")):
        try:
            candidate = PointerScanSnapshot.from_path(path)
        except Exception:
            continue
        candidate_path = (
            Path(candidate.module_path).as_posix().casefold()
            if candidate.module_path
            else ""
        )
        same_path = not wanted_path or not candidate_path or candidate_path == wanted_path
        if (
            candidate.module_name.casefold() == module.name.casefold()
            and candidate.module_size == module.size
            and same_path
        ):
            snapshots.append(candidate)
    return tuple(snapshots)
