from __future__ import annotations

import json
import math
import struct
from bisect import bisect_right
from collections import deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep
from collections.abc import Callable
from typing import Protocol

from .AnchoredPointerDiscovery import AnchoredPlayerObservation
from .Win32ProcessMemory import ModuleInfo

SNAPSHOT_SCHEMA_VERSION = 1


class PointerWorkflowMemory(Protocol):
    pid: int

    def read(self, address: int, size: int) -> bytes: ...


@dataclass(frozen=True, slots=True)
class ReadableRegionIndex:
    starts: tuple[int, ...]
    stops: tuple[int, ...]

    @classmethod
    def build(cls, regions: tuple[object, ...]) -> ReadableRegionIndex:
        bounds = sorted(
            (
                int(getattr(region, "base_address")),
                int(getattr(region, "base_address"))
                + max(0, int(getattr(region, "size"))),
            )
            for region in regions
            if int(getattr(region, "size")) > 0
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
        stop = start + max(1, int(size))
        index = bisect_right(self.starts, start) - 1
        return index >= 0 and stop <= self.stops[index]


@dataclass(frozen=True, slots=True)
class PointerPath:
    """One module-rooted path resolving to the live player object.

    ``root_module_offset`` identifies the DWORD slot in the module image.
    ``field_offsets`` are dereferenced in order after reading that slot. An
    empty tuple is a direct module global.
    """

    root_module_offset: int
    field_offsets: tuple[int, ...]

    @property
    def depth(self) -> int:
        return len(self.field_offsets)

    @property
    def signature(self) -> str:
        fields = ",".join(f"0x{value:X}" for value in self.field_offsets)
        return f"0x{self.root_module_offset:X}|{fields}"


@dataclass(frozen=True, slots=True)
class MovementEvidence:
    confirmed: bool
    distance_native: float
    final_x: float
    final_y: float
    final_z: float
    samples: int
    message: str


@dataclass(frozen=True, slots=True)
class PointerScanSnapshot:
    schema_version: int
    captured_at_utc: str
    pid: int
    module_name: str
    module_path: str
    module_size: int
    module_base: int
    player_base: int
    player_fields: dict[str, int]
    baseline: dict[str, float | int]
    movement: dict[str, object]
    direct_module_offsets: tuple[int, ...]
    pointer_paths: tuple[dict[str, object], ...]
    anchor_outcome: str
    monster_hp_anchors: tuple[tuple[int, int], ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_path(cls, path: str | Path) -> PointerScanSnapshot:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            schema_version=int(payload["schema_version"]),
            captured_at_utc=str(payload["captured_at_utc"]),
            pid=int(payload["pid"]),
            module_name=str(payload["module_name"]),
            module_path=str(payload.get("module_path", "")),
            module_size=int(payload["module_size"]),
            module_base=int(payload["module_base"]),
            player_base=int(payload["player_base"]),
            player_fields={
                str(key): int(value)
                for key, value in dict(payload["player_fields"]).items()
            },
            baseline=dict(payload["baseline"]),
            movement=dict(payload["movement"]),
            direct_module_offsets=tuple(
                int(value) for value in payload["direct_module_offsets"]
            ),
            pointer_paths=tuple(dict(value) for value in payload["pointer_paths"]),
            anchor_outcome=str(payload["anchor_outcome"]),
            monster_hp_anchors=tuple(
                (int(species), int(hp))
                for species, hp in payload.get("monster_hp_anchors", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class StablePathRank:
    path: PointerPath
    sessions_present: int
    total_sessions: int
    movement_confirmed_sessions: int

    @property
    def stable_across_all_sessions(self) -> bool:
        return self.sessions_present == self.total_sessions


def scan_module_pointer_index(
    memory: PointerWorkflowMemory,
    module: ModuleInfo,
    readable: ReadableRegionIndex,
    *,
    chunk_size: int = 1 << 20,
    maximum_slots_per_target: int = 64,
    check: Callable[[], None] | None = None,
) -> dict[int, tuple[int, ...]]:
    """Index aligned module DWORDs whose values point into readable memory."""

    if module.size <= 0:
        raise ValueError("module image size must be positive")
    if chunk_size < 0x1000:
        raise ValueError("chunk_size must be at least 0x1000")
    if maximum_slots_per_target <= 0:
        raise ValueError("maximum_slots_per_target must be positive")

    refs: dict[int, list[int]] = {}
    start = int(module.base_address)
    stop = start + int(module.size)
    cursor = start
    carry = b""
    while cursor < stop:
        if check is not None:
            check()
        amount = min(int(chunk_size), stop - cursor)
        data = memory.read(cursor, amount)
        haystack = carry + data
        haystack_base = cursor - len(carry)
        first = (-haystack_base) % 4
        for relative in range(first, len(haystack) - 3, 4):
            target = int(struct.unpack_from("<I", haystack, relative)[0])
            if target <= 0x10000 or target % 4 != 0:
                continue
            if not readable.contains(target, 4):
                continue
            slot = haystack_base + relative
            if not start <= slot < stop:
                continue
            bucket = refs.setdefault(target, [])
            if len(bucket) < maximum_slots_per_target:
                bucket.append(slot)
        carry = haystack[-3:] if len(haystack) >= 3 else haystack
        cursor += amount
    return {target: tuple(sorted(set(slots))) for target, slots in refs.items()}


def _read_u32(memory: PointerWorkflowMemory, address: int) -> int:
    return int(struct.unpack("<I", memory.read(address, 4))[0])


def resolve_pointer_path(
    memory: PointerWorkflowMemory,
    module_base: int,
    path: PointerPath,
) -> int:
    value = _read_u32(memory, int(module_base) + path.root_module_offset)
    for offset in path.field_offsets:
        value = _read_u32(memory, value + offset)
    return value


def scan_module_rooted_paths(
    memory: PointerWorkflowMemory,
    module: ModuleInfo,
    readable: ReadableRegionIndex,
    module_refs: dict[int, tuple[int, ...]],
    target: int,
    *,
    configured_root_offsets: tuple[int, ...] = (),
    maximum_depth: int = 3,
    field_span: int = 0x1000,
    maximum_roots: int = 12000,
    maximum_nodes: int = 100000,
    maximum_children_per_node: int = 96,
    maximum_paths: int = 512,
    check: Callable[[], None] | None = None,
) -> tuple[PointerPath, ...]:
    """Find bounded module-rooted pointer paths to an exact target.

    This is a read-only, forward pointer-map scan. It starts from pointer-like
    DWORDs in the module image and explores aligned pointer fields inside the
    referenced objects. The bounds deliberately prevent a CE-style explosion.
    """

    if maximum_depth < 0:
        raise ValueError("maximum_depth cannot be negative")
    if field_span < 4 or field_span % 4:
        raise ValueError("field_span must be a positive multiple of four")

    module_base = int(module.base_address)
    configured_slots = tuple(module_base + int(value) for value in configured_root_offsets)
    roots: list[tuple[tuple[int, int, int], int, int]] = []
    for root_value, slots in module_refs.items():
        for slot in slots:
            distance = (
                min(abs(slot - configured) for configured in configured_slots)
                if configured_slots
                else slot - module_base
            )
            roots.append(
                (
                    (
                        int(root_value != target),
                        int(distance),
                        int(slot),
                    ),
                    slot,
                    root_value,
                )
            )
    roots.sort(key=lambda item: item[0])
    roots = roots[: max(1, int(maximum_roots))]

    found: dict[str, PointerPath] = {}
    queue: deque[tuple[int, int, tuple[int, ...], tuple[int, ...]]] = deque()
    for _priority, slot, root_value in roots:
        root_offset = slot - module_base
        if root_value == target:
            path = PointerPath(root_offset, ())
            found[path.signature] = path
            continue
        if maximum_depth > 0 and readable.contains(root_value, 4):
            queue.append((root_offset, root_value, (), (root_value,)))

    visited: set[tuple[int, int, int]] = set()
    nodes = 0
    while queue and nodes < maximum_nodes and len(found) < maximum_paths:
        if check is not None:
            check()
        root_offset, address, offsets, ancestors = queue.popleft()
        state = (root_offset, address, len(offsets))
        if state in visited:
            continue
        visited.add(state)
        nodes += 1
        try:
            data = memory.read(address, field_span)
        except Exception:
            continue

        children: list[tuple[int, int]] = []
        for field_offset in range(0, len(data) - 3, 4):
            value = int(struct.unpack_from("<I", data, field_offset)[0])
            if value == target:
                path = PointerPath(root_offset, offsets + (field_offset,))
                found[path.signature] = path
                if len(found) >= maximum_paths:
                    break
                continue
            if len(offsets) + 1 >= maximum_depth:
                continue
            if (
                value <= 0x10000
                or value % 4 != 0
                or value in ancestors
                or not readable.contains(value, 4)
            ):
                continue
            children.append((field_offset, value))

        # Small offsets are more often stable manager/object fields than deep
        # array payloads. This also makes repeated captures deterministic.
        children.sort(key=lambda item: (item[0], item[1]))
        for field_offset, value in children[:maximum_children_per_node]:
            queue.append(
                (
                    root_offset,
                    value,
                    offsets + (field_offset,),
                    ancestors + (value,),
                )
            )

    return tuple(
        sorted(
            found.values(),
            key=lambda path: (
                path.depth,
                min(
                    (
                        abs(path.root_module_offset - configured)
                        for configured in configured_root_offsets
                    ),
                    default=path.root_module_offset,
                ),
                path.root_module_offset,
                path.field_offsets,
            ),
        )
    )


def read_player_observation(
    memory: PointerWorkflowMemory,
    observation: AnchoredPlayerObservation,
) -> tuple[float, float, float, int, int, int]:
    base = observation.player_base
    data = memory.read(
        base,
        max(
            observation.self_pointer_offset,
            observation.hp_offset,
            observation.max_hp_offset,
            observation.z_offset,
        )
        + 4,
    )
    x = float(struct.unpack_from("<f", data, observation.x_offset)[0])
    y = float(struct.unpack_from("<f", data, observation.y_offset)[0])
    z = float(struct.unpack_from("<f", data, observation.z_offset)[0])
    hp = int(struct.unpack_from("<i", data, observation.hp_offset)[0])
    maximum_hp = int(
        struct.unpack_from("<i", data, observation.max_hp_offset)[0]
    )
    self_pointer = int(
        struct.unpack_from("<I", data, observation.self_pointer_offset)[0]
    )
    return x, y, z, hp, maximum_hp, self_pointer


def wait_for_player_movement(
    memory: PointerWorkflowMemory,
    observation: AnchoredPlayerObservation,
    *,
    minimum_distance_native: float = 0.5,
    timeout_seconds: float = 180.0,
    interval_seconds: float = 0.10,
    stable_samples: int = 3,
    stable_tolerance_native: float = 0.05,
    check: Callable[[], None] | None = None,
) -> MovementEvidence:
    """Wait for the exact anchored object to move, stop, and remain stable."""

    deadline = monotonic() + max(0.1, float(timeout_seconds))
    baseline_x = observation.x
    baseline_z = observation.z
    samples: deque[tuple[float, float, float]] = deque(
        maxlen=max(3, int(stable_samples))
    )
    total_samples = 0
    greatest_distance = 0.0

    while monotonic() < deadline:
        if check is not None:
            check()
        try:
            x, y, z, hp, maximum_hp, self_pointer = read_player_observation(
                memory,
                observation,
            )
        except Exception as error:
            return MovementEvidence(
                False,
                greatest_distance,
                observation.x,
                observation.y,
                observation.z,
                total_samples,
                f"Player object became unreadable: {error}",
            )
        total_samples += 1
        if (
            not all(math.isfinite(value) for value in (x, y, z))
            or self_pointer != observation.player_base
            or hp != observation.current_hp
            or maximum_hp != observation.maximum_hp
        ):
            return MovementEvidence(
                False,
                greatest_distance,
                x,
                y,
                z,
                total_samples,
                "Player identity/HP fields changed during movement validation.",
            )

        distance = math.hypot(x - baseline_x, z - baseline_z)
        greatest_distance = max(greatest_distance, distance)
        samples.append((x, y, z))
        if distance >= minimum_distance_native and len(samples) == samples.maxlen:
            final_x, final_y, final_z = samples[-1]
            stationary = all(
                math.hypot(px - final_x, pz - final_z)
                <= stable_tolerance_native
                for px, _py, pz in samples
            )
            if stationary:
                return MovementEvidence(
                    True,
                    distance,
                    final_x,
                    final_y,
                    final_z,
                    total_samples,
                    f"Confirmed coherent movement of {distance:.3f} native units.",
                )
        sleep(max(0.01, float(interval_seconds)))

    return MovementEvidence(
        False,
        greatest_distance,
        observation.x,
        observation.y,
        observation.z,
        total_samples,
        (
            f"No stable movement of at least {minimum_distance_native:.3f} "
            "native units was observed before the timeout."
        ),
    )


def make_snapshot(
    *,
    memory: PointerWorkflowMemory,
    module: ModuleInfo,
    observation: AnchoredPlayerObservation,
    movement: MovementEvidence,
    paths: tuple[PointerPath, ...],
    anchor_outcome: str,
    monster_hp_anchors: tuple[tuple[int, int], ...] = (),
) -> PointerScanSnapshot:
    direct_offsets = tuple(
        sorted(
            slot - module.base_address
            for slot in observation.direct_module_slots
            if module.base_address <= slot < module.base_address + module.size
        )
    )
    return PointerScanSnapshot(
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        captured_at_utc=datetime.now(UTC).isoformat(),
        pid=int(memory.pid),
        module_name=module.name,
        module_path=module.path,
        module_size=module.size,
        module_base=module.base_address,
        player_base=observation.player_base,
        player_fields={
            "self": observation.self_pointer_offset,
            "hp": observation.hp_offset,
            "max_hp": observation.max_hp_offset,
            "species": observation.species_offset,
            "active_species": observation.active_species_offset,
            "x": observation.x_offset,
            "y": observation.y_offset,
            "z": observation.z_offset,
        },
        baseline={
            "x": observation.x,
            "y": observation.y,
            "z": observation.z,
            "current_hp": observation.current_hp,
            "maximum_hp": observation.maximum_hp,
        },
        movement=asdict(movement),
        direct_module_offsets=direct_offsets,
        pointer_paths=tuple(
            {
                "root_module_offset": path.root_module_offset,
                "field_offsets": list(path.field_offsets),
                "signature": path.signature,
                "depth": path.depth,
            }
            for path in paths
        ),
        anchor_outcome=anchor_outcome,
        monster_hp_anchors=monster_hp_anchors,
    )


def save_snapshot(
    snapshot: PointerScanSnapshot,
    directory: str | Path,
) -> Path:
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = output_dir / f"pointer_scan_{timestamp}_pid{snapshot.pid}.json"
    destination.write_text(snapshot.to_json(), encoding="utf-8")
    return destination


def load_compatible_snapshots(
    directory: str | Path,
    current: PointerScanSnapshot,
) -> tuple[PointerScanSnapshot, ...]:
    snapshots: list[PointerScanSnapshot] = []
    for path in sorted(Path(directory).glob("pointer_scan_*.json")):
        try:
            candidate = PointerScanSnapshot.from_path(path)
        except Exception:
            continue
        same_path = (
            not current.module_path
            or not candidate.module_path
            or Path(candidate.module_path).as_posix().casefold()
            == Path(current.module_path).as_posix().casefold()
        )
        if (
            candidate.schema_version == SNAPSHOT_SCHEMA_VERSION
            and candidate.module_name.casefold() == current.module_name.casefold()
            and candidate.module_size == current.module_size
            and same_path
        ):
            snapshots.append(candidate)
    return tuple(snapshots)


def rank_stable_paths(
    snapshots: tuple[PointerScanSnapshot, ...],
) -> tuple[StablePathRank, ...]:
    if not snapshots:
        return ()
    counts: dict[str, tuple[PointerPath, int, int]] = {}
    for snapshot in snapshots:
        seen: set[str] = set()
        movement_confirmed = bool(snapshot.movement.get("confirmed", False))
        for payload in snapshot.pointer_paths:
            path = PointerPath(
                int(payload["root_module_offset"]),
                tuple(int(value) for value in payload["field_offsets"]),
            )
            if path.signature in seen:
                continue
            seen.add(path.signature)
            current = counts.get(path.signature)
            if current is None:
                counts[path.signature] = (
                    path,
                    1,
                    int(movement_confirmed),
                )
            else:
                counts[path.signature] = (
                    current[0],
                    current[1] + 1,
                    current[2] + int(movement_confirmed),
                )
    total = len(snapshots)
    ranked = tuple(
        StablePathRank(path, present, total, confirmed)
        for path, present, confirmed in counts.values()
    )
    return tuple(
        sorted(
            ranked,
            key=lambda item: (
                item.sessions_present,
                item.movement_confirmed_sessions,
                -item.path.depth,
                -item.path.root_module_offset,
            ),
            reverse=True,
        )
    )
