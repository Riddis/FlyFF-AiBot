from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from time import monotonic
from typing import Mapping, Protocol

from .NativeTraceTargets import TraceMonsterTarget


class MonsterRediscoveryMemory(Protocol):
    last_search_diagnostics: object

    def read(self, address: int, size: int) -> bytes: ...

    def find_u32(
        self,
        value: int,
        *,
        maximum_address: int,
        private_only: bool,
        chunk_size: int,
        cancellation: object | None = None,
        deadline: float | None = None,
    ) -> tuple[int, ...]: ...


@dataclass(frozen=True, slots=True)
class MonsterRediscoveryEvidence:
    elapsed_seconds: float
    species_hits: int
    base_candidates: int
    exact_full_hp_anchors: int
    self_rejections: int
    hp_rejections: int
    coordinate_rejections: int
    read_failures: int
    bytes_scanned: int
    regions_scanned: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MonsterRediscoveryResult:
    targets: tuple[TraceMonsterTarget, ...]
    evidence: MonsterRediscoveryEvidence

    def to_dict(self) -> dict[str, object]:
        return {
            "targets": [asdict(target) for target in self.targets],
            "evidence": self.evidence.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SelectedSpeciesRediscoveryEvidence:
    elapsed_seconds: float
    species_hits: int
    base_candidates: int
    validated_targets: int
    self_rejections: int
    hp_rejections: int
    coordinate_rejections: int
    read_failures: int
    bytes_scanned: int
    regions_scanned: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SelectedSpeciesRediscoveryResult:
    targets: tuple[TraceMonsterTarget, ...]
    evidence: SelectedSpeciesRediscoveryEvidence

    def to_dict(self) -> dict[str, object]:
        return {
            "targets": [asdict(target) for target in self.targets],
            "evidence": self.evidence.to_dict(),
        }


def _i32(data: bytes) -> int:
    return int.from_bytes(data, "little", signed=True)


def _u32(data: bytes) -> int:
    return int.from_bytes(data, "little", signed=False)


def _f32(data: bytes) -> float:
    import struct

    return float(struct.unpack("<f", data)[0])


def rediscover_known_layout_monsters(
    memory: MonsterRediscoveryMemory,
    *,
    template: TraceMonsterTarget,
    species_hp: Mapping[int, int],
    maximum_address: int,
    coordinate_limit: float,
    chunk_size: int = 4 << 20,
    deadline: float | None = None,
) -> MonsterRediscoveryResult:
    """Find exact full-health monster anchors using an already proven layout.

    This intentionally does not rediscover the player and therefore remains valid
    after the player has moved away from spawn or changed HP. It performs no
    writes and does not attach a debugger.
    """

    started = monotonic()
    targets: dict[int, TraceMonsterTarget] = {}
    species_hits = 0
    base_candidates = 0
    self_rejections = 0
    hp_rejections = 0
    coordinate_rejections = 0
    read_failures = 0
    bytes_scanned = 0
    regions_scanned = 0

    for species, exact_hp in sorted((int(s), int(h)) for s, h in species_hp.items()):
        if species <= 0 or exact_hp <= 0:
            continue
        hits = memory.find_u32(
            species,
            maximum_address=int(maximum_address),
            private_only=True,
            chunk_size=max(4096, int(chunk_size)),
            deadline=deadline,
        )
        species_hits += len(hits)
        diagnostics = getattr(memory, "last_search_diagnostics", None)
        bytes_scanned += int(getattr(diagnostics, "bytes_read", 0))
        regions_scanned += int(getattr(diagnostics, "regions_read", 0))
        for address in hits:
            if address % 4:
                continue
            base = int(address) - int(template.species_offset)
            if base <= 0x10000 or base in targets:
                continue
            base_candidates += 1
            try:
                has_self_alias = False
                for offset in template.self_pointer_offsets:
                    try:
                        if _u32(memory.read(base + offset, 4)) == base:
                            has_self_alias = True
                            break
                    except Exception:
                        continue
                if not has_self_alias:
                    self_rejections += 1
                    continue
                if _i32(memory.read(base + template.species_offset, 4)) != species:
                    continue
                hp = _i32(memory.read(base + template.hp_offset, 4))
                if hp != exact_hp:
                    hp_rejections += 1
                    continue
                x = _f32(memory.read(base + template.x_offset, 4))
                y = _f32(memory.read(base + template.y_offset, 4))
                z = _f32(memory.read(base + template.z_offset, 4))
            except Exception:
                read_failures += 1
                continue
            if not all(
                math.isfinite(value) and abs(value) <= float(coordinate_limit)
                for value in (x, y, z)
            ):
                coordinate_rejections += 1
                continue
            targets[base] = TraceMonsterTarget(
                base=base,
                species=species,
                hp=hp,
                x=x,
                y=y,
                z=z,
                self_pointer_offsets=template.self_pointer_offsets,
                species_offset=template.species_offset,
                active_species_offset=template.active_species_offset,
                hp_offset=template.hp_offset,
                x_offset=template.x_offset,
                y_offset=template.y_offset,
                z_offset=template.z_offset,
            )

    evidence = MonsterRediscoveryEvidence(
        elapsed_seconds=monotonic() - started,
        species_hits=species_hits,
        base_candidates=base_candidates,
        exact_full_hp_anchors=len(targets),
        self_rejections=self_rejections,
        hp_rejections=hp_rejections,
        coordinate_rejections=coordinate_rejections,
        read_failures=read_failures,
        bytes_scanned=bytes_scanned,
        regions_scanned=regions_scanned,
    )
    return MonsterRediscoveryResult(
        targets=tuple(sorted(targets.values(), key=lambda item: item.base)),
        evidence=evidence,
    )


def rediscover_selected_layout_monsters(
    memory: MonsterRediscoveryMemory,
    *,
    template: TraceMonsterTarget,
    species_ids: tuple[int, ...] | list[int] | set[int],
    maximum_address: int,
    coordinate_limit: float,
    chunk_size: int = 4 << 20,
    cancellation: object | None = None,
    deadline: float | None = None,
) -> SelectedSpeciesRediscoveryResult:
    """Find selected species through an actor layout proven earlier.

    This is intentionally a post-recovery operation. It never participates in
    player selection, actor-layout inference, or HP-offset inference. The exact
    anchor species proves those fields first; this scan merely finds additional
    actors (including damaged and zero-HP slots) that share that proven layout.
    """

    started = monotonic()
    targets: dict[int, TraceMonsterTarget] = {}
    species_hits = 0
    base_candidates = 0
    self_rejections = 0
    hp_rejections = 0
    coordinate_rejections = 0
    read_failures = 0
    bytes_scanned = 0
    regions_scanned = 0

    for species in sorted({int(value) for value in species_ids if int(value) > 0}):
        hits = memory.find_u32(
            species,
            maximum_address=int(maximum_address),
            private_only=True,
            chunk_size=max(4096, int(chunk_size)),
            cancellation=cancellation,
            deadline=deadline,
        )
        species_hits += len(hits)
        diagnostics = getattr(memory, "last_search_diagnostics", None)
        bytes_scanned += int(getattr(diagnostics, "bytes_read", 0))
        regions_scanned += int(getattr(diagnostics, "regions_read", 0))
        for address in hits:
            if address % 4:
                continue
            base = int(address) - int(template.species_offset)
            if base <= 0x10000 or base in targets:
                continue
            base_candidates += 1
            try:
                has_self_alias = False
                for offset in template.self_pointer_offsets:
                    try:
                        if _u32(memory.read(base + offset, 4)) == base:
                            has_self_alias = True
                            break
                    except Exception:
                        continue
                if not has_self_alias:
                    self_rejections += 1
                    continue
                if _i32(memory.read(base + template.species_offset, 4)) != species:
                    continue
                hp = _i32(memory.read(base + template.hp_offset, 4))
                if hp < 0:
                    hp_rejections += 1
                    continue
                x = _f32(memory.read(base + template.x_offset, 4))
                y = _f32(memory.read(base + template.y_offset, 4))
                z = _f32(memory.read(base + template.z_offset, 4))
            except Exception:
                read_failures += 1
                continue
            if not all(
                math.isfinite(value) and abs(value) <= float(coordinate_limit)
                for value in (x, y, z)
            ):
                coordinate_rejections += 1
                continue
            targets[base] = TraceMonsterTarget(
                base=base,
                species=species,
                hp=hp,
                x=x,
                y=y,
                z=z,
                self_pointer_offsets=template.self_pointer_offsets,
                species_offset=template.species_offset,
                active_species_offset=template.active_species_offset,
                hp_offset=template.hp_offset,
                x_offset=template.x_offset,
                y_offset=template.y_offset,
                z_offset=template.z_offset,
            )

    evidence = SelectedSpeciesRediscoveryEvidence(
        elapsed_seconds=monotonic() - started,
        species_hits=species_hits,
        base_candidates=base_candidates,
        validated_targets=len(targets),
        self_rejections=self_rejections,
        hp_rejections=hp_rejections,
        coordinate_rejections=coordinate_rejections,
        read_failures=read_failures,
        bytes_scanned=bytes_scanned,
        regions_scanned=regions_scanned,
    )
    return SelectedSpeciesRediscoveryResult(
        targets=tuple(sorted(targets.values(), key=lambda item: item.base)),
        evidence=evidence,
    )
