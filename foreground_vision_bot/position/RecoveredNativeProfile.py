from __future__ import annotations

import json
import math
import os
import struct
from dataclasses import asdict, dataclass
from hashlib import sha256
from datetime import datetime, timezone
from time import time_ns
from pathlib import Path, PureWindowsPath
from typing import Iterable, Mapping, Protocol

from .AuthoritativeActorDiscovery import AuthoritativeActorRefresh, refresh_authoritative_actors
from .NativeTraceTargets import (
    TraceMonsterTarget,
    TracePlayerTarget,
    TraceTargetDiscovery,
    TraceTargetEvidence,
)
from .Win32ProcessMemory import ModuleInfo

PROFILE_VERSION = 1


def _module_filename(value: str) -> str:
    text = str(value or "")
    if "\\" in text:
        return PureWindowsPath(text).name
    return Path(text).name


class ProfileMemory(Protocol):
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


def default_profile_path() -> Path:
    root = os.environ.get("LOCALAPPDATA")
    base = Path(root) if root else Path.home() / ".flyffcv"
    return base / "FlyFFCV" / "native_recovery_profile.json"


@dataclass(frozen=True, slots=True)
class RecoveredNativeProfile:
    version: int
    module_name: str
    module_size: int
    module_filename: str
    module_sha256: str
    player_slot_offsets: tuple[int, ...]
    player_self_offsets: tuple[int, ...]
    monster_self_offsets: tuple[int, ...]
    player_hp_offset: int
    species_offset: int
    active_species_offset: int
    monster_hp_offset: int
    x_offset: int
    y_offset: int
    z_offset: int
    actor_stride: int | None
    authoritative_relation_offset: int
    anchor_species: int
    anchor_hp: int
    expected_full_hp_by_species: tuple[tuple[int, int], ...]
    saved_at_utc: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key in (
            "player_slot_offsets",
            "player_self_offsets",
            "monster_self_offsets",
            "expected_full_hp_by_species",
        ):
            payload[key] = [
                list(item) if isinstance(item, tuple) else item
                for item in payload[key]
            ]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RecoveredNativeProfile":
        if int(payload.get("version", 0)) != PROFILE_VERSION:
            raise ValueError("unsupported native recovery profile version")

        def ints(name: str) -> tuple[int, ...]:
            raw = payload.get(name, ())
            if not isinstance(raw, (list, tuple)):
                raise ValueError(f"{name} must be a list")
            return tuple(int(value) for value in raw)

        raw_expected = payload.get("expected_full_hp_by_species", ())
        if not isinstance(raw_expected, (list, tuple)):
            raise ValueError("expected_full_hp_by_species must be a list")
        expected = tuple(
            (int(item[0]), int(item[1]))
            for item in raw_expected
            if isinstance(item, (list, tuple)) and len(item) == 2
        )
        actor_stride_raw = payload.get("actor_stride")
        actor_stride = None if actor_stride_raw is None else int(actor_stride_raw)
        profile = cls(
            version=PROFILE_VERSION,
            module_name=str(payload["module_name"]),
            module_size=int(payload["module_size"]),
            module_filename=str(payload.get("module_filename", "")),
            module_sha256=str(payload.get("module_sha256", "")),
            player_slot_offsets=ints("player_slot_offsets"),
            player_self_offsets=ints("player_self_offsets"),
            monster_self_offsets=ints("monster_self_offsets"),
            player_hp_offset=int(payload["player_hp_offset"]),
            species_offset=int(payload["species_offset"]),
            active_species_offset=int(payload["active_species_offset"]),
            monster_hp_offset=int(payload["monster_hp_offset"]),
            x_offset=int(payload["x_offset"]),
            y_offset=int(payload["y_offset"]),
            z_offset=int(payload["z_offset"]),
            actor_stride=actor_stride,
            authoritative_relation_offset=int(payload["authoritative_relation_offset"]),
            anchor_species=int(payload["anchor_species"]),
            anchor_hp=int(payload["anchor_hp"]),
            expected_full_hp_by_species=expected,
            saved_at_utc=str(payload.get("saved_at_utc", "")),
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        if not self.module_name or self.module_size <= 0:
            raise ValueError("module identity is incomplete")
        if self.module_sha256 and (
            len(self.module_sha256) != 64
            or any(
                character not in "0123456789abcdefABCDEF"
                for character in self.module_sha256
            )
        ):
            raise ValueError("module_sha256 must be hexadecimal SHA-256")
        if not self.player_slot_offsets:
            raise ValueError("profile has no module-relative player slots")
        if not self.player_self_offsets or not self.monster_self_offsets:
            raise ValueError("profile has no validated self-reference offsets")
        if self.anchor_species <= 0 or self.anchor_hp <= 0:
            raise ValueError("profile has no trusted monster anchor")
        for name, value in (
            ("player_hp_offset", self.player_hp_offset),
            ("species_offset", self.species_offset),
            ("active_species_offset", self.active_species_offset),
            ("monster_hp_offset", self.monster_hp_offset),
            ("x_offset", self.x_offset),
            ("y_offset", self.y_offset),
            ("z_offset", self.z_offset),
            ("authoritative_relation_offset", self.authoritative_relation_offset),
        ):
            if value < 0 or value > 0x10000 or value % 4:
                raise ValueError(f"invalid {name}: {value}")
        if self.actor_stride is not None and (
            self.actor_stride <= 0 or self.actor_stride > 0x100000
        ):
            raise ValueError("invalid actor stride")


@dataclass(frozen=True, slots=True)
class RestoredNativeState:
    profile: RecoveredNativeProfile
    discovery: TraceTargetDiscovery
    relation_offset: int
    relation_value: int
    authoritative: AuthoritativeActorRefresh


def load_profile(path: str | Path | None = None) -> RecoveredNativeProfile | None:
    selected = default_profile_path() if path is None else Path(path)
    if not selected.is_file():
        return None
    try:
        payload = json.loads(selected.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        return RecoveredNativeProfile.from_dict(payload)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def save_profile(
    profile: RecoveredNativeProfile,
    path: str | Path | None = None,
) -> Path:
    profile.validate()
    selected = default_profile_path() if path is None else Path(path)
    selected.parent.mkdir(parents=True, exist_ok=True)
    temporary = selected.with_name(
        f".{selected.name}.{os.getpid()}.{time_ns()}.tmp"
    )
    encoded = (
        json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        RecoveredNativeProfile.from_dict(
            json.loads(temporary.read_text(encoding="utf-8"))
        )
        os.replace(temporary, selected)
        _fsync_directory(selected.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return selected


def _sha256_file(path: str) -> str:
    selected = Path(path)
    if not selected.is_file():
        return ""
    digest = sha256()
    try:
        with selected.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest().upper()


def profile_from_reader(
    *,
    module: ModuleInfo,
    player_slots: Iterable[int],
    player_target: TracePlayerTarget,
    monster_target: TraceMonsterTarget,
    actor_stride: int | None,
    authoritative_relation_offset: int,
    expected_full_hp_by_species: Mapping[int, int],
) -> RecoveredNativeProfile:
    offsets = tuple(
        sorted(
            {
                int(slot) - int(module.base_address)
                for slot in player_slots
                if module.base_address <= int(slot) < module.base_address + module.size
            }
        )
    )
    expected = tuple(
        sorted(
            (int(species), int(hp))
            for species, hp in expected_full_hp_by_species.items()
            if int(species) > 0 and int(hp) > 0
        )
    )
    anchor_hp = int(dict(expected).get(monster_target.species, monster_target.hp))
    profile = RecoveredNativeProfile(
        version=PROFILE_VERSION,
        module_name=module.name,
        module_size=int(module.size),
        module_filename=_module_filename(module.path) if module.path else module.name,
        module_sha256=_sha256_file(module.path),
        player_slot_offsets=offsets,
        player_self_offsets=tuple(
            int(value) for value in player_target.self_pointer_offsets
        ),
        monster_self_offsets=tuple(
            int(value) for value in monster_target.self_pointer_offsets
        ),
        player_hp_offset=int(player_target.hp_offset),
        species_offset=int(monster_target.species_offset),
        active_species_offset=int(monster_target.active_species_offset),
        monster_hp_offset=int(monster_target.hp_offset),
        x_offset=int(monster_target.x_offset),
        y_offset=int(monster_target.y_offset),
        z_offset=int(monster_target.z_offset),
        actor_stride=None if actor_stride is None else int(actor_stride),
        authoritative_relation_offset=int(authoritative_relation_offset),
        anchor_species=int(monster_target.species),
        anchor_hp=anchor_hp,
        expected_full_hp_by_species=expected,
        saved_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    profile.validate()
    return profile


def _u32(memory: ProfileMemory, address: int) -> int:
    return int(struct.unpack("<I", memory.read(int(address), 4))[0])


def _i32(memory: ProfileMemory, address: int) -> int:
    return int(struct.unpack("<i", memory.read(int(address), 4))[0])


def _f32(memory: ProfileMemory, address: int) -> float:
    return float(struct.unpack("<f", memory.read(int(address), 4))[0])


def _evidence(profile: RecoveredNativeProfile) -> TraceTargetEvidence:
    return TraceTargetEvidence(
        bytes_scanned=0,
        regions_scanned=0,
        read_failures=0,
        species_hits=0,
        spawn_x_hits=0,
        monster_candidates=1,
        player_candidates=1,
        monster_hp_rejections=0,
        monster_coordinate_rejections=0,
        player_hp_rejections=0,
        player_coordinate_rejections=0,
        observed_hp_values=(),
        monster_base_hypotheses=1,
        monster_layout_ties=0,
        monster_self_aliases=len(profile.monster_self_offsets),
        player_self_rejections=0,
        inferred_species_offset=profile.species_offset,
        inferred_active_species_offset=profile.active_species_offset,
        inferred_monster_hp_offset=profile.monster_hp_offset,
        inferred_x_offset=profile.x_offset,
        inferred_y_offset=profile.y_offset,
        inferred_z_offset=profile.z_offset,
        selected_player_hp_offset=profile.player_hp_offset,
    )


def restore_profile(
    memory: ProfileMemory,
    module: ModuleInfo,
    profile: RecoveredNativeProfile,
    *,
    selected_species_ids: Iterable[int],
    maximum_address: int,
    private_memory_only: bool,
    chunk_size: int,
    coordinate_limit: float,
    cancellation: object | None = None,
    deadline: float | None = None,
) -> RestoredNativeState:
    profile.validate()
    if module.name.casefold() != profile.module_name.casefold():
        raise ValueError("cached profile belongs to another module")
    if int(module.size) != int(profile.module_size):
        raise ValueError("cached profile belongs to another client build")
    if (
        profile.module_filename
        and _module_filename(module.path).casefold()
        != profile.module_filename.casefold()
    ):
        raise ValueError("cached profile module filename does not match")
    if profile.module_sha256:
        current_hash = _sha256_file(module.path)
        if not current_hash or current_hash != profile.module_sha256.upper():
            raise ValueError("cached profile belongs to another client executable")

    valid_slots: list[int] = []
    candidate_rows: list[tuple[int, int, float, float, float, int]] = []
    for relative in profile.player_slot_offsets:
        if relative < 0 or relative + 4 > module.size:
            continue
        slot = module.base_address + int(relative)
        try:
            base = _u32(memory, slot)
            if base <= 0x10000:
                continue
            if not any(
                _u32(memory, base + offset) == base
                for offset in profile.player_self_offsets
            ):
                continue
            hp = _i32(memory, base + profile.player_hp_offset)
            x = _f32(memory, base + profile.x_offset)
            y = _f32(memory, base + profile.y_offset)
            z = _f32(memory, base + profile.z_offset)
            relation_value = _u32(
                memory,
                base + profile.authoritative_relation_offset,
            )
            if hp < 0 or relation_value <= 0x10000:
                continue
            if not all(
                math.isfinite(value) and abs(value) <= coordinate_limit
                for value in (x, y, z)
            ):
                continue
            memory.read(relation_value, 4)
        except Exception:
            continue
        valid_slots.append(slot)
        candidate_rows.append((base, hp, x, y, z, relation_value))

    unique_bases = {row[0] for row in candidate_rows}
    if len(unique_bases) != 1:
        raise ValueError("cached player aliases did not resolve to one current player")
    base, hp, x, y, z, relation_value = candidate_rows[0]
    valid_slots = [slot for slot in valid_slots if _u32(memory, slot) == base]
    if not valid_slots:
        raise ValueError("cached player aliases became inconsistent")

    player = TracePlayerTarget(
        base=base,
        hp=hp,
        x=x,
        y=y,
        z=z,
        self_pointer_offsets=profile.player_self_offsets,
        direct_module_slots=tuple(sorted(set(valid_slots))),
        hp_offset=profile.player_hp_offset,
        species_offset=profile.species_offset,
        active_species_offset=profile.active_species_offset,
        x_offset=profile.x_offset,
        y_offset=profile.y_offset,
        z_offset=profile.z_offset,
    )
    synthetic = TraceMonsterTarget(
        base=base,
        species=profile.anchor_species,
        hp=profile.anchor_hp,
        x=x,
        y=y,
        z=z,
        self_pointer_offsets=profile.monster_self_offsets,
        species_offset=profile.species_offset,
        active_species_offset=profile.active_species_offset,
        hp_offset=profile.monster_hp_offset,
        x_offset=profile.x_offset,
        y_offset=profile.y_offset,
        z_offset=profile.z_offset,
    )
    provisional = TraceTargetDiscovery(
        player=player,
        monsters=(synthetic,),
        evidence=_evidence(profile),
        outcome="success",
        message="Restored validated native layout from the last known profile.",
    )
    selected = {int(value) for value in selected_species_ids if int(value) > 0}
    selected.add(profile.anchor_species)
    authoritative = refresh_authoritative_actors(
        memory,
        provisional,
        relation_offset=profile.authoritative_relation_offset,
        relation_value=relation_value,
        selected_species_ids=selected,
        actor_stride=profile.actor_stride,
        maximum_address=maximum_address,
        private_memory_only=private_memory_only,
        chunk_size=chunk_size,
        coordinate_limit=coordinate_limit,
        cancellation=cancellation,
        deadline=deadline,
    )
    if len(authoritative.actor_bases) < 2:
        raise ValueError("cached authoritative relation did not recover enough actors")
    species_counts = dict(authoritative.species_counts)
    if species_counts.get(profile.anchor_species, 0) < 1:
        raise ValueError("cached relation did not recover the trusted anchor species")

    real_targets: list[TraceMonsterTarget] = []
    for actor_base in authoritative.actor_bases[:256]:
        try:
            species = _i32(memory, actor_base + profile.species_offset)
            actor_hp = _i32(memory, actor_base + profile.monster_hp_offset)
            actor_x = _f32(memory, actor_base + profile.x_offset)
            actor_y = _f32(memory, actor_base + profile.y_offset)
            actor_z = _f32(memory, actor_base + profile.z_offset)
        except Exception:
            continue
        real_targets.append(
            TraceMonsterTarget(
                base=actor_base,
                species=species,
                hp=actor_hp,
                x=actor_x,
                y=actor_y,
                z=actor_z,
                self_pointer_offsets=profile.monster_self_offsets,
                species_offset=profile.species_offset,
                active_species_offset=profile.active_species_offset,
                hp_offset=profile.monster_hp_offset,
                x_offset=profile.x_offset,
                y_offset=profile.y_offset,
                z_offset=profile.z_offset,
            )
        )
    if not real_targets:
        raise ValueError("cached relation recovered no readable actor samples")
    discovery = TraceTargetDiscovery(
        player=player,
        monsters=tuple(real_targets),
        evidence=_evidence(profile),
        outcome="success",
        message="Last known native profile validated against the current process.",
    )
    return RestoredNativeState(
        profile=profile,
        discovery=discovery,
        relation_offset=profile.authoritative_relation_offset,
        relation_value=relation_value,
        authoritative=authoritative,
    )
