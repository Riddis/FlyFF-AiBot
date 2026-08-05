from __future__ import annotations

import json
import math
import os
import struct
from dataclasses import asdict, dataclass
from hashlib import sha256
from datetime import datetime, timezone
from time import monotonic, time_ns
from pathlib import Path, PureWindowsPath
from typing import Callable, Iterable, Mapping, Protocol

from .AuthoritativeActorDiscovery import (
    AuthoritativeActorRefresh,
    PresenceFieldCandidate,
    RelationScanEvidence,
    _presence_field_candidates,
    refresh_authoritative_actors,
)
from .NativeTraceTargets import (
    TraceMonsterTarget,
    TracePlayerTarget,
    TraceTargetDiscovery,
    TraceTargetEvidence,
)
from .Win32ProcessMemory import ModuleInfo

PROFILE_VERSION = 2


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
    presence_species_offset: int | None = None
    presence_species_validated: bool = False
    presence_evidence: PresenceFieldCandidate | None = None
    runtime_pid: int | None = None
    runtime_module_base: int | None = None
    runtime_player_base: int | None = None
    runtime_relation_value: int | None = None
    runtime_actor_bases: tuple[int, ...] = ()
    runtime_species_counts: tuple[tuple[int, int], ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key in (
            "player_slot_offsets",
            "player_self_offsets",
            "monster_self_offsets",
            "expected_full_hp_by_species",
            "runtime_actor_bases",
            "runtime_species_counts",
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
        raw_runtime_species = payload.get("runtime_species_counts", ())
        if not isinstance(raw_runtime_species, (list, tuple)):
            raise ValueError("runtime_species_counts must be a list")
        runtime_species_counts = tuple(
            (int(item[0]), int(item[1]))
            for item in raw_runtime_species
            if isinstance(item, (list, tuple)) and len(item) == 2
        )
        raw_presence_evidence = payload.get("presence_evidence")
        presence_evidence = None
        if isinstance(raw_presence_evidence, Mapping):
            evidence_fields = {
                field: int(raw_presence_evidence.get(field, 0))
                for field in (
                    "offset",
                    "selected_matches",
                    "selected_samples",
                    "zero_hp_matches",
                    "zero_hp_samples",
                    "dormant_clears",
                    "dormant_samples",
                    "cross_slot_alias_matches",
                    "lifecycle_death_retained",
                    "lifecycle_dormant_clears",
                    "lifecycle_reappearances",
                )
            }
            presence_evidence = PresenceFieldCandidate(
                **evidence_fields,
                validated=bool(raw_presence_evidence.get("validated", False)),
            )
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
            presence_species_offset=(
                None
                if payload.get("presence_species_offset") is None
                else int(payload["presence_species_offset"])
            ),
            presence_species_validated=bool(
                payload.get("presence_species_validated", False)
            ),
            presence_evidence=presence_evidence,
            runtime_pid=(
                None if payload.get("runtime_pid") is None
                else int(payload["runtime_pid"])
            ),
            runtime_module_base=(
                None if payload.get("runtime_module_base") is None
                else int(payload["runtime_module_base"])
            ),
            runtime_player_base=(
                None if payload.get("runtime_player_base") is None
                else int(payload["runtime_player_base"])
            ),
            runtime_relation_value=(
                None if payload.get("runtime_relation_value") is None
                else int(payload["runtime_relation_value"])
            ),
            runtime_actor_bases=ints("runtime_actor_bases"),
            runtime_species_counts=runtime_species_counts,
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
        if self.presence_species_offset is not None:
            offset = int(self.presence_species_offset)
            if offset < 0 or offset > 0x10000 or offset % 4:
                raise ValueError("invalid presence_species_offset")
            if self.actor_stride is not None and offset >= self.actor_stride:
                raise ValueError("presence_species_offset is outside the actor stride")
        if self.presence_species_validated and self.presence_species_offset is None:
            raise ValueError("validated presence field has no offset")
        if (
            self.presence_evidence is not None
            and self.presence_species_offset != self.presence_evidence.offset
        ):
            raise ValueError("presence evidence does not match the saved offset")
        runtime_values = (
            self.runtime_pid,
            self.runtime_module_base,
            self.runtime_player_base,
            self.runtime_relation_value,
        )
        if self.runtime_actor_bases or any(value is not None for value in runtime_values):
            if any(value is None or int(value) <= 0 for value in runtime_values):
                raise ValueError("runtime cache identity is incomplete")
            if not self.runtime_actor_bases:
                raise ValueError("runtime cache has no actor bases")


@dataclass(frozen=True, slots=True)
class RestoredNativeState:
    profile: RecoveredNativeProfile
    discovery: TraceTargetDiscovery
    relation_offset: int
    relation_value: int
    authoritative: AuthoritativeActorRefresh
    restore_mode: str = "stable_profile_scan"
    elapsed_seconds: float = 0.0


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
    process_id: int | None = None,
    player_slots: Iterable[int],
    player_target: TracePlayerTarget,
    monster_target: TraceMonsterTarget,
    actor_stride: int | None,
    authoritative_relation_offset: int,
    authoritative_relation_value: int | None = None,
    actor_bases: Iterable[int] = (),
    authoritative_species_counts: Iterable[tuple[int, int]] = (),
    expected_full_hp_by_species: Mapping[int, int],
    presence_species_offset: int | None = None,
    presence_species_validated: bool = False,
    presence_evidence: PresenceFieldCandidate | None = None,
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
        presence_species_offset=(
            None if presence_species_offset is None else int(presence_species_offset)
        ),
        presence_species_validated=bool(presence_species_validated),
        presence_evidence=presence_evidence,
        runtime_pid=(None if process_id is None else int(process_id)),
        runtime_module_base=(
            None if process_id is None else int(module.base_address)
        ),
        runtime_player_base=(
            None if process_id is None else int(player_target.base)
        ),
        runtime_relation_value=(
            None
            if process_id is None or authoritative_relation_value is None
            else int(authoritative_relation_value)
        ),
        runtime_actor_bases=(
            () if process_id is None else tuple(sorted({int(value) for value in actor_bases if int(value) > 0}))
        ),
        runtime_species_counts=(
            ()
            if process_id is None
            else tuple(sorted((int(species), int(count)) for species, count in authoritative_species_counts if int(species) > 0 and int(count) >= 0))
        ),
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



def _runtime_cache_identity_matches(
    memory: ProfileMemory,
    module: ModuleInfo,
    profile: RecoveredNativeProfile,
) -> bool:
    return bool(
        profile.runtime_actor_bases
        and profile.runtime_pid is not None
        and int(getattr(memory, "pid", 0)) == int(profile.runtime_pid)
        and profile.runtime_module_base is not None
        and int(module.base_address) == int(profile.runtime_module_base)
        and profile.runtime_player_base is not None
        and profile.runtime_relation_value is not None
    )


def _restore_runtime_cache(
    memory: ProfileMemory,
    module: ModuleInfo,
    profile: RecoveredNativeProfile,
    *,
    selected_species_ids: Iterable[int],
    coordinate_limit: float,
) -> RestoredNativeState | None:
    """Validate the exact same-process actor cache without scanning memory."""

    if not _runtime_cache_identity_matches(memory, module, profile):
        return None
    started = monotonic()
    expected_player = int(profile.runtime_player_base or 0)
    expected_relation = int(profile.runtime_relation_value or 0)
    valid_slots: list[int] = []
    for relative in profile.player_slot_offsets:
        if relative < 0 or relative + 4 > module.size:
            continue
        slot = int(module.base_address) + int(relative)
        try:
            if _u32(memory, slot) == expected_player:
                valid_slots.append(slot)
        except Exception:
            continue
    if not valid_slots:
        return None
    try:
        if not any(
            _u32(memory, expected_player + offset) == expected_player
            for offset in profile.player_self_offsets
        ):
            return None
        player_hp = _i32(memory, expected_player + profile.player_hp_offset)
        player_x = _f32(memory, expected_player + profile.x_offset)
        player_y = _f32(memory, expected_player + profile.y_offset)
        player_z = _f32(memory, expected_player + profile.z_offset)
        current_relation = _u32(
            memory,
            expected_player + profile.authoritative_relation_offset,
        )
    except Exception:
        return None
    if current_relation != expected_relation or player_hp < 0:
        return None
    if not all(
        math.isfinite(value) and abs(value) <= coordinate_limit
        for value in (player_x, player_y, player_z)
    ):
        return None

    selected = {int(value) for value in selected_species_ids if int(value) > 0}
    selected.add(int(profile.anchor_species))
    valid_bases: list[int] = []
    real_targets: list[TraceMonsterTarget] = []
    species_counts: dict[int, int] = {}
    structural_matches = 0
    unreadable = 0
    self_rejections = 0
    relation_rejections = 0
    species_rejections = 0
    hp_rejections = 0
    coordinate_rejections = 0
    structural_bases: list[int] = []

    for actor_base in profile.runtime_actor_bases:
        base = int(actor_base)
        try:
            if not any(
                _u32(memory, base + offset) == base
                for offset in profile.monster_self_offsets
            ):
                self_rejections += 1
                continue
            if _u32(memory, base + profile.authoritative_relation_offset) != expected_relation:
                relation_rejections += 1
                continue
            structural_matches += 1
            structural_bases.append(base)
            species = _i32(memory, base + profile.species_offset)
            actor_hp = _i32(memory, base + profile.monster_hp_offset)
            actor_x = _f32(memory, base + profile.x_offset)
            actor_y = _f32(memory, base + profile.y_offset)
            actor_z = _f32(memory, base + profile.z_offset)
        except Exception:
            unreadable += 1
            continue
        if selected and species not in selected:
            species_rejections += 1
            continue
        if actor_hp < 0:
            hp_rejections += 1
            continue
        if not all(
            math.isfinite(value) and abs(value) <= coordinate_limit
            for value in (actor_x, actor_y, actor_z)
        ):
            coordinate_rejections += 1
            continue
        valid_bases.append(base)
        species_counts[species] = species_counts.get(species, 0) + 1
        if len(real_targets) < 256:
            real_targets.append(
                TraceMonsterTarget(
                    base=base,
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

    cached_count = len(profile.runtime_actor_bases)
    minimum_structural = max(2, math.ceil(cached_count * 0.60))
    minimum_selected = max(2, math.ceil(cached_count * 0.25))
    saved_species_counts = dict(profile.runtime_species_counts)
    species_coverage_ok = all(
        species in saved_species_counts
        and species_counts.get(species, 0)
        >= max(1, math.ceil(saved_species_counts[species] * 0.50))
        for species in selected
        if species in saved_species_counts or species != profile.anchor_species
    )
    if (
        structural_matches < minimum_structural
        or len(valid_bases) < minimum_selected
        or not real_targets
        or not species_coverage_ok
    ):
        return None

    player = TracePlayerTarget(
        base=expected_player,
        hp=player_hp,
        x=player_x,
        y=player_y,
        z=player_z,
        self_pointer_offsets=profile.player_self_offsets,
        direct_module_slots=tuple(sorted(set(valid_slots))),
        hp_offset=profile.player_hp_offset,
        species_offset=profile.species_offset,
        active_species_offset=profile.active_species_offset,
        x_offset=profile.x_offset,
        y_offset=profile.y_offset,
        z_offset=profile.z_offset,
    )
    discovery = TraceTargetDiscovery(
        player=player,
        monsters=tuple(real_targets),
        evidence=_evidence(profile),
        outcome="success",
        message="Validated the same-process authoritative actor cache.",
    )
    presence_candidates = _presence_field_candidates(
        memory,
        discovery,
        tuple(sorted(set(valid_bases))),
        tuple(sorted(set(structural_bases))),
        selected_species_ids=selected,
        actor_stride=profile.actor_stride,
        relation_offset=profile.authoritative_relation_offset,
        preferred_offsets=tuple(
            dict.fromkeys(
                value
                for value in (profile.presence_species_offset, 0x1DCC)
                if value is not None
            )
        ),
    )
    presence = presence_candidates[0] if presence_candidates else None
    evidence = RelationScanEvidence(
        offset=int(profile.authoritative_relation_offset),
        value=expected_relation,
        references=cached_count,
        unique_candidate_bases=cached_count,
        valid_actor_bases=len(valid_bases),
        exact_anchor_coverage=0,
        exact_anchor_total=0,
        selected_species_counts=tuple(sorted(species_counts.items())),
        self_rejections=self_rejections,
        relation_rejections=relation_rejections,
        species_rejections=species_rejections,
        hp_rejections=hp_rejections,
        coordinate_rejections=coordinate_rejections,
        unreadable_rejections=unreadable,
        search_bytes_read=0,
        search_regions_read=0,
    )
    authoritative = AuthoritativeActorRefresh(
        actor_bases=tuple(sorted(set(valid_bases))),
        species_counts=tuple(sorted(species_counts.items())),
        evidence=evidence,
        active_species_offset=None,
        active_species_validated=False,
        active_candidates=(),
        presence_species_offset=(
            None if presence is None else int(presence.offset)
        ),
        presence_species_validated=bool(
            presence is not None and presence.validated
        ),
        presence_candidates=presence_candidates,
    )
    return RestoredNativeState(
        profile=profile,
        discovery=discovery,
        relation_offset=int(profile.authoritative_relation_offset),
        relation_value=expected_relation,
        authoritative=authoritative,
        restore_mode="same_process_cache",
        elapsed_seconds=max(0.0, monotonic() - started),
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
    status_callback: Callable[[str], None] | None = None,
) -> RestoredNativeState:
    started = monotonic()
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

    runtime_cached = _restore_runtime_cache(
        memory,
        module,
        profile,
        selected_species_ids=selected_species_ids,
        coordinate_limit=coordinate_limit,
    )
    if runtime_cached is not None:
        return runtime_cached

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
        status_callback=status_callback,
        preferred_presence_offsets=tuple(
            dict.fromkeys(
                value
                for value in (profile.presence_species_offset, 0x1DCC)
                if value is not None
            )
        ),
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
        restore_mode="stable_profile_scan",
        elapsed_seconds=max(0.0, monotonic() - started),
    )
