from __future__ import annotations

import gzip
import hashlib
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import msgpack

from legacy.manifest_compat import (
    DEFAULT_PROVENANCE_REGISTRY,
    attested_by_registry,
    missing_map_contract_warning,
    missing_policy_contract_warning,
)

SUPPORTED_RECORDING_SCHEMA_VERSIONS = frozenset({2})
REQUIRED_ARCHIVE_MEMBERS = frozenset(
    {"manifest.json", "frames.msgpack.gz", "events.msgpack.gz", "inputs.msgpack.gz"}
)

DIRECT_KEYBOARD_RECORDING_ROLE = "direct_keyboard_demonstration"
DIRECT_KEYBOARD_CONTROL_SCHEME = "keyboard_wasd"


def has_validated_presence(manifest: dict[str, object]) -> bool:
    """Return whether dormant-slot presence was dynamically proven for this session."""

    sampling = manifest.get("sampling")
    if not isinstance(sampling, dict) or sampling.get("presence_species_validated") is not True:
        return False
    offset = sampling.get("presence_species_offset")
    return bool(
        isinstance(offset, int)
        and not isinstance(offset, bool)
        and offset >= 0
        and offset % 4 == 0
    )


def allows_direct_movement_labels(
    manifest: dict[str, object],
    *,
    recording_hash: str | None = None,
    registry_path: str | Path = DEFAULT_PROVENANCE_REGISTRY,
) -> bool:
    """Require an explicit recorder declaration before exporting movement labels.

    Current-format archives embed this directly. Archives predating the
    embedded ``recording_provenance`` block fall back to the legacy external
    attestation registry (``legacy.manifest_compat``)."""

    provenance = manifest.get("recording_provenance")
    embedded = bool(
        isinstance(provenance, dict)
        and provenance.get("recording_role") == DIRECT_KEYBOARD_RECORDING_ROLE
        and provenance.get("movement_control_scheme") == DIRECT_KEYBOARD_CONTROL_SCHEME
        and provenance.get("direct_movement_labels_allowed") is True
    )
    if embedded:
        return True
    return attested_by_registry(
        recording_hash=recording_hash,
        required_role=DIRECT_KEYBOARD_RECORDING_ROLE,
        required_scheme=DIRECT_KEYBOARD_CONTROL_SCHEME,
        registry_path=registry_path,
    )


def direct_movement_provenance_source(
    manifest: dict[str, object],
    *,
    recording_hash: str | None = None,
    registry_path: str | Path = DEFAULT_PROVENANCE_REGISTRY,
) -> str | None:
    if allows_direct_movement_labels(manifest):
        return "embedded_manifest"
    if allows_direct_movement_labels(
        manifest,
        recording_hash=recording_hash,
        registry_path=registry_path,
    ):
        return "sha256_attestation_registry"
    return None


def recording_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def unique_recording_paths(paths) -> tuple[Path, ...]:
    """Reject the same archive content appearing more than once in one job."""

    result: list[Path] = []
    seen: dict[str, Path] = {}
    for raw_path in paths:
        path = Path(raw_path)
        value = recording_sha256(path)
        previous = seen.get(value)
        if previous is not None:
            raise ValueError(
                f"Duplicate recording content supplied: {path} duplicates {previous}"
            )
        seen[value] = path
        result.append(path)
    return tuple(result)


def validate_recording_contract(
    archive: "RecordingArchive",
    *,
    action_names: tuple[str, ...],
    origin_native_x: float,
    origin_native_z: float,
    native_units_per_cell: float,
    observation_schema_id: str | None = None,
    observation_schema_hash: str | None = None,
) -> tuple[str, ...]:
    """Validate explicit provenance while keeping schema-2 legacy data usable."""

    warnings: list[str] = []
    policy = archive.manifest.get("policy_contract")
    if policy is None:
        warnings.append(missing_policy_contract_warning(archive.path.name))
    elif not isinstance(policy, dict):
        raise ValueError(f"{archive.path}: policy_contract must be an object")
    else:
        recorded_actions = policy.get("action_names")
        if recorded_actions != list(action_names):
            raise ValueError(
                f"{archive.path}: action contract mismatch; got "
                f"{recorded_actions!r}, expected {list(action_names)!r}"
            )
        if observation_schema_id is not None and (
            policy.get("observation_schema_id") != observation_schema_id
        ):
            raise ValueError(
                f"{archive.path}: observation schema ID does not match "
                f"{observation_schema_id}"
            )
        if observation_schema_hash is not None and str(
            policy.get("observation_schema_hash", "")
        ).upper() != observation_schema_hash.upper():
            raise ValueError(
                f"{archive.path}: observation schema hash does not match the "
                "current export contract"
            )

    mapping = archive.manifest.get("map_contract")
    if mapping is None:
        warnings.append(missing_map_contract_warning(archive.path.name))
    elif not isinstance(mapping, dict):
        raise ValueError(f"{archive.path}: map_contract must be an object")
    else:
        expected = (
            float(origin_native_x),
            float(origin_native_z),
            float(native_units_per_cell),
        )
        try:
            recorded = (
                float(mapping["origin_native_x"]),
                float(mapping["origin_native_z"]),
                float(mapping["native_units_per_cell"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{archive.path}: invalid map_contract coordinate frame"
            ) from error
        if not all(
            math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1.0e-9)
            for actual, wanted in zip(recorded, expected, strict=True)
        ):
            raise ValueError(
                f"{archive.path}: coordinate-frame mismatch; got {recorded}, "
                f"expected {expected}"
            )
    return tuple(warnings)


@dataclass(frozen=True, slots=True)
class RecordedActor:
    base: int
    species: int
    hp: int
    x: float
    y: float
    z: float
    state_code: int

    @property
    def living(self) -> bool:
        return self.state_code == 1 and self.hp > 0 and self.species > 0


@dataclass(frozen=True, slots=True)
class RecordedFrame:
    sequence: int
    elapsed_ms: int
    phase: int
    player_hp: int
    player_x: float
    player_y: float
    player_z: float
    heading_radians: float
    focused: bool
    key_mask: int
    action: int
    keyframe: bool
    cached_actor_slots: int
    living_monsters: int
    unreadable_slots: int
    actors: tuple[RecordedActor, ...]


@dataclass(frozen=True, slots=True)
class RecordedEvent:
    kind: str
    values: tuple[object, ...]


class RecordingArchive:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        with zipfile.ZipFile(self.path) as archive:
            members = set(archive.namelist())
            missing = REQUIRED_ARCHIVE_MEMBERS.difference(members)
            if missing:
                raise ValueError(
                    f"Recording archive is missing required files: {', '.join(sorted(missing))}"
                )
            self.manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        if not isinstance(self.manifest, dict):
            raise ValueError("Recording manifest must be a JSON object")
        schema_version = self.manifest.get("schema_version")
        if schema_version not in SUPPORTED_RECORDING_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported recording schema_version {schema_version!r}; "
                f"supported versions are {sorted(SUPPORTED_RECORDING_SCHEMA_VERSIONS)}"
            )
        try:
            self.quantum = float(
                self.manifest["sampling"]["position_quantum_native"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Recording manifest has no valid sampling.position_quantum_native"
            ) from error
        if not math.isfinite(self.quantum) or self.quantum <= 0.0:
            raise ValueError(
                "Recording position_quantum_native must be finite and positive"
            )

    def _stream(self, member: str):
        archive = zipfile.ZipFile(self.path)
        raw = archive.open(member)
        gzip_file = gzip.GzipFile(fileobj=raw, mode="rb")
        unpacker = msgpack.Unpacker(gzip_file, raw=False, strict_map_key=False)
        try:
            for value in unpacker:
                yield value
        finally:
            gzip_file.close()
            raw.close()
            archive.close()

    def frames(self) -> Iterator[RecordedFrame]:
        actor_state: dict[int, tuple[int, int, int, int, int, int]] = {}
        first = True
        for record in self._stream("frames.msgpack.gz"):
            if first:
                first = False
                if not isinstance(record, dict) or record.get("type") != "header":
                    raise ValueError("frames stream is missing its header")
                continue
            if not isinstance(record, list) or not record or record[0] != "frame":
                continue
            (
                _tag,
                sequence,
                elapsed_ms,
                phase,
                player_hp,
                player_x_q,
                player_y_q,
                player_z_q,
                heading_milliradians,
                focused,
                key_mask,
                action,
                keyframe,
                updates,
                cached_actor_slots,
                living_monsters,
                unreadable_slots,
            ) = record
            if keyframe:
                actor_state.clear()
            for update in updates:
                base, species, hp, x_q, y_q, z_q, state_code = update
                actor_state[int(base)] = (
                    int(species), int(hp), int(x_q), int(y_q), int(z_q), int(state_code)
                )
            actors = tuple(
                RecordedActor(
                    base=base,
                    species=value[0],
                    hp=value[1],
                    x=value[2] * self.quantum,
                    y=value[3] * self.quantum,
                    z=value[4] * self.quantum,
                    state_code=value[5],
                )
                for base, value in sorted(actor_state.items())
            )
            yield RecordedFrame(
                sequence=int(sequence),
                elapsed_ms=int(elapsed_ms),
                phase=int(phase),
                player_hp=int(player_hp),
                player_x=float(player_x_q) * self.quantum,
                player_y=float(player_y_q) * self.quantum,
                player_z=float(player_z_q) * self.quantum,
                heading_radians=float(heading_milliradians) / 1000.0,
                focused=bool(focused),
                key_mask=int(key_mask),
                action=int(action),
                keyframe=bool(keyframe),
                cached_actor_slots=int(cached_actor_slots),
                living_monsters=int(living_monsters),
                unreadable_slots=int(unreadable_slots),
                actors=actors,
            )

    def events(self) -> Iterator[RecordedEvent]:
        first = True
        for record in self._stream("events.msgpack.gz"):
            if first:
                first = False
                if not isinstance(record, dict) or record.get("type") != "header":
                    raise ValueError("events stream is missing its header")
                continue
            if isinstance(record, list) and record:
                yield RecordedEvent(str(record[0]), tuple(record[1:]))

    def inputs(self) -> Iterator[tuple[int, bool, int, int]]:
        first = True
        for record in self._stream("inputs.msgpack.gz"):
            if first:
                first = False
                continue
            if isinstance(record, list) and len(record) == 5 and record[0] == "input":
                yield int(record[1]), bool(record[2]), int(record[3]), int(record[4])
