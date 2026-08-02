from __future__ import annotations

import json
import os
import zipfile
from datetime import date, datetime
from enum import Enum
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from time import time_ns
from typing import Protocol

import numpy as np

from .model_contract import ModelContractMetadata, sha256_file


class SavableModel(Protocol):
    def save(self, path: str) -> None: ...




def _json_key(value: object) -> str:
    """Return a stable JSON object key for non-string diagnostic keys."""

    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return str(value)
    return repr(value)


def to_json_safe(value: object, *, _seen: set[int] | None = None) -> object:
    """Convert nested runtime diagnostics into values accepted by ``json``.

    Stable-Baselines and Gymnasium info dictionaries may contain NumPy arrays
    (notably ``terminal_observation``), NumPy scalar values, paths, enums, and
    tuples. Session reporting is a best-effort diagnostic path and must never
    turn a successfully saved training model into a failed worker merely
    because one diagnostic value is not directly JSON serializable.
    """

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, np.generic):
        return to_json_safe(value.item(), _seen=_seen)

    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist(), _seen=_seen)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Enum):
        return to_json_safe(value.value, _seen=_seen)

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    seen = _seen if _seen is not None else set()
    identity = id(value)
    if identity in seen:
        return "<recursive-reference>"

    if is_dataclass(value) and not isinstance(value, type):
        seen.add(identity)
        try:
            return to_json_safe(asdict(value), _seen=seen)
        finally:
            seen.discard(identity)

    if isinstance(value, Mapping):
        seen.add(identity)
        try:
            return {
                _json_key(key): to_json_safe(item, _seen=seen)
                for key, item in value.items()
            }
        finally:
            seen.discard(identity)

    if isinstance(value, (list, tuple)):
        seen.add(identity)
        try:
            return [to_json_safe(item, _seen=seen) for item in value]
        finally:
            seen.discard(identity)

    if isinstance(value, (set, frozenset)):
        seen.add(identity)
        try:
            ordered = sorted(value, key=repr)
            return [to_json_safe(item, _seen=seen) for item in ordered]
        finally:
            seen.discard(identity)

    # Reporting is diagnostic and must not abort training teardown. Preserve an
    # intelligible representation rather than raising for an incidental type.
    return repr(value)


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    path: str
    sha256: str
    size_bytes: int


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


def _temporary_path(destination: Path, suffix: str) -> Path:
    return destination.with_name(
        f".{destination.name}.{os.getpid()}.{time_ns()}.tmp{suffix}"
    )


def atomic_write_json(
    destination: str | Path,
    payload: Mapping[str, object],
    *,
    replace: Callable[
        [
            str | bytes | os.PathLike[str] | os.PathLike[bytes],
            str | bytes | os.PathLike[str] | os.PathLike[bytes],
        ],
        None,
    ] = os.replace,
) -> ArtifactRecord:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path, ".json")
    serializable = to_json_safe(dict(payload))
    if not isinstance(serializable, dict):
        raise TypeError("JSON report payload must serialize to an object")
    encoded = (json.dumps(serializable, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        json.loads(temporary.read_text(encoding="utf-8"))
        replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return ArtifactRecord(str(path), sha256_file(path), path.stat().st_size)


def atomic_save_model(
    model: SavableModel,
    destination: str | Path,
    *,
    metadata: ModelContractMetadata | None = None,
    replace: Callable[
        [
            str | bytes | os.PathLike[str] | os.PathLike[bytes],
            str | bytes | os.PathLike[str] | os.PathLike[bytes],
        ],
        None,
    ] = os.replace,
) -> ArtifactRecord:
    path = Path(destination)
    if path.suffix.lower() != ".zip":
        path = path.with_suffix(".zip")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path, ".zip")
    contract = metadata or ModelContractMetadata.current()
    setattr(model, "farming_contract_metadata", contract.as_dict())
    try:
        model.save(str(temporary))
        if not temporary.is_file():
            raise RuntimeError("Model save did not create the requested ZIP artifact")
        with zipfile.ZipFile(temporary, "r") as archive:
            corrupt = archive.testzip()
            if corrupt is not None:
                raise RuntimeError(f"Model ZIP failed validation at {corrupt!r}")
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return ArtifactRecord(str(path), sha256_file(path), path.stat().st_size)


def atomic_copy_artifact(
    source: str | Path,
    destination: str | Path,
    *,
    replace: Callable[
        [
            str | bytes | os.PathLike[str] | os.PathLike[bytes],
            str | bytes | os.PathLike[str] | os.PathLike[bytes],
        ],
        None,
    ] = os.replace,
) -> ArtifactRecord:
    """Atomically publish a byte-for-byte copy of a validated artifact."""

    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination_path, destination_path.suffix or ".bin")
    try:
        with source_path.open("rb") as reader, temporary.open("xb") as writer:
            while chunk := reader.read(1024 * 1024):
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        replace(temporary, destination_path)
        _fsync_directory(destination_path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return ArtifactRecord(
        str(destination_path),
        sha256_file(destination_path),
        destination_path.stat().st_size,
    )


@dataclass(frozen=True, slots=True)
class SessionArtifacts:
    model: ArtifactRecord
    report: ArtifactRecord
    manifest: ArtifactRecord


def save_session_artifacts(
    model: SavableModel,
    *,
    model_path: str | Path,
    report_path: str | Path,
    manifest_path: str | Path,
    report: Mapping[str, object],
) -> SessionArtifacts:
    """Publish validated model/report artifacts, then one recovery manifest."""

    model_record = atomic_save_model(model, model_path)
    report_record = atomic_write_json(report_path, report)
    manifest_record = atomic_write_json(
        manifest_path,
        {
            "version": 1,
            "model": asdict(model_record),
            "report": asdict(report_record),
        },
    )
    return SessionArtifacts(model_record, report_record, manifest_record)
