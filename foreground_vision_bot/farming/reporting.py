from __future__ import annotations

import json
import os
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from time import time_ns
from typing import Protocol

from .model_contract import ModelContractMetadata, sha256_file


class SavableModel(Protocol):
    def save(self, path: str) -> None: ...


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
    encoded = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode(
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
