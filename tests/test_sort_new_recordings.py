"""Recording index portability (this pass's remediation section 14):
recordings/INDEX.json must never bake in a user-machine absolute
source path -- it is tracked metadata (filename/hash/classification),
not a runtime source of truth. Local source-directory discovery stays
an explicit runtime/tool input."""

from __future__ import annotations

import gzip
import json
import sys
import zipfile
from pathlib import Path

import msgpack
import pytest

from devtools.archives import sort_new_recordings


def _write_stream(path: Path, values: list[object]) -> None:
    packer = msgpack.Packer(use_bin_type=True)
    with gzip.open(path, "wb") as handle:
        for value in values:
            handle.write(packer.pack(value))


def _build_minimal_archive(directory: Path, name: str) -> Path:
    session = directory / f".{name}_session"
    session.mkdir()
    manifest = {
        "schema_version": 2,
        "recorder_version": "1.11.0",
        "sampling": {"position_quantum_native": 0.05},
    }
    (session / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for member in ("frames", "events", "inputs"):
        _write_stream(session / f"{member}.msgpack.gz", [{"type": "header"}])
    zip_path = directory / f"{name}.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for path in session.iterdir():
            archive.write(path, path.name)
    return zip_path


def test_index_json_never_contains_an_absolute_source_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recordings_root = tmp_path / "recordings"
    training_dir = recordings_root / "training"
    training_dir.mkdir(parents=True)
    _build_minimal_archive(training_dir, "sample")

    monkeypatch.setattr(sys, "argv", ["sort_new_recordings.py", str(recordings_root)])
    exit_code = sort_new_recordings.main()
    assert exit_code == 0

    index_path = recordings_root / "INDEX.json"
    entries = json.loads(index_path.read_text(encoding="utf-8"))
    assert len(entries) == 1
    entry = entries[0]

    assert "path" not in entry
    assert entry["bucket"] == "training"
    assert entry["filename"] == "sample.zip"

    raw_text = index_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in raw_text
    assert str(recordings_root) not in raw_text


def test_index_json_error_entries_also_never_contain_an_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recordings_root = tmp_path / "recordings"
    training_dir = recordings_root / "training"
    training_dir.mkdir(parents=True)
    (training_dir / "corrupt.zip").write_bytes(b"not a zip file")

    monkeypatch.setattr(sys, "argv", ["sort_new_recordings.py", str(recordings_root)])
    exit_code = sort_new_recordings.main()
    assert exit_code == 0

    index_path = recordings_root / "INDEX.json"
    entries = json.loads(index_path.read_text(encoding="utf-8"))
    assert len(entries) == 1
    entry = entries[0]

    assert "path" not in entry
    assert entry["bucket"] == "training"
    assert entry["filename"] == "corrupt.zip"
    assert "error" in entry
