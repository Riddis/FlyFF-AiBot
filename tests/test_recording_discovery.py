"""Malformed-archive discovery behavior (this pass's remediation section
13): a corrupt/unreadable *.zip under a scanned directory must never
silently disappear, indistinguishable from "no recording existed" --
it must be logged with its path + reason, and strict callers must be
able to fail loudly instead."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import pytest

from simulator.recording_discovery import (
    RecordingDiscoveryError,
    discover_direct_demonstration_eligible,
    discover_eva_only_supplementary,
    discover_world_model_eligible,
)


def _write_garbage_zip(path: Path) -> None:
    path.write_bytes(b"not actually a zip file at all")


def _write_zip_missing_manifest(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("frames.msgpack.gz", b"")
        archive.writestr("events.msgpack.gz", b"")
        archive.writestr("inputs.msgpack.gz", b"")


@pytest.mark.parametrize(
    "writer", [_write_garbage_zip, _write_zip_missing_manifest]
)
@pytest.mark.parametrize(
    "discover",
    [
        discover_world_model_eligible,
        discover_direct_demonstration_eligible,
        discover_eva_only_supplementary,
    ],
)
def test_malformed_archive_is_logged_and_skipped_by_default(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, writer, discover
) -> None:
    directory = tmp_path / "recordings"
    directory.mkdir()
    bad_path = directory / "corrupt.zip"
    writer(bad_path)

    with caplog.at_level(logging.WARNING, logger="simulator.recording_discovery"):
        result = discover([directory])

    assert result == []
    assert any(str(bad_path) in record.message for record in caplog.records), (
        "the corrupt archive's own path must appear in a logged warning, "
        "not disappear silently"
    )


@pytest.mark.parametrize(
    "writer", [_write_garbage_zip, _write_zip_missing_manifest]
)
@pytest.mark.parametrize(
    "discover",
    [
        discover_world_model_eligible,
        discover_direct_demonstration_eligible,
        discover_eva_only_supplementary,
    ],
)
def test_malformed_archive_raises_in_strict_mode(
    tmp_path: Path, writer, discover
) -> None:
    directory = tmp_path / "recordings"
    directory.mkdir()
    bad_path = directory / "corrupt.zip"
    writer(bad_path)

    with pytest.raises(RecordingDiscoveryError, match=r"corrupt\.zip"):
        discover([directory], strict=True)


def test_a_non_zip_file_without_zip_extension_is_never_scanned(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Non-recording files sitting alongside real archives (readme.txt,
    a thumbnail, etc.) must not flood the log -- only *.zip is ever
    considered a candidate archive at all."""
    directory = tmp_path / "recordings"
    directory.mkdir()
    (directory / "README.txt").write_text("not a recording", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="simulator.recording_discovery"):
        result = discover_world_model_eligible([directory])

    assert result == []
    assert caplog.records == []
