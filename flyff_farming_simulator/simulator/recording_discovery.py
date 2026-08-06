"""Discover recording archives by eligibility, so pipelines consider every
currently-classified recording instead of one hardcoded path.

This deliberately depends only on ``simulator.schema`` (not on
flyff_farming_recorder), so it stays importable from the core training CLI
without a cross-project import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from farming.actions import FarmingAction

from .schema import (
    RecordingArchive,
    allows_direct_movement_labels,
    has_validated_presence,
    recording_sha256,
)


def iter_archive_paths(directories: Iterable[str | Path]) -> list[Path]:
    """Every ``*.zip`` directly inside each directory, sorted and de-duplicated.

    Does not recurse -- a directory containing further subdirectories (e.g.
    an originals-backup folder placed alongside classified archives) is not
    scanned beneath its top level.
    """

    found: dict[str, Path] = {}
    for raw in directories:
        directory = Path(raw)
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.zip")):
            found[str(path.resolve())] = path
    return [found[key] for key in sorted(found)]


def discover_world_model_eligible(directories: Iterable[str | Path]) -> list[Path]:
    """Archives with a dynamically validated (or explicitly attested) presence field.

    Eligibility is independent of movement classification -- a click-to-move
    session can be world-model-eligible, and a keyboard_wasd session can be
    world-model-ineligible. Never assume one folder holds only one kind.
    """

    eligible: list[Path] = []
    for path in iter_archive_paths(directories):
        try:
            archive = RecordingArchive(path)
        except Exception:
            continue
        if has_validated_presence(archive.manifest):
            eligible.append(path)
    return eligible


def discover_direct_demonstration_eligible(directories: Iterable[str | Path]) -> list[Path]:
    """Archives whose movement is explicitly eligible for steering supervision.

    Requires movement_classification == keyboard_wasd AND explicit provenance
    (embedded recording_role, or a hash-pinned recording_provenance.json
    entry) -- never granted from retroactive classification confidence alone.
    """

    eligible: list[Path] = []
    for path in iter_archive_paths(directories):
        try:
            archive = RecordingArchive(path)
        except Exception:
            continue
        sha256 = recording_sha256(path)
        if allows_direct_movement_labels(archive.manifest, recording_hash=sha256):
            eligible.append(path)
    return eligible


def discover_eva_only_supplementary(
    directories: Iterable[str | Path], *, exclude: Iterable[str | Path] = ()
) -> list[Path]:
    """Archives with real EVA presses that are not demonstration-eligible.

    These may supervise the event head only -- their movement is
    click-to-move, mixed, or otherwise unattested and must never supervise
    steering. Pass the result of ``discover_direct_demonstration_eligible``
    as ``exclude`` so a single archive is never double-counted as both.
    """

    excluded = {str(Path(item).resolve()) for item in exclude}
    eligible: list[Path] = []
    for path in iter_archive_paths(directories):
        if str(path.resolve()) in excluded:
            continue
        try:
            archive = RecordingArchive(path)
            has_eva = any(
                frame.action == int(FarmingAction.CAST_EVA) for frame in archive.frames()
            )
        except Exception:
            continue
        if has_eva:
            eligible.append(path)
    return eligible
