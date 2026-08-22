"""Discover recording archives by eligibility, so pipelines consider every
currently-classified recording instead of one hardcoded path.

This deliberately depends only on ``simulator.schema`` (not on
flyff_farming_recorder), so it stays importable from the core training CLI
without a cross-project import.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from farming.actions import FarmingAction

from .schema import (
    RecordingArchive,
    allows_direct_movement_labels,
    has_validated_presence,
    recording_sha256,
)

logger = logging.getLogger(__name__)


class RecordingDiscoveryError(RuntimeError):
    """A ``*.zip`` under a scanned directory could not be opened/read as a
    recording archive. Raised by every discovery function's ``strict=True``
    mode -- training/dataset preparation that cannot tolerate silently
    losing an archive should pass it rather than let a corrupt/unreadable
    file disappear indistinguishably from "no recording existed"."""


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


def _open_archive_or_skip(path: Path, *, strict: bool) -> RecordingArchive | None:
    """Open ``path`` as a RecordingArchive, or handle the failure per
    ``strict``: log a clear warning (path + reason) and return ``None`` when
    lenient, or raise ``RecordingDiscoveryError`` when strict. Never a bare
    silent ``continue`` -- a corrupt archive, a permission error, and a
    genuinely-missing recording must never look identical."""

    try:
        return RecordingArchive(path)
    except Exception as error:  # noqa: BLE001 - reported below, never silent.
        message = f"{path}: could not be opened as a recording archive ({type(error).__name__}: {error})"
        if strict:
            raise RecordingDiscoveryError(message) from error
        logger.warning(message)
        return None


def discover_world_model_eligible(
    directories: Iterable[str | Path], *, strict: bool = False
) -> list[Path]:
    """Archives with a dynamically validated (or explicitly attested) presence field.

    Eligibility is independent of movement classification -- a click-to-move
    session can be world-model-eligible, and a keyboard_wasd session can be
    world-model-ineligible. Never assume one folder holds only one kind.

    ``strict=True`` raises ``RecordingDiscoveryError`` on the first archive
    that cannot be opened, instead of logging a warning and skipping it.
    """

    eligible: list[Path] = []
    for path in iter_archive_paths(directories):
        archive = _open_archive_or_skip(path, strict=strict)
        if archive is None:
            continue
        if has_validated_presence(archive.manifest):
            eligible.append(path)
    return eligible


def discover_direct_demonstration_eligible(
    directories: Iterable[str | Path], *, strict: bool = False
) -> list[Path]:
    """Archives whose movement is explicitly eligible for steering supervision.

    Requires movement_classification == keyboard_wasd AND explicit provenance
    (embedded recording_role, or a hash-pinned recording_provenance.json
    entry) -- never granted from retroactive classification confidence alone.

    ``strict=True`` raises ``RecordingDiscoveryError`` on the first archive
    that cannot be opened, instead of logging a warning and skipping it.
    """

    eligible: list[Path] = []
    for path in iter_archive_paths(directories):
        archive = _open_archive_or_skip(path, strict=strict)
        if archive is None:
            continue
        sha256 = recording_sha256(path)
        if allows_direct_movement_labels(archive.manifest, recording_hash=sha256):
            eligible.append(path)
    return eligible


def discover_eva_only_supplementary(
    directories: Iterable[str | Path],
    *,
    exclude: Iterable[str | Path] = (),
    strict: bool = False,
) -> list[Path]:
    """Archives with real EVA presses that are not demonstration-eligible.

    These may supervise the event head only -- their movement is
    click-to-move, mixed, or otherwise unattested and must never supervise
    steering. Pass the result of ``discover_direct_demonstration_eligible``
    as ``exclude`` so a single archive is never double-counted as both.

    ``strict=True`` raises ``RecordingDiscoveryError`` on the first archive
    that cannot be opened or scanned, instead of logging a warning and
    skipping it.
    """

    excluded = {str(Path(item).resolve()) for item in exclude}
    eligible: list[Path] = []
    for path in iter_archive_paths(directories):
        if str(path.resolve()) in excluded:
            continue
        archive = _open_archive_or_skip(path, strict=strict)
        if archive is None:
            continue
        try:
            has_eva = any(
                frame.action == int(FarmingAction.CAST_EVA) for frame in archive.frames()
            )
        except Exception as error:  # noqa: BLE001 - reported below, never silent.
            message = (
                f"{path}: could not be scanned for EVA frames "
                f"({type(error).__name__}: {error})"
            )
            if strict:
                raise RecordingDiscoveryError(message) from error
            logger.warning(message)
            continue
        if has_eva:
            eligible.append(path)
    return eligible
