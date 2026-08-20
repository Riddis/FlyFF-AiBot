"""Post-hoc evidence catalog (docs/PROJECT_GOALS.md section 6).

Purpose/controller/data-use classification is a scientific concept, not
a capture-time UI burden: raw recording is the primary immutable
evidence, and interpretation is attached AFTERWARD, in a separate
sidecar file, never by mutating the raw archive. Claude/Codex or the
user already know why a given controlled test was run at the time they
ran it -- this module lets that be recorded once, after the fact,
against a specific archive.

Used for BOTH the dev bot's own recordings (recording_sink.py's
RecordingSink) and the standalone recorder's (recorder/session.py's
RecorderController) -- neither writes provenance interpretation into
its own capture path; this is the one place it is attached."""

from __future__ import annotations

import json
from pathlib import Path

from recording_format import atomic_json
from .provenance import ExperimentProvenance


def sidecar_path(archive_path: Path) -> Path:
    return archive_path.with_suffix(archive_path.suffix + ".evidence.json")


def attach_evidence_label(
    archive_path: Path,
    provenance: ExperimentProvenance,
    *,
    labeled_by: str = "agent",
    note: str | None = None,
) -> Path:
    """Writes/overwrites a sidecar JSON next to ``archive_path`` -- the
    raw archive itself is never opened or modified. Safe to call on an
    archive that does not exist yet (recording still in progress) or
    that has already been labeled (overwrites the label, not the
    archive)."""
    payload = {
        "archive": str(archive_path),
        "labeled_by": labeled_by,
        "note": note,
        **provenance.to_dict(),
    }
    path = sidecar_path(archive_path)
    atomic_json(path, payload)
    return path


def read_evidence_label(archive_path: Path) -> dict | None:
    path = sidecar_path(archive_path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
