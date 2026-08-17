"""One resolved context for repository-owned paths and a per-launch
correlation identity, so Phase-10 entrypoints and the specialist process
orchestrator (``devtools/processes.py``) stop hard-coding relative-path
assumptions of their own.

This module only *resolves* paths that already exist at their Phase-0-
authoritative locations -- it never physically relocates a scientific
artifact directory, and it never invents a directory scheme broader than
what current entrypoints actually need (models, map assets, recordings,
evaluations, and the telemetry session output directory
``tools/run_observation_telemetry.py``/``apps/telemetry_cli.py`` already
default to).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def _repo_root() -> Path:
    # devtools/session_context.py -> devtools/ -> repository root.
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class SessionContext:
    """Canonical repository-relative paths, resolved once. Every field is
    an existing-or-creatable directory already used by current tooling --
    nothing here is a new scientific-artifact location."""

    repo_root: Path
    models_dir: Path
    map_assets_dir: Path
    recordings_dir: Path
    evaluations_dir: Path
    telemetry_sessions_dir: Path

    def ensure_output_dirs(self) -> None:
        """Create only the directories a launcher might need to write
        into (telemetry sessions). Never creates or touches models/,
        map_assets/, recordings/, or evaluations/ -- those are read from,
        not written into, by devtools launchers."""
        self.telemetry_sessions_dir.mkdir(parents=True, exist_ok=True)


def resolve_session_context(repo_root: Path | None = None) -> SessionContext:
    root = (repo_root or _repo_root()).resolve()
    return SessionContext(
        repo_root=root,
        models_dir=root / "models",
        map_assets_dir=root / "map_assets",
        recordings_dir=root / "recordings",
        evaluations_dir=root / "evaluations",
        telemetry_sessions_dir=root / "telemetry_sessions",
    )


@dataclass(frozen=True, slots=True)
class LaunchIdentity:
    """A lightweight orchestration identity for correlating one specialist
    subprocess launch with the dev session that started it -- deliberately
    NOT a replacement for any specialist's own persisted provenance (e.g.
    farming.telemetry's TelemetrySessionProvenance, recorder's own session
    metadata). No archive schema is bumped and no persisted scientific
    record is altered to carry this; it exists for the launcher's own
    session/status log only."""

    session_id: str
    started_at_utc: str
    git_head: str | None
    git_dirty: bool | None


def new_launch_identity(repo_root: Path | None = None) -> LaunchIdentity:
    root = (repo_root or _repo_root()).resolve()
    head, dirty = _git_provenance(root)
    return LaunchIdentity(
        session_id=str(uuid.uuid4()),
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        git_head=head,
        git_dirty=dirty,
    )


def _git_provenance(repo_root: Path) -> tuple[str | None, bool | None]:
    """Best-effort only, matching farming/telemetry.py's own
    build_session_provenance convention -- never fails a launch."""
    import subprocess

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root), capture_output=True, text=True, timeout=5.0, check=True,
        ).stdout.strip()
    except Exception:
        return None, None
    dirty: bool | None
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root), capture_output=True, text=True, timeout=5.0, check=True,
        ).stdout
        dirty = bool(status.strip())
    except Exception:
        dirty = None
    return (commit or None), dirty
