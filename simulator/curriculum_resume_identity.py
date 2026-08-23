"""Resume-identity gate for the canonical Beginner/Intermediate/Advanced
curriculum runners (`simulator/tools/RUN_CANONICAL_*.py`).

Each runner persists progress across restarts via plain JSON files on disk
(a `canonical_<stage>_run_summary.json` round list, a zero-shot diagnostic
cache, per-round pre-/post-rehearsal evaluation caches) and previously
resumed from ANY such file purely because it existed -- no check that its
content actually belonged to the CURRENT learned-target-selection lineage
(`MultiDiscrete([13, 3])`, raw 923-value observation, target selection as a
learned action) rather than an incompatible historical one (steering+event,
event-only, or any earlier action/observation contract). A restart could
silently inherit `consecutive_passes`, a `carried_forward_checkpoint` path,
or a cached evaluation result from a lineage that no longer means the same
thing as the current one.

This module defines the minimum identity a stored artifact must carry to be
trusted for resume, and the load helpers that apply it. Deliberately NOT
tied to `git.commit` (unlike `run_provenance.build_run_manifest`'s
per-checkpoint manifest) -- binding resume validity to every source commit
would make legitimate resume impossible after a documentation-only change;
only the parts of the contract that actually change what a stored artifact
MEANS are checked.

Never mutates or deletes a file whose identity doesn't match -- see
`archive_legacy_artifact` below and docs/agent/PROJECT_RULES.md section 7.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

CURRENT_GENERATION_ID = "learned-target-event-multidiscrete-13x3-v1"


def current_curriculum_generation_identity(*, declared_parent_checkpoint: str | Path | None = None) -> dict[str, Any]:
    """The exact facts a stored round report / cached evaluation must match
    to be trusted as belonging to the CURRENT curriculum lineage.

    `declared_parent_checkpoint`, when given, is the stage's own declared
    entry-point checkpoint (e.g. `GRADUATED_BASIC_CHECKPOINT` for Beginner)
    -- a resolved-path identity, not a content hash, cheap to compare on
    every resume without re-hashing a (potentially large) checkpoint file.
    Round reports across an entire resumable chain are stamped with the
    SAME declared parent (the chain's origin), so a stale chain that
    started from a since-repointed or different-lineage parent is rejected
    even if its action/observation contract happens to still match."""
    from farming.actions import FarmingEvent
    from farming.observation import DIRECT_ACTOR_SLOTS
    from navigation.navigation_evidence import RAW_OBSERVATION_SIZE

    from .navigation_subpolicy import FROZEN_NAVIGATION_CHECKPOINT_SHA256

    identity: dict[str, Any] = {
        "generation_id": CURRENT_GENERATION_ID,
        "policy_action_nvec": [DIRECT_ACTOR_SLOTS + 1, len(FarmingEvent)],
        "raw_observation_size": RAW_OBSERVATION_SIZE,
        "navigation_checkpoint_sha256": FROZEN_NAVIGATION_CHECKPOINT_SHA256,
    }
    if declared_parent_checkpoint is not None:
        identity["declared_parent_checkpoint"] = str(Path(declared_parent_checkpoint).resolve())
    return identity


def identity_mismatch_reason(
    stored: dict[str, Any] | None, *, expected_declared_parent_checkpoint: str | Path | None = None,
) -> str | None:
    """Returns `None` if `stored` (a `generation_identity` dict previously
    written by `current_curriculum_generation_identity()`) matches the
    CURRENT curriculum generation; otherwise a human-readable reason it must
    NOT be trusted for resume/cache reuse. A missing/empty `stored` dict --
    the case for every artifact written before this identity scheme existed,
    including every tracked pre-2026-08-23 run summary -- is always a
    mismatch: absence of identity is never treated as an implicit match.

    When `expected_declared_parent_checkpoint` is given, `stored` must also
    carry a matching `declared_parent_checkpoint` -- catching a resumable
    chain that descends from a DIFFERENT parent checkpoint than this stage
    currently expects, even if its action/observation contract still
    matches (e.g. a chain accidentally started against a stale or
    quarantined Basic/Beginner/Intermediate graduated checkpoint)."""
    current = current_curriculum_generation_identity(declared_parent_checkpoint=expected_declared_parent_checkpoint)
    if not stored:
        return "no generation_identity recorded (pre-dates this identity scheme -- an untyped historical/legacy artifact)"
    if stored.get("generation_id") != current["generation_id"]:
        return f"generation_id={stored.get('generation_id')!r} does not match current {current['generation_id']!r}"
    if stored.get("policy_action_nvec") != current["policy_action_nvec"]:
        return f"policy_action_nvec={stored.get('policy_action_nvec')!r} does not match current {current['policy_action_nvec']!r}"
    if stored.get("raw_observation_size") != current["raw_observation_size"]:
        return f"raw_observation_size={stored.get('raw_observation_size')!r} does not match current {current['raw_observation_size']!r}"
    if stored.get("navigation_checkpoint_sha256") != current["navigation_checkpoint_sha256"]:
        return "navigation_checkpoint_sha256 does not match the current frozen navigation checkpoint"
    if expected_declared_parent_checkpoint is not None:
        if stored.get("declared_parent_checkpoint") != current["declared_parent_checkpoint"]:
            return (
                f"declared_parent_checkpoint={stored.get('declared_parent_checkpoint')!r} does not match "
                f"expected {current['declared_parent_checkpoint']!r}"
            )
    return None


def archive_legacy_artifact(path: Path, *, log: Callable[[str], None]) -> None:
    """Copies (never moves/deletes) a soon-to-be-superseded artifact aside,
    byte-identical, before the current lineage's own state starts writing to
    `path` again -- `path` itself is a live, continuously-rewritten resume
    artifact under NORMAL operation (every passing round rewrites the same
    summary file), so once its content is judged non-resumable the ordinary
    next write would otherwise silently destroy the only copy of the
    historical evidence it held. Never overwrites an existing archive."""
    if not path.exists():
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = path.with_name(f"{path.stem}.legacy-{timestamp}{path.suffix}")
    if archive_path.exists():
        return
    shutil.copy2(path, archive_path)
    log(f"Archived non-resumable legacy artifact {path} -> {archive_path} (original left untouched at its own path)")


def load_resumable_round_reports(
    summary_path: Path, *, log: Callable[[str], None], declared_parent_checkpoint: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Loads a stage's `canonical_<stage>_run_summary.json` round list.
    Returns it unchanged if its LAST round's `generation_identity` matches
    the current lineage (and, when `declared_parent_checkpoint` is given,
    the same declared parent); otherwise archives the file and returns `[]`
    (a fresh start for this stage's `consecutive_passes`/`current_checkpoint`
    state) without ever mutating the file's own path in place."""
    if not summary_path.exists():
        return []
    try:
        round_reports = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not round_reports:
        return []
    reason = identity_mismatch_reason(
        round_reports[-1].get("generation_identity"), expected_declared_parent_checkpoint=declared_parent_checkpoint,
    )
    if reason is not None:
        log(f"Ignoring existing run summary at {summary_path} for resume: {reason}. "
            "Starting this stage's consecutive_passes/current_checkpoint fresh from the declared parent checkpoint.")
        archive_legacy_artifact(summary_path, log=log)
        return []
    return round_reports


def load_cached_report_if_current(path: Path, *, log: Callable[[str], None]) -> dict[str, Any] | None:
    """Loads a cached evaluation report (zero-shot diagnostic, pre-/post-
    rehearsal evaluation) ONLY if its own `generation_identity` matches the
    current lineage. Returns `None` (never resume/reuse) on any mismatch,
    missing file, or unreadable content -- the caller is expected to
    recompute and then write a fresh report (via
    `with_current_generation_identity`) to the same path, which is normal
    operation for these continuously-refreshed caches, not a mutation of
    historical evidence -- but the stale content is archived first so the
    old evidence survives that overwrite."""
    if not path.exists():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    reason = identity_mismatch_reason(report.get("generation_identity") if isinstance(report, dict) else None)
    if reason is not None:
        log(f"Ignoring cached evaluation at {path}: {reason}. Recomputing.")
        archive_legacy_artifact(path, log=log)
        return None
    return report


def with_current_generation_identity(
    report: dict[str, Any], *, declared_parent_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Stamps a freshly-computed report/round-entry dict with the current
    curriculum generation identity before it is written to disk, so a later
    resume attempt can validate it. Returns a new dict; does not mutate the
    input in place."""
    return {
        **report,
        "generation_identity": current_curriculum_generation_identity(declared_parent_checkpoint=declared_parent_checkpoint),
    }
