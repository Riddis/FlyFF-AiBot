"""Resume-identity gate for the canonical Beginner/Intermediate/Advanced
curriculum runners (`simulator/tools/RUN_CANONICAL_*.py`).

Each runner persists progress across restarts via plain JSON files on disk
(a `canonical_<stage>_run_summary.json` round list, a zero-shot diagnostic
cache, per-round pre-/post-rehearsal evaluation caches) and previously
resumed from ANY such file purely because it existed -- no check that its
content actually belonged to the CURRENT learned-target-selection lineage,
and even after an earlier fix added an architecture-generation check, that
check alone was still insufficient: a cache from the SAME architecture
generation but a DIFFERENT checkpoint, parent, curriculum stage, manifest,
or evaluation configuration was still silently reusable.

Two layers close this gap:

1. **Path-level generation separation** (`current_generation_path`):
   current-generation outputs are written under a namespaced filename
   (`GENERATION_NAMESPACE`), distinct from the historical filename an
   earlier architecture generation used. Current code never reads or
   writes the historical path at all, so historical evidence is left
   alone by construction -- not by a read-then-reject-then-archive dance
   that could still end up overwriting it (the bug this replaces).

2. **Content-level identity validation** (`round_record_validity_reason`,
   `identity_mismatch_reason`): even within the current-generation
   namespace, a stored round record or cached evaluation must prove it
   was produced by the EXACT checkpoint/parent/stage/manifest/config the
   current run is about to use -- checkpoint identity is content-based
   (SHA-256), never path-alone, since a same-named file can hold different
   model bytes across restarts/quarantines.

No archive-copy mechanism: these JSON files are git-tracked (history
recoverable via git if ever needed), the namespaced paths already keep
different architecture generations from ever colliding, and a content-level
mismatch within one generation is a rare edge case, not the common path
this module exists to handle -- see docs/agent/PROJECT_RULES.md section 16
("prefer simpler semantics") and MISTAKES.md's 2026-08-23 entry on the
archive-then-overwrite bug this module's predecessor had.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

GENERATION_NAMESPACE = "target_event_v1"
CURRENT_GENERATION_ID = "learned-target-event-multidiscrete-13x3-v1"


def current_generation_path(path: Path | str) -> Path:
    """Inserts the current-generation namespace tag before the final
    suffix: `canonical_beginner_run_summary.json` ->
    `canonical_beginner_run_summary.target_event_v1.json`. This IS the
    "current target+event generation" output; the un-suffixed filename
    (if it exists) belongs to an earlier architecture generation and is
    never read or written by current code."""
    p = Path(path)
    return p.with_name(f"{p.stem}.{GENERATION_NAMESPACE}{p.suffix}")


def sha256_of_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(path: Path | str | None) -> dict[str, Any] | None:
    """`None` when `path` is `None` or does not (yet) exist on disk --
    represents "not yet materialized" explicitly, rather than a caller
    guessing from a directory glob."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return {"path": str(p.resolve()), "sha256": sha256_of_file(p)}


def manifest_identity(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    return {"path": str(p.resolve()), "content_sha256": sha256_of_file(p)}


def stable_config_fingerprint(config: dict[str, Any]) -> str:
    """Canonical JSON (sorted keys, stable separators) + SHA-256 -- avoids
    Python's unordered dict/set repr instability. Only include configuration
    that changes the MEANING of a run/evaluation (episode duration, seeds,
    manifest choice, recovery config if applicable), never irrelevant
    repository state."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _generation_core() -> dict[str, Any]:
    from farming.actions import FarmingEvent
    from farming.observation import DIRECT_ACTOR_SLOTS
    from navigation.navigation_evidence import RAW_OBSERVATION_SIZE

    from .navigation_subpolicy import FROZEN_NAVIGATION_CHECKPOINT_SHA256

    return {
        "generation_id": CURRENT_GENERATION_ID,
        "policy_action_nvec": [DIRECT_ACTOR_SLOTS + 1, len(FarmingEvent)],
        "raw_observation_size": RAW_OBSERVATION_SIZE,
        "navigation_checkpoint_sha256": FROZEN_NAVIGATION_CHECKPOINT_SHA256,
    }


def identity_mismatch_reason(stored: dict[str, Any] | None, expected: dict[str, Any]) -> str | None:
    """Returns `None` if `stored` carries every key `expected` names with a
    matching value; otherwise a human-readable reason it must NOT be
    trusted for resume/cache reuse. A missing/empty `stored` dict -- the
    case for every artifact written before this identity scheme existed --
    is always a mismatch: absence of identity is never an implicit match.
    Keys `stored` carries that `expected` does not name are ignored (lets
    round records carry their own extra `current_checkpoint*` fields that
    round-level validation checks separately, see
    `round_record_validity_reason`)."""
    if not stored:
        return "no identity recorded (pre-dates this identity scheme -- an untyped historical/legacy artifact)"
    for key, value in expected.items():
        if stored.get(key) != value:
            return f"{key}={stored.get(key)!r} does not match expected {value!r}"
    return None


def _round_expected_identity(*, stage: str, declared_parent_checkpoint: Path | str) -> dict[str, Any]:
    """The part of round identity known AHEAD of a resume decision --
    generation core, stage, and the declared parent checkpoint's own
    content identity. Deliberately excludes `current_checkpoint*`: that is
    the round record's OWN claim, self-validated against disk by
    `round_record_validity_reason`, not compared against a caller-supplied
    expectation (the caller does not know it in advance -- that is the
    entire point of resuming)."""
    parent_identity = checkpoint_identity(declared_parent_checkpoint)
    return {
        **_generation_core(),
        "curriculum_stage": stage,
        "declared_parent_checkpoint": (
            parent_identity["path"] if parent_identity else str(Path(declared_parent_checkpoint).resolve())
        ),
        "declared_parent_checkpoint_sha256": parent_identity["sha256"] if parent_identity else None,
    }


def round_identity(*, stage: str, declared_parent_checkpoint: Path | str, current_checkpoint: Path | str) -> dict[str, Any]:
    """Full identity to STAMP onto a freshly-written round record --
    includes the round's own current-checkpoint content identity, unlike
    `_round_expected_identity` (which a resume decision compares against)."""
    current_identity = checkpoint_identity(current_checkpoint)
    return {
        **_round_expected_identity(stage=stage, declared_parent_checkpoint=declared_parent_checkpoint),
        "current_checkpoint": (
            current_identity["path"] if current_identity else str(Path(current_checkpoint).resolve())
        ),
        "current_checkpoint_sha256": current_identity["sha256"] if current_identity else None,
    }


def round_record_validity_reason(record: dict[str, Any], *, stage: str, declared_parent_checkpoint: Path | str) -> str | None:
    """Returns `None` if `record` (one element of a resumable round list)
    may be trusted to resume from: matching generation/stage/declared
    parent (+ its live content identity), AND its own recorded
    `carried_forward_checkpoint` still exists on disk with UNCHANGED bytes
    (content-based, not path-alone -- a same-named file can hold different
    model bytes across restarts/quarantines). Covers every case section 8
    of the task requires: wrong parent, wrong parent bytes, wrong stage,
    wrong architecture, stale checkpoint path, changed checkpoint bytes."""
    expected = _round_expected_identity(stage=stage, declared_parent_checkpoint=declared_parent_checkpoint)
    stored = record.get("identity")
    reason = identity_mismatch_reason(stored, expected)
    if reason is not None:
        return reason
    checkpoint_path = record.get("carried_forward_checkpoint")
    recorded_sha = (stored or {}).get("current_checkpoint_sha256")
    if not checkpoint_path or not recorded_sha:
        return "round record is missing carried_forward_checkpoint or its recorded current_checkpoint_sha256"
    resolved = Path(checkpoint_path)
    if not resolved.exists():
        return f"carried_forward_checkpoint {checkpoint_path!r} no longer exists on disk (stale path -- no orphan-checkpoint resume)"
    live_sha = sha256_of_file(resolved)
    if live_sha != recorded_sha:
        return (
            f"carried_forward_checkpoint {checkpoint_path!r} bytes changed since this round was recorded "
            f"(recorded sha256 {recorded_sha[:12]}..., live {live_sha[:12]}...)"
        )
    return None


def evaluation_cache_identity(
    *,
    stage: str,
    declared_parent_checkpoint: Path | str,
    evaluated_checkpoint: Path | str,
    evaluation_role: str,
    manifests: dict[str, str],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Full identity for a cached evaluation report (zero-shot diagnostic,
    pre-/post-rehearsal heldout/unseen/challenge). `evaluation_role`
    disambiguates e.g. "heldout" from "challenge" for the SAME checkpoint
    (task section 4). `manifests` maps a short name (e.g. "heldout") to
    that manifest file's path -- content-hashed so a manifest edit
    invalidates cached results graded against the old version."""
    parent_identity = checkpoint_identity(declared_parent_checkpoint)
    evaluated_identity = checkpoint_identity(evaluated_checkpoint)
    return {
        **_generation_core(),
        "curriculum_stage": stage,
        "declared_parent_checkpoint": (
            parent_identity["path"] if parent_identity else str(Path(declared_parent_checkpoint).resolve())
        ),
        "declared_parent_checkpoint_sha256": parent_identity["sha256"] if parent_identity else None,
        "evaluated_checkpoint": (
            evaluated_identity["path"] if evaluated_identity else str(Path(evaluated_checkpoint).resolve())
        ),
        "evaluated_checkpoint_sha256": evaluated_identity["sha256"] if evaluated_identity else None,
        "evaluation_role": evaluation_role,
        "manifests": {name: manifest_identity(manifest_path) for name, manifest_path in sorted(manifests.items())},
        "evaluation_config_fingerprint": stable_config_fingerprint(config),
    }


def with_round_identity(
    report: dict[str, Any], *, stage: str, declared_parent_checkpoint: Path | str, current_checkpoint: Path | str,
) -> dict[str, Any]:
    """Stamps a freshly-computed round-report dict with its full identity
    before it is written to disk. Returns a new dict; does not mutate the
    input in place."""
    return {
        **report,
        "identity": round_identity(
            stage=stage, declared_parent_checkpoint=declared_parent_checkpoint, current_checkpoint=current_checkpoint,
        ),
    }


def with_evaluation_cache_identity(
    report: dict[str, Any],
    *,
    stage: str,
    declared_parent_checkpoint: Path | str,
    evaluated_checkpoint: Path | str,
    evaluation_role: str,
    manifests: dict[str, str],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Stamps a freshly-computed cached-evaluation dict with its full
    identity before it is written to disk. Returns a new dict; does not
    mutate the input in place."""
    return {
        **report,
        "identity": evaluation_cache_identity(
            stage=stage, declared_parent_checkpoint=declared_parent_checkpoint, evaluated_checkpoint=evaluated_checkpoint,
            evaluation_role=evaluation_role, manifests=manifests, config=config,
        ),
    }


def load_resumable_round_reports(
    summary_path: Path, *, log: Callable[[str], None], stage: str, declared_parent_checkpoint: Path | str,
) -> list[dict[str, Any]]:
    """Loads a stage's CURRENT-GENERATION `canonical_<stage>_run_summary.
    <namespace>.json` round list (pass `summary_path =
    current_generation_path(...)`  -- this function does not apply the
    namespace itself, so callers control exactly which path is read).
    Returns it unchanged if its LAST round validates
    (`round_record_validity_reason`); otherwise logs why and returns `[]`
    (a fresh start for this stage's `consecutive_passes`/
    `current_checkpoint` state) WITHOUT ever mutating the file."""
    if not summary_path.exists():
        return []
    try:
        round_reports = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not round_reports:
        return []
    reason = round_record_validity_reason(round_reports[-1], stage=stage, declared_parent_checkpoint=declared_parent_checkpoint)
    if reason is not None:
        log(f"Ignoring existing run summary at {summary_path} for resume: {reason}. "
            "Starting this stage's consecutive_passes/current_checkpoint fresh from the declared parent checkpoint.")
        return []
    return round_reports


def load_cached_evaluation_if_current(
    path: Path, *, log: Callable[[str], None], expected_identity: dict[str, Any],
) -> dict[str, Any] | None:
    """Loads a cached evaluation report (pass `path =
    current_generation_path(...)`) ONLY if its own identity matches
    `expected_identity` exactly (stage, generation, declared parent,
    evaluated-checkpoint content, evaluation role, manifest content,
    evaluation config). Returns `None` (never resume/reuse) on any
    mismatch, missing file, or unreadable content."""
    if not path.exists():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    reason = identity_mismatch_reason(report.get("identity") if isinstance(report, dict) else None, expected_identity)
    if reason is not None:
        log(f"Ignoring cached evaluation at {path}: {reason}. Recomputing.")
        return None
    return report
