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
   model bytes across restarts/quarantines. A round record's own
   `carried_forward_checkpoint` must also refer to the SAME canonical path
   as that same record's `identity.current_checkpoint` -- matching bytes
   at a DIFFERENT path is never sufficient, since that would let a record
   vouch for one checkpoint while a completely different file is actually
   resumed from.

3. **Whole-chain validation** (`load_resumable_round_reports`): every
   round record in a persisted summary is validated in order, and the
   recorded `round` numbers must form the contiguous 1-based sequence
   `1, 2, ..., N` -- an invalid or non-contiguous prefix always discards
   the ENTIRE summary (never a partial resume of a validated suffix),
   and the next round to train is derived from the last validated round's
   own recorded number (`next_resumable_round`), not merely list length.

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
    `carried_forward_checkpoint` refers to the SAME checkpoint the record's
    own `identity.current_checkpoint` vouches for (canonical path equality,
    not raw-string equality -- a legitimately differently-spelled but
    physically identical path must still match), AND that checkpoint still
    exists on disk with UNCHANGED bytes (content-based, not path-alone -- a
    same-named file can hold different model bytes across
    restarts/quarantines). Covers every case section 8 of the task
    requires: wrong parent, wrong parent bytes, wrong stage, wrong
    architecture, a `carried_forward_checkpoint` that names a DIFFERENT
    path than the round's own vouched-for `current_checkpoint` (even with
    identical bytes elsewhere), stale checkpoint path, changed checkpoint
    bytes."""
    expected = _round_expected_identity(stage=stage, declared_parent_checkpoint=declared_parent_checkpoint)
    stored = record.get("identity")
    reason = identity_mismatch_reason(stored, expected)
    if reason is not None:
        return reason
    checkpoint_path = record.get("carried_forward_checkpoint")
    recorded_current_checkpoint = (stored or {}).get("current_checkpoint")
    recorded_sha = (stored or {}).get("current_checkpoint_sha256")
    if not checkpoint_path or not recorded_current_checkpoint or not recorded_sha:
        return (
            "round record is missing carried_forward_checkpoint or its recorded "
            "identity.current_checkpoint/current_checkpoint_sha256"
        )
    resolved = Path(checkpoint_path).resolve()
    recorded_resolved = Path(recorded_current_checkpoint).resolve()
    if resolved != recorded_resolved:
        return (
            f"carried_forward_checkpoint {checkpoint_path!r} (resolves to {resolved}) does not refer to the same "
            f"checkpoint as this round's own recorded identity.current_checkpoint {recorded_current_checkpoint!r} "
            f"(resolves to {recorded_resolved}) -- a round record must vouch for one exact checkpoint identity "
            "(path + content SHA), not merely equivalent bytes at a different path"
        )
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


def _round_schema_reason(record: Any) -> str | None:
    """Returns a reason string if `record` fails structural/type validation
    for a persisted round entry, independent of checkpoint/generation
    identity. Every check uses an EXACT type comparison (`type(x) is T`),
    never `isinstance`: `bool` is a subclass of `int` in Python (`True ==
    1`, `isinstance(True, int)` is `True`), so `isinstance`-based checks
    would silently accept `round: true` as round 1 or `consecutive_passes:
    True` as 1 -- persisted JSON is untrusted input and must never be
    allowed to exploit that looseness to manufacture a valid-looking round
    or graduation progress that was never actually earned."""
    if type(record) is not dict:
        return f"round record is not a JSON object (got {type(record).__name__})"
    if type(record.get("round")) is not int:
        return f"round field is not an exact JSON integer (got {record.get('round')!r})"
    if type(record.get("round_passed_absolute_bar")) is not bool:
        return f"round_passed_absolute_bar is not an exact JSON boolean (got {record.get('round_passed_absolute_bar')!r})"
    consecutive_passes = record.get("consecutive_passes")
    if type(consecutive_passes) is not int or consecutive_passes < 0:
        return f"consecutive_passes is not a non-negative exact JSON integer (got {consecutive_passes!r})"
    return None


def load_resumable_round_reports(
    summary_path: Path, *, log: Callable[[str], None], stage: str, declared_parent_checkpoint: Path | str,
) -> list[dict[str, Any]]:
    """Loads a stage's CURRENT-GENERATION `canonical_<stage>_run_summary.
    <namespace>.json` round list (pass `summary_path =
    current_generation_path(...)`  -- this function does not apply the
    namespace itself, so callers control exactly which path is read).
    Persisted state is treated as untrusted/corruptible input: malformed
    JSON, a non-list top-level payload, a non-dict round entry, a
    non-canonical field type (e.g. `round: 1.0` or `round: true`), and a
    `consecutive_passes` value inconsistent with the record's own
    `round_passed_absolute_bar` history (see `_round_schema_reason` and the
    per-round pass-sequence check below) are ALL rejected the same way as
    an identity mismatch -- never raised as an exception, never used to
    manufacture graduation progress, never partially resumed. Returns the
    round list unchanged only if EVERY round in it validates, in order:
    structurally (`_round_schema_reason`), the recorded `round` numbers
    form the contiguous 1-based sequence `1, 2, ..., len(round_reports)`,
    `consecutive_passes` is exactly `0` when `round_passed_absolute_bar` is
    `False` and exactly one more than the previous round's
    `consecutive_passes` when it is `True`, and finally checkpoint/
    generation identity (`round_record_validity_reason`) -- a validated
    suffix following an invalid or non-contiguous prefix is NEVER partially
    resumed. Otherwise logs why and returns `[]` (a fresh start for this
    stage's `consecutive_passes`/`current_checkpoint` state) WITHOUT ever
    mutating the file."""
    if not summary_path.exists():
        return []
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if type(payload) is not list:
        log(f"Ignoring existing run summary at {summary_path} for resume: top-level content is not a JSON list "
            f"(got {type(payload).__name__}). Starting this stage's consecutive_passes/current_checkpoint fresh "
            "from the declared parent checkpoint.")
        return []
    if not payload:
        return []
    previous_consecutive_passes = 0
    for expected_round_number, record in enumerate(payload, start=1):
        schema_reason = _round_schema_reason(record)
        if schema_reason is not None:
            log(f"Ignoring existing run summary at {summary_path} for resume: position {expected_round_number} in "
                f"the list is invalid: {schema_reason}. Starting this stage's consecutive_passes/current_checkpoint "
                "fresh from the declared parent checkpoint.")
            return []
        recorded_round_number = record["round"]
        if recorded_round_number != expected_round_number:
            log(f"Ignoring existing run summary at {summary_path} for resume: round sequence is not the contiguous "
                f"1-based sequence expected -- position {expected_round_number} in the list has round="
                f"{recorded_round_number!r}, expected {expected_round_number}. "
                "Starting this stage's consecutive_passes/current_checkpoint fresh from the declared parent checkpoint.")
            return []
        round_passed = record["round_passed_absolute_bar"]
        consecutive_passes = record["consecutive_passes"]
        expected_consecutive_passes = previous_consecutive_passes + 1 if round_passed else 0
        if consecutive_passes != expected_consecutive_passes:
            log(f"Ignoring existing run summary at {summary_path} for resume: round {expected_round_number} has "
                f"consecutive_passes={consecutive_passes} inconsistent with its round_passed_absolute_bar="
                f"{round_passed!r} and the preceding pass history (expected {expected_consecutive_passes}). "
                "Starting this stage's consecutive_passes/current_checkpoint fresh from the declared parent checkpoint.")
            return []
        previous_consecutive_passes = consecutive_passes
        reason = round_record_validity_reason(record, stage=stage, declared_parent_checkpoint=declared_parent_checkpoint)
        if reason is not None:
            log(f"Ignoring existing run summary at {summary_path} for resume: round {expected_round_number} invalid: "
                f"{reason}. Starting this stage's consecutive_passes/current_checkpoint fresh from the declared "
                "parent checkpoint.")
            return []
    return payload


def next_resumable_round(round_reports: list[dict[str, Any]]) -> int:
    """The next round to train, derived from the LAST VALIDATED round's own
    recorded `round` number (never merely `len(round_reports) + 1`, even
    though `load_resumable_round_reports`'s contiguity check makes those
    equivalent in practice -- this keeps the semantic dependency on the
    validated round number explicit). Canonical initial round is 1."""
    if not round_reports:
        return 1
    return round_reports[-1]["round"] + 1


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
