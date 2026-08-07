"""Run provenance: a small JSON manifest written next to every canonical
Basic/Beginner checkpoint, so six months from now nobody has to infer what
produced it from a filename.

No existing facility for this was found anywhere in the codebase (checked:
no git-commit capture, no run-manifest writer, anywhere under simulator/) --
this is new, not a duplicate of something already there.

Deliberately NOT a general experiment-tracking framework: one function,
one JSON file per checkpoint, plain dict fields.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def capture_git_state() -> dict[str, Any]:
    """Best-effort git commit + dirty-tree state. Never raises: a run
    should still complete (and still write a manifest saying git info was
    unavailable) even outside a git checkout."""

    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=10, check=True,
            )
            return result.stdout.strip()
        except Exception:
            return None

    commit = run("rev-parse", "HEAD")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    dirty_output = run("status", "--porcelain")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": None if dirty_output is None else bool(dirty_output),
        "available": commit is not None,
    }


def build_run_manifest(
    *,
    stage: str,
    milestone: str,
    seeds: list[int] | int,
    config: dict[str, Any],
    curriculum_path: str | None = None,
    heldout_manifest_path: str | None = None,
    recording_paths: list[str] | None = None,
    recovery_config: dict[str, Any] | None = None,
    dagger_config: dict[str, Any] | None = None,
    architecture_contract: dict[str, Any] | None = None,
    starting_checkpoint: str | None = None,
    output_checkpoint: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one run's provenance record.

    `starting_checkpoint=None` is the explicit, unambiguous way to record
    "this run started from a fresh current-architecture initialization, not
    any historical checkpoint" -- the specific fact the fresh-lineage
    decision depends on being verifiable later, not inferred from a
    filename. Pass the historical/Basic/Beginner checkpoint path here for
    every later stage that legitimately continues from one.
    """

    from .navigation_history import POLICY_INPUT_SIZE, RAW_OBSERVATION_SIZE, STEERING_POLICY_INPUT_SCHEMA_ID

    default_contract = {
        "raw_observation_schema_id": "923-value production observation contract",
        "raw_observation_size": RAW_OBSERVATION_SIZE,
        "policy_input_schema_id": STEERING_POLICY_INPUT_SCHEMA_ID,
        "policy_input_size": POLICY_INPUT_SIZE,
        "policy_class": "SplitSteeringNavigationPolicy",
    }
    if architecture_contract:
        default_contract.update(architecture_contract)

    recording_hashes: list[dict[str, str]] | None = None
    if recording_paths:
        from .schema import recording_sha256

        recording_hashes = [
            {"path": str(Path(p)), "sha256": recording_sha256(Path(p))} for p in recording_paths
        ]

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_stage": stage,
        "milestone": milestone,
        "git": capture_git_state(),
        "seeds": seeds if isinstance(seeds, list) else [seeds],
        "config": config,
        "curriculum_path": curriculum_path,
        "heldout_manifest_path": heldout_manifest_path,
        "recordings": recording_hashes,
        "recovery_config": recovery_config,
        "dagger_config": dagger_config,
        "architecture_contract": default_contract,
        "starting_checkpoint": starting_checkpoint,
        "fresh_initialization": starting_checkpoint is None,
        "output_checkpoint": output_checkpoint,
        "extra": extra or {},
    }


def write_run_manifest(checkpoint_path: str | Path, manifest: dict[str, Any]) -> Path:
    """Write the manifest to `<checkpoint stem>.provenance.json`, next to
    the checkpoint it describes."""

    checkpoint = Path(checkpoint_path)
    manifest_path = checkpoint.with_suffix("").with_suffix(".provenance.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return manifest_path
