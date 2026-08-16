"""G10b -- representative real ``PPO.load`` gate (EXPENSIVE, Torch-backed).

Deliberately separated from ``phase2_fingerprints.py`` so that ordinary
migration-integrity runs never import Torch or stable_baselines3.

Two-step protocol, in this order:

    declare   freeze the representative set (path, sha256, category, reason)
    run       load ONLY what was declared and record each outcome

``declare`` must be committed/inspected before ``run`` executes, so the file
list can never be chosen after seeing which checkpoints load.

Each checkpoint is loaded in its OWN subprocess with exactly one repository root
on ``PYTHONPATH``: ``farming.*`` exists in both the bot and simulator roots and
cannot be imported twice in one interpreter, and a hard crash in one load must
not destroy the rest of the run.

Nothing here trains, predicts, or writes to a model. A load FAILURE is valid
evidence and is recorded exactly, not repaired.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_DEFAULT = Path(__file__).resolve().parents[3]
INVENTORY = "docs/migration/CHECKPOINT_INVENTORY.tsv"
SELECTION = "docs/migration/PHASE2_REPRESENTATIVE_SELECTION.tsv"
BASELINE = "docs/migration/PHASE2_REPRESENTATIVE_LOAD_BASELINE.tsv"

SELECTION_FIELDS = ("category", "path", "sha256", "reason")
BASELINE_FIELDS = (
    "category", "path", "sha256", "outcome", "exception_type", "exception_message",
    "policy_class_module", "policy_class_qualname", "observation_space", "action_space",
    "contract_metadata_version", "inventory_policy_class_module", "inventory_policy_class_qualname",
    "matches_inventory", "provenance",
)

PROVENANCE = "first real-load outcome frozen in Phase 2; no earlier baseline existed"

# The accepted plan names seven categories but identifies exact files for only
# three of them. Categories 3-6 are resolved by an explicit deterministic rule
# rather than by judgement: within a category, take the lexicographically first
# repo-relative path in the frozen Phase-0 inventory that is not already
# selected. Narrow name-based categories are resolved BEFORE broad shape-based
# ones so that each category contributes a distinct checkpoint. This ordering is
# fixed here, in advance, and is independent of any load result.
RESOLUTION_ORDER = (
    "generalized_waypoint_both_seed2_lineage",
    "split_branch_pilot",
    "canonical_advanced_ppo",
    "quarantine",
    "era_925",
    "era_928",
    "foreground_bot_models",
)


def _categories(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    return {
        "generalized_waypoint_both_seed2_lineage": {
            "take": "all",
            "match": lambda r: "generalized_waypoint_both_seed2" in r["path"],
            "reason": "accepted plan: frozen generalized-waypoint seed2 lineage (all declared steps)",
        },
        "split_branch_pilot": {
            "take": "all",
            "match": lambda r: "split_branch_pilot" in Path(r["path"]).name,
            "reason": "accepted plan: all five split_branch_pilot* checkpoints",
        },
        "canonical_advanced_ppo": {
            "take": "first",
            "match": lambda r: Path(r["path"]).name.startswith("canonical_advanced_ppo_"),
            "reason": "accepted plan: one canonical_advanced_ppo_*; deterministic rule = lexicographically first unselected",
        },
        "quarantine": {
            "take": "first",
            "match": lambda r: "/_quarantine/" in r["path"],
            "reason": "accepted plan: one _quarantine checkpoint; deterministic rule = lexicographically first unselected",
        },
        "era_925": {
            "take": "first",
            "match": lambda r: r["obs_space_shape"] == "[925]",
            "reason": "accepted plan: one 925-era checkpoint; deterministic rule = lexicographically first unselected",
        },
        "era_928": {
            "take": "first",
            "match": lambda r: r["obs_space_shape"] == "[928]",
            "reason": "accepted plan: one 928-era checkpoint; deterministic rule = lexicographically first unselected",
        },
        "foreground_bot_models": {
            "take": "all",
            "match": lambda r: r["path"].startswith("foreground_vision_bot/"),
            "reason": "accepted plan: both foreground-bot models (contract-validation failure is an expected, valid outcome)",
        },
    }


def declare(repo: Path) -> list[dict[str, str]]:
    with (repo / INVENTORY).open(encoding="utf-8", newline="") as handle:
        rows = sorted(csv.DictReader(handle, delimiter="\t"), key=lambda r: r["path"])
    specs = _categories(rows)
    chosen: list[dict[str, str]] = []
    taken: set[str] = set()
    for category in RESOLUTION_ORDER:
        spec = specs[category]
        candidates = [r for r in rows if spec["match"](r) and r["path"] not in taken]
        picked = candidates if spec["take"] == "all" else candidates[:1]
        if not picked:
            raise RuntimeError(f"G10b category produced no candidate: {category}")
        for row in picked:
            taken.add(row["path"])
            chosen.append({
                "category": category,
                "path": row["path"],
                "sha256": row["sha256"],
                "reason": spec["reason"],
            })
    out = repo / SELECTION
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTION_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(chosen)
    return chosen


_LOAD_PROBE = r"""
import json, sys
result = {"outcome": "unknown"}
try:
    from stable_baselines3 import PPO
    model = PPO.load(sys.argv[1], device="cpu")
    policy = type(model.policy)
    result.update({
        "outcome": "loaded",
        "policy_class_module": policy.__module__,
        "policy_class_qualname": policy.__qualname__,
        "observation_space": str(model.observation_space),
        "action_space": str(model.action_space),
    })
except BaseException as error:
    result.update({
        "outcome": "failed",
        "exception_type": type(error).__name__,
        "exception_message": str(error),
    })
sys.stdout.write("@@RESULT@@" + json.dumps(result))
"""


def _root_for(path: str) -> str:
    return path.split("/", 1)[0]


def run(repo: Path, corpus: Path) -> tuple[list[dict[str, Any]], list[str]]:
    import os

    selection_path = repo / SELECTION
    if not selection_path.is_file():
        raise RuntimeError("G10b selection has not been declared; run `declare` first")
    with selection_path.open(encoding="utf-8", newline="") as handle:
        selection = list(csv.DictReader(handle, delimiter="\t"))
    with (repo / INVENTORY).open(encoding="utf-8", newline="") as handle:
        inventory = {r["path"]: r for r in csv.DictReader(handle, delimiter="\t")}

    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for entry in selection:
        rel = entry["path"]
        source = corpus / rel
        if not source.is_file():
            failures.append(f"G10b checkpoint absent from corpus: {rel}")
            continue
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            failures.append(f"G10b checkpoint bytes changed: {rel} {digest} != {entry['sha256']}")
            continue

        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo / _root_for(rel))
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.run(
            [sys.executable, "-s", "-c", _LOAD_PROBE, str(source)],
            capture_output=True, text=True, env=env, cwd=str(repo / _root_for(rel)),
        )
        marker = proc.stdout.rfind("@@RESULT@@")
        if marker == -1:
            payload = {
                "outcome": "failed",
                "exception_type": "ProbeCrashed",
                "exception_message": (proc.stderr or proc.stdout)[-500:].strip(),
            }
        else:
            payload = json.loads(proc.stdout[marker + len("@@RESULT@@"):])

        inv = inventory.get(rel, {})
        matches = (
            payload.get("policy_class_module") == inv.get("policy_class_module")
            and payload.get("policy_class_qualname") == inv.get("policy_class_qualname")
        ) if payload["outcome"] == "loaded" else None

        results.append({
            "category": entry["category"],
            "path": rel,
            "sha256": digest,
            "outcome": payload["outcome"],
            "exception_type": payload.get("exception_type", ""),
            "exception_message": " ".join(payload.get("exception_message", "").split())[:500],
            "policy_class_module": payload.get("policy_class_module", ""),
            "policy_class_qualname": payload.get("policy_class_qualname", ""),
            "observation_space": payload.get("observation_space", ""),
            "action_space": payload.get("action_space", ""),
            "contract_metadata_version": inv.get("farming_contract_metadata_version", ""),
            "inventory_policy_class_module": inv.get("policy_class_module", ""),
            "inventory_policy_class_qualname": inv.get("policy_class_qualname", ""),
            "matches_inventory": "" if matches is None else str(matches),
            "provenance": PROVENANCE,
        })
        # A successful load MUST resolve to the class the inventory recorded.
        if payload["outcome"] == "loaded" and matches is False:
            failures.append(
                f"G10b {rel} loaded to {payload.get('policy_class_module')}."
                f"{payload.get('policy_class_qualname')} but inventory says "
                f"{inv.get('policy_class_module')}.{inv.get('policy_class_qualname')}"
            )

    results.sort(key=lambda r: (r["category"], r["path"]))
    with (repo / BASELINE).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BASELINE_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
    return results, failures


def compare(repo: Path, corpus: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Re-run the declared set and require every outcome to match the frozen baseline."""
    baseline_path = repo / BASELINE
    if not baseline_path.is_file():
        raise RuntimeError("no frozen load baseline to compare against")
    with baseline_path.open(encoding="utf-8", newline="") as handle:
        frozen = {r["path"]: r for r in csv.DictReader(handle, delimiter="\t")}
    results, failures = run(repo, corpus)
    for row in results:
        old = frozen.get(row["path"])
        if old is None:
            failures.append(f"G10b new checkpoint not in frozen baseline: {row['path']}")
            continue
        for field in ("outcome", "exception_type", "policy_class_module", "policy_class_qualname",
                      "observation_space", "action_space", "sha256"):
            if row[field] != old[field]:
                failures.append(f"G10b {row['path']} {field}: now={row[field]!r} frozen={old[field]!r}")
    return results, failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "run", "compare"))
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args(argv)
    repo, corpus = args.repo.resolve(), args.corpus.resolve()

    if args.command == "declare":
        chosen = declare(repo)
        print(json.dumps({"declared": len(chosen), "selection": chosen}, indent=2))
        return 0

    results, failures = (run if args.command == "run" else compare)(repo, corpus)
    loaded = sum(r["outcome"] == "loaded" for r in results)
    print(json.dumps({
        "checkpoints": len(results),
        "loaded": loaded,
        "failed": len(results) - loaded,
        "gate_failures": failures,
        "ok": not failures,
        "results": results,
    }, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
