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

# ---------------------------------------------------------------------------
# G10b-v2 -- coordinator-authorized correction.
#
# The v1 `declare()` above resolved categories 3-6 (canonical_advanced_ppo,
# quarantine, era_925, era_928) with an invented "lexicographically first
# unselected" rule after finding them genuinely ambiguous. That violated the
# plan's own instruction to STOP on unresolved ambiguity rather than choose.
# A read-only audit (PHASE2_G10B_SELECTION_AUDIT.tsv) then confirmed 0 of 4
# categories are uniquely determined by any pre-existing evidence.
#
# v1's SELECTION/BASELINE files and outcomes are PRESERVED, unmodified, as
# superseded/provisional diagnostic history -- never deleted, never silently
# treated as satisfying G10b.
#
# The coordinator authorized a deterministic, outcome-independent hash-sample
# in their place: for each ambiguous category, score every eligible candidate
# as sha256("G10b-v2|<category>|<checkpoint_sha256>") and take the lowest hex
# score. This is a preregistered compatibility-coverage sample, explicitly
# NOT a "best checkpoint" rule -- it is independent of policy performance,
# prior load outcomes, and filename ordering, and it is fixed here in source
# before any new PPO.load is executed.
SELECTION_V2 = "docs/migration/PHASE2_REPRESENTATIVE_SELECTION_V2.tsv"
BASELINE_V2 = "docs/migration/PHASE2_REPRESENTATIVE_LOAD_BASELINE_V2.tsv"

SELECTION_V2_FIELDS = (
    "category", "path", "checkpoint_sha256", "selection_input", "selection_score_sha256",
    "pool_size_before_exclusions", "excluded_count", "eligible_pool_size", "selection_rule_version",
)

SELECTION_RULE_VERSION = "G10b-v2-hash-stratified"
FIXED_RULE_LABEL = "accepted-plan-fixed-category"

# The three categories the accepted plan identifies exact files for directly
# (no ambiguity, unchanged from v1): take every matching checkpoint.
FIXED_CATEGORIES = (
    "generalized_waypoint_both_seed2_lineage",
    "split_branch_pilot",
    "foreground_bot_models",
)

PROVENANCE_V2 = (
    "first authorized G10b real-load baseline frozen in Phase 2 "
    "(V2 coordinator-authorized hash-stratified representative selection)"
)


def _v2_hash_score(category_name: str, checkpoint_sha256: str) -> str:
    payload = f"G10b-v2|{category_name}|{checkpoint_sha256}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def declare_v2(repo: Path) -> list[dict[str, str]]:
    with (repo / INVENTORY).open(encoding="utf-8", newline="") as handle:
        rows = sorted(csv.DictReader(handle, delimiter="\t"), key=lambda r: r["path"])
    specs = _categories(rows)
    chosen: list[dict[str, str]] = []

    for category in FIXED_CATEGORIES:
        spec = specs[category]
        candidates = [r for r in rows if spec["match"](r)]
        if not candidates:
            raise RuntimeError(f"G10b-v2 fixed category produced no candidate: {category}")
        for row in candidates:
            chosen.append({
                "category": category,
                "path": row["path"],
                "checkpoint_sha256": row["sha256"],
                "selection_input": "",
                "selection_score_sha256": "",
                "pool_size_before_exclusions": str(len(candidates)),
                "excluded_count": "0",
                "eligible_pool_size": str(len(candidates)),
                "selection_rule_version": FIXED_RULE_LABEL,
            })
    fixed_shas = {c["checkpoint_sha256"] for c in chosen}

    canon_pool = [r for r in rows if specs["canonical_advanced_ppo"]["match"](r)]
    quar_pool = [r for r in rows if specs["quarantine"]["match"](r)]
    canon_paths = {r["path"] for r in canon_pool}
    quar_paths = {r["path"] for r in quar_pool}

    def pick_stratum(category_name: str, pool_before: list[dict[str, str]], eligible: list[dict[str, str]]) -> None:
        if not eligible:
            raise RuntimeError(f"G10b-v2 stratum produced zero eligible candidates: {category_name}")
        scored = sorted(
            ((_v2_hash_score(category_name, r["sha256"]), r) for r in eligible),
            key=lambda item: item[0],
        )
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            raise RuntimeError(f"G10b-v2 hash tie in stratum {category_name}: {scored[0][0]!r}")
        score, winner = scored[0]
        chosen.append({
            "category": category_name,
            "path": winner["path"],
            "checkpoint_sha256": winner["sha256"],
            "selection_input": f"G10b-v2|{category_name}|{winner['sha256']}",
            "selection_score_sha256": score,
            "pool_size_before_exclusions": str(len(pool_before)),
            "excluded_count": str(len(pool_before) - len(eligible)),
            "eligible_pool_size": str(len(eligible)),
            "selection_rule_version": SELECTION_RULE_VERSION,
        })

    # 1. canonical_advanced_ppo: all 45 candidates, no exclusions.
    pick_stratum("canonical_advanced_ppo", canon_pool, canon_pool)

    # 2. quarantine: all 8 candidates, no exclusions.
    pick_stratum("quarantine", quar_pool, quar_pool)

    # 3. era_925: shape==[925], excluding canonical_advanced_ppo, quarantine,
    #    and anything already mandated elsewhere in the accepted G10b set --
    #    those categories already have their own dedicated strata, so letting
    #    them win the broad 925 stratum would reduce rather than increase
    #    compatibility coverage.
    e925_pool = [r for r in rows if r["obs_space_shape"] == "[925]"]
    e925_eligible = [
        r for r in e925_pool
        if r["path"] not in canon_paths and r["path"] not in quar_paths and r["sha256"] not in fixed_shas
    ]
    pick_stratum("era_925", e925_pool, e925_eligible)

    # 4. era_928: shape==[928], excluding the six already-mandatory
    #    generalized_waypoint_both_seed2_* checkpoints and anything else
    #    already mandated elsewhere in the accepted G10b set.
    e928_pool = [r for r in rows if r["obs_space_shape"] == "[928]"]
    e928_eligible = [r for r in e928_pool if r["sha256"] not in fixed_shas]
    pick_stratum("era_928", e928_pool, e928_eligible)

    out = repo / SELECTION_V2
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTION_V2_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(chosen)
    return chosen

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


def run_v2(repo: Path, corpus: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Load exactly the V2 selection (13 fixed + 4 hash-stratified). Same
    per-checkpoint isolated-subprocess protocol as ``run()``, against the
    coordinator-authorized selection file instead of the withdrawn v1 one."""
    import os

    selection_path = repo / SELECTION_V2
    if not selection_path.is_file():
        raise RuntimeError("G10b-v2 selection has not been declared; run `declare-v2` first")
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
            failures.append(f"G10b-v2 checkpoint absent from corpus: {rel}")
            continue
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != entry["checkpoint_sha256"]:
            failures.append(f"G10b-v2 checkpoint bytes changed: {rel} {digest} != {entry['checkpoint_sha256']}")
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
            "provenance": PROVENANCE_V2,
        })
        if payload["outcome"] == "loaded" and matches is False:
            failures.append(
                f"G10b-v2 {rel} loaded to {payload.get('policy_class_module')}."
                f"{payload.get('policy_class_qualname')} but inventory says "
                f"{inv.get('policy_class_module')}.{inv.get('policy_class_qualname')}"
            )

    results.sort(key=lambda r: (r["category"], r["path"]))
    with (repo / BASELINE_V2).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BASELINE_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
    return results, failures


def compare_v2(repo: Path, corpus: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Re-run the V2 declared set and require every outcome to match the frozen V2 baseline."""
    baseline_path = repo / BASELINE_V2
    if not baseline_path.is_file():
        raise RuntimeError("no frozen V2 load baseline to compare against")
    with baseline_path.open(encoding="utf-8", newline="") as handle:
        frozen = {r["path"]: r for r in csv.DictReader(handle, delimiter="\t")}
    results, failures = run_v2(repo, corpus)
    for row in results:
        old = frozen.get(row["path"])
        if old is None:
            failures.append(f"G10b-v2 new checkpoint not in frozen baseline: {row['path']}")
            continue
        for field in ("outcome", "exception_type", "policy_class_module", "policy_class_qualname",
                      "observation_space", "action_space", "sha256"):
            if row[field] != old[field]:
                failures.append(f"G10b-v2 {row['path']} {field}: now={row[field]!r} frozen={old[field]!r}")
    return results, failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "run", "compare", "declare-v2", "run-v2", "compare-v2"))
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args(argv)
    repo, corpus = args.repo.resolve(), args.corpus.resolve()

    if args.command == "declare":
        chosen = declare(repo)
        print(json.dumps({"declared": len(chosen), "selection": chosen}, indent=2))
        return 0

    if args.command == "declare-v2":
        chosen = declare_v2(repo)
        print(json.dumps({"declared": len(chosen), "selection": chosen}, indent=2))
        return 0

    if args.command in ("run-v2", "compare-v2"):
        results, failures = (run_v2 if args.command == "run-v2" else compare_v2)(repo, corpus)
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
