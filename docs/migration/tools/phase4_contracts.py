"""Read-only Phase-4 gates for canonical observation and geodesic contracts."""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import msgpack


REPO_DEFAULT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = Path("tests/fixtures/migration")
PHASE3_TOOL = Path("docs/migration/tools/phase3_capture.py")
EXPECTED_OBSERVATION_COUNT = 10_016
EXPECTED_OBSERVATION_SIZE = 923
EXPECTED_DIRECT_CASES = 4_126
SHARED_MODULES = (
    "actions",
    "model_contract",
    "map_masks",
    "reward",
    "session",
    "observation",
    "map_features",
    "observation_contract",
)
BOT_ONLY_MODULES = (
    "config",
    "control",
    "debug_validation",
    "environment",
    "kills",
    "map_context",
    "native_world",
    "reporting",
    "sb3_adapter",
    "sb3_training",
    "startup",
    "telemetry",
    "trainer",
)
B1_PREIMPORT_ORDER = {
    "foreground_vision_bot/foreground_vision_farm.py": "from Bot import Bot",
    "foreground_vision_bot/conftest.py": "import pytest",
    "foreground_vision_bot/tests/conftest.py": "sys.path[:] = [str(CANONICAL_FARMING_PARENT)",
    "foreground_vision_bot/tools/run_observation_telemetry.py": "from assets.Assets import MobInfo",
    "flyff_farming_recorder/app.py": "from recorder.gui import run_gui",
    "flyff_farming_recorder/recorder/session.py": "from position.IndependentMonsterRediscovery import",
    "flyff_farming_recorder/tests/conftest.py": "sys.path[:] = [str(CANONICAL_FARMING_PARENT)",
    "flyff_farming_recorder/FlyffFarmingRecorder.spec": "a = Analysis(",
}

EXPECTED_CALL_SITES = {
    "bounded_geodesic_field": {
        "docs/migration/tools/phase3_capture.py": 2,
        "flyff_farming_simulator/simulator/demonstrations.py": 1,
        "flyff_farming_simulator/simulator/environment.py": 1,
        "flyff_farming_simulator/simulator/route_waypoint_generator.py": 1,
        "flyff_farming_simulator/tests/test_deep_review.py": 1,
        "flyff_farming_simulator/tests/test_reward_audit_v17.py": 1,
    },
    "geodesic_distance": {
        "docs/migration/tools/phase3_capture.py": 1,
        "flyff_farming_simulator/tests/test_deep_review.py": 1,
        "foreground_vision_bot/tests/test_farming_map_features.py": 4,
    },
    "geodesic_distances": {
        "foreground_vision_bot/farming/native_world.py": 1,
    },
}


def _load_phase3(repo: Path) -> Any:
    path = repo / PHASE3_TOOL
    spec = importlib.util.spec_from_file_location("phase3_capture_for_phase4", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Phase-3 capture tool at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_observation(repo: Path) -> tuple[list[str], dict[str, Any]]:
    """Reproduce the frozen live vectors and direct hypot reference exactly."""
    failures: list[str] = []
    fixture = repo / FIXTURE_ROOT
    input_path = fixture / "observation_inputs.msgpack.gz"
    expected = _json(fixture / "observation_expected.json")
    boundary = _json(fixture / "neighbour_boundary.json")
    phase3 = _load_phase3(repo)

    if _sha256(input_path) != expected["input_corpus_sha256"]:
        failures.append("G3 frozen input corpus SHA differs from observation_expected.json")
    rows = msgpack.unpackb(gzip.decompress(input_path.read_bytes()), raw=False)
    if len(rows) != EXPECTED_OBSERVATION_COUNT:
        failures.append(f"G3 input count={len(rows)}, expected {EXPECTED_OBSERVATION_COUNT}")

    with tempfile.TemporaryDirectory(prefix="flyffrl-phase4-g3-") as raw_temp:
        temp = Path(raw_temp)
        output = temp / "canonical.f32"
        meta_path = temp / "canonical.json"
        phase3._run(
            repo,
            "_obs_worker",
            str(repo / "flyff_farming_simulator"),
            str(input_path),
            str(output),
            str(meta_path),
        )
        raw = output.read_bytes()
        row_size = EXPECTED_OBSERVATION_SIZE * 4
        actual_hashes = [
            hashlib.sha256(raw[offset : offset + row_size]).hexdigest()
            for offset in range(0, len(raw), row_size)
        ]
        if len(raw) != EXPECTED_OBSERVATION_COUNT * row_size:
            failures.append(f"G3 canonical byte length={len(raw)}")
        vector_differences = []
        if actual_hashes != expected["expected_row_sha256"]:
            vector_differences = [
                index
                for index, (actual, wanted) in enumerate(
                    zip(actual_hashes, expected["expected_row_sha256"], strict=False)
                )
                if actual != wanted
            ]
            failures.append(f"G3 canonical live-vector mismatches={vector_differences[:20]}")
        aggregate = hashlib.sha256(raw).hexdigest()
        if aggregate != expected["aggregate_output_sha256"]:
            failures.append(
                f"G3 aggregate={aggregate}, expected={expected['aggregate_output_sha256']}"
            )
        meta = _json(meta_path)
        if meta != {
            "count": EXPECTED_OBSERVATION_COUNT,
            "dtype": "float32",
            "finite": True,
            "shape": [EXPECTED_OBSERVATION_SIZE],
        }:
            failures.append(f"G3 canonical shape/dtype/finite metadata differs: {meta!r}")

        cases = phase3.build_boundary_cases()
        cases_path = temp / "boundary_cases.json"
        cases_path.write_bytes(phase3.json_bytes(cases))
        direct_path = temp / "canonical_boundary.json"
        phase3._run(
            repo,
            "_nearby_worker",
            str(repo / "flyff_farming_simulator"),
            str(cases_path),
            str(direct_path),
        )
        direct = _json(direct_path)

    direct_failures: list[str] = []
    for case, actual in zip(cases, direct, strict=True):
        positions = {str(item[0]): (float(item[1]), float(item[2])) for item in case["positions"]}
        wanted = {
            actor_id: sum(
                math.hypot(other_x - actor_x, other_y - actor_y) <= 8.0
                for other_x, other_y in positions.values()
            )
            for actor_id, (actor_x, actor_y) in positions.items()
        }
        if actual["counts"] != wanted:
            direct_failures.append(case["name"])
    if direct_failures:
        failures.append(f"G3 direct live-hypot mismatches={direct_failures[:20]}")

    frozen_diagonals = {row["name"]: row["bot"] for row in boundary["direct_mismatches"]}
    current_diagonals = {
        row["name"]: row["counts"] for row in direct if row["name"] in frozen_diagonals
    }
    if current_diagonals != frozen_diagonals:
        failures.append("G3 four signed diagonal-nextabove cases differ from frozen live results")
    if len(direct) != EXPECTED_DIRECT_CASES:
        failures.append(f"G3 direct case count={len(direct)}, expected {EXPECTED_DIRECT_CASES}")

    evidence = {
        "observation_count": len(actual_hashes),
        "exact_live_vectors": len(actual_hashes) - len(vector_differences),
        "aggregate_output_sha256": hashlib.sha256(raw).hexdigest(),
        "direct_case_count": len(direct),
        "direct_hypot_mismatch_count": len(direct_failures),
        "signed_diagonal_nextabove_exact": current_diagonals == frozen_diagonals,
    }
    return failures, evidence


def _tracked_python(repo: Path) -> list[str]:
    proc = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo}",
            "ls-files",
            "*.py",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.splitlines()


def geodesic_call_sites(repo: Path) -> dict[str, dict[str, int]]:
    methods = tuple(EXPECTED_CALL_SITES)
    found = {method: Counter() for method in methods}
    for relative in _tracked_python(repo):
        if relative.endswith("/farming/map_features.py"):
            continue
        path = repo / relative
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr in found:
                found[node.func.attr][relative] += 1
    return {method: dict(counter) for method, counter in found.items()}


def _geodesic_classification(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"one_ulp": 0, "two_ulp": 0, "field_absent_point_finite": 0}
    infinity = "7ff0000000000000"
    for row in rows:
        field_bits = row["field_value_bits"]
        point_bits = row["point_value_bits"]
        if field_bits == infinity and point_bits != infinity:
            counts["field_absent_point_finite"] += 1
            continue
        difference = abs(int(field_bits, 16) - int(point_bits, 16))
        if difference == 1:
            counts["one_ulp"] += 1
        elif difference == 2:
            counts["two_ulp"] += 1
    return counts


def check_geodesic(repo: Path) -> tuple[list[str], dict[str, Any]]:
    """Reproduce point and field outputs independently, including cache identity."""
    failures: list[str] = []
    expected = _json(repo / FIXTURE_ROOT / "bounded_geodesic.json")
    phase3 = _load_phase3(repo)
    with tempfile.TemporaryDirectory(prefix="flyffrl-phase4-ggeo-") as raw_temp:
        output = Path(raw_temp) / "geodesic.json"
        phase3._run(
            repo,
            "_geodesic_worker",
            str(repo / "flyff_farming_simulator"),
            str(output),
        )
        actual = _json(output)
    if actual["cases"] != expected["cases"]:
        differing = [
            wanted["name"]
            for wanted, current in zip(expected["cases"], actual["cases"], strict=False)
            if wanted != current
        ]
        failures.append(f"G-GEO independent point/field case differences={differing[:20]}")
    for key in ("comparison_count", "exact_match_count", "mismatch_count"):
        if actual[key] != expected[key]:
            failures.append(f"G-GEO {key}={actual[key]}, expected={expected[key]}")
    classification = _geodesic_classification(actual["mismatches"])
    expected_classification = {
        "one_ulp": 105,
        "two_ulp": 1,
        "field_absent_point_finite": 2,
    }
    if classification != expected_classification:
        failures.append(
            f"G-GEO mismatch classification={classification}, expected={expected_classification}"
        )
    call_sites = geodesic_call_sites(repo)
    if call_sites != EXPECTED_CALL_SITES:
        failures.append(f"G-GEO call-site inventory changed: {call_sites!r}")
    return failures, {
        "comparison_count": actual["comparison_count"],
        "exact_match_count": actual["exact_match_count"],
        "mismatch_count": actual["mismatch_count"],
        "classification": classification,
        "call_sites": call_sites,
    }


_B1_PROBE = r"""
import importlib, json, pathlib, sys
mode, repo_raw = sys.argv[1:3]
repo = pathlib.Path(repo_raw)
sim = repo / "flyff_farming_simulator"
bot = repo / "foreground_vision_bot"
recorder = repo / "flyff_farming_recorder"
if mode.startswith("bot-"):
    sys.path[:0] = [str(sim), str(bot)]
elif mode.startswith("recorder-"):
    sys.path[:0] = [str(sim), str(recorder)]
elif mode == "simulator":
    sys.path.insert(0, str(sim))
else:
    raise RuntimeError(mode)
shared = {}
bot_only = {}
if mode.startswith("bot-") or mode == "simulator":
    package = importlib.import_module("farming")
    for name in json.loads(sys.argv[3]):
        shared[name] = str(pathlib.Path(importlib.import_module(f"farming.{name}").__file__).resolve())
    if mode.startswith("bot-"):
        for name in json.loads(sys.argv[4]):
            bot_only[name] = str(pathlib.Path(importlib.import_module(f"farming.{name}").__file__).resolve())
    package_origin = str(pathlib.Path(package.__file__).resolve())
else:
    contract = importlib.import_module("farming.observation_contract")
    importlib.import_module("recorder.session")
    shared["observation_contract"] = str(pathlib.Path(contract.__file__).resolve())
    package_origin = str(pathlib.Path(importlib.import_module("farming").__file__).resolve())
blocked = sorted(name for name in sys.modules if mode.startswith("recorder-") and (name == "numpy" or name.startswith(("numpy.", "gymnasium", "stable_baselines3", "torch"))))
print(json.dumps({"package": package_origin, "shared": shared, "bot_only": bot_only, "blocked": blocked}, sort_keys=True))
"""


def _b1_probe(repo: Path, mode: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            _B1_PROBE,
            mode,
            str(repo),
            json.dumps(SHARED_MODULES),
            json.dumps(BOT_ONLY_MODULES),
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if result.returncode:
        raise RuntimeError(f"B1 {mode} probe failed:\n{result.stdout}\n{result.stderr}")
    return json.loads(result.stdout)


def _shim_api(repo: Path, relative: str) -> tuple[set[str], set[str], bool]:
    tree = ast.parse((repo / relative).read_text(encoding="utf-8"), filename=relative)
    imported: set[str] = set()
    exported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (
            (node.module or "").startswith("flyff_farming_simulator.farming")
        ):
            imported.update(alias.asname or alias.name for alias in node.names)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            exported.update(ast.literal_eval(node.value))
    has_behavior = any(isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree))
    return imported, exported, has_behavior


def check_b1(repo: Path) -> tuple[list[str], dict[str, Any]]:
    """Prove canonical origins, bot-only package visibility, shim API, and purity."""
    failures: list[str] = []
    sim_root = (repo / "flyff_farming_simulator").resolve()
    bot_root = (repo / "foreground_vision_bot").resolve()
    evidence: dict[str, Any] = {"contexts": {}, "shims": {}}
    marker = "# BRIDGE B1 — removed in Phase 7"
    evidence["preimport_order"] = {}
    for relative, first_consumer in B1_PREIMPORT_ORDER.items():
        source = (repo / relative).read_text(encoding="utf-8")
        marker_index = source.find(marker)
        consumer_index = source.find(first_consumer)
        ordered = marker_index >= 0 and consumer_index >= 0 and marker_index < consumer_index
        evidence["preimport_order"][relative] = ordered
        if not ordered:
            failures.append(f"B1 marker/bootstrap is not before the first consumer in {relative}")
    spec_source = (repo / "flyff_farming_recorder/FlyffFarmingRecorder.spec").read_text(encoding="utf-8")
    if "pathex=[str(canonical_farming_parent), str(app_root)]" not in spec_source:
        failures.append("B1 recorder PyInstaller pathex does not put canonical farming first")
    for mode in ("bot-app", "bot-test", "recorder-app", "recorder-test", "simulator"):
        try:
            payload = _b1_probe(repo, mode)
        except RuntimeError as error:
            failures.append(str(error))
            continue
        evidence["contexts"][mode] = payload
        if not Path(payload["package"]).is_relative_to(sim_root):
            failures.append(f"B1 {mode} farming package origin is not canonical: {payload['package']}")
        for name, origin in payload["shared"].items():
            if not Path(origin).is_relative_to(sim_root):
                failures.append(f"B1 {mode} farming.{name} origin is not canonical: {origin}")
        for name, origin in payload["bot_only"].items():
            if not Path(origin).is_relative_to(bot_root):
                failures.append(f"B1 {mode} bot-only farming.{name} is hidden: {origin}")
        if payload["blocked"]:
            failures.append(f"B1 {mode} recorder metadata import loaded heavy modules: {payload['blocked']}")

    shim_names = ("actions", "model_contract", "map_masks", "reward", "session", "observation", "map_features")
    for name in shim_names:
        relative = f"foreground_vision_bot/farming/{name}.py"
        imported, exported, has_behavior = _shim_api(repo, relative)
        evidence["shims"][name] = {
            "imported": sorted(imported),
            "exported": sorted(exported),
            "has_behavior_definitions": has_behavior,
        }
        expected_imported = set(exported)
        if name == "observation":
            expected_imported.add("_DIRECT_ACTOR_FIELD_NAMES")
        if imported != expected_imported:
            failures.append(f"B1 {relative} import/export parity differs")
        if has_behavior:
            failures.append(f"B1 {relative} still defines behavioral classes/functions")
    if "_DIRECT_ACTOR_FIELD_NAMES" not in evidence["shims"]["observation"]["imported"]:
        failures.append("B1 observation shim lost depended-on private _DIRECT_ACTOR_FIELD_NAMES")
    return failures, evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("g3", "ggeo", "b1", "all"))
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    args = parser.parse_args()
    repo = args.repo.resolve()
    failures: list[str] = []
    evidence: dict[str, Any] = {}
    if args.command in ("g3", "all"):
        current, evidence["g3"] = check_observation(repo)
        failures.extend(current)
    if args.command in ("ggeo", "all"):
        current, evidence["ggeo"] = check_geodesic(repo)
        failures.extend(current)
    if args.command in ("b1", "all"):
        current, evidence["b1"] = check_b1(repo)
        failures.extend(current)
    print(json.dumps({"ok": not failures, "failures": failures, "evidence": evidence}, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
