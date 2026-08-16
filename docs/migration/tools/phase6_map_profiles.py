"""Read-only Phase-6 gates for the two preserved Tower map profiles."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_DEFAULT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = Path("tests/fixtures/migration")
G11_HASHES = {
    "coordinate_frame.json": "40339f6c397d38fe01d5b3a5300e5b9b6d499f06292f436b1f91ea34523a0414",
    "map.json": "faaf8633457bc1bcdb61c781c8ca62c6f2e008174ed5b284c3d6c08df92fe815",
    "occupancy.npy": "62fa3c9ec3aed0b3b134b82577292c0a8a67b0acc4111fde3a36e3d2684d789b",
}


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _probe(repo: Path, source: str, *arguments: str) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-I", "-c", source, str(repo), *arguments],
        cwd=repo,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"isolated probe failed ({result.returncode})\n{result.stdout}\n{result.stderr}"
        )
    return json.loads(result.stdout)


def _load_phase3(repo: Path) -> Any:
    path = repo / "docs/migration/tools/phase3_capture.py"
    spec = importlib.util.spec_from_file_location("phase3_capture_for_phase6", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Phase-3 capture tool at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check_profiles(repo: Path) -> tuple[list[str], dict[str, Any]]:
    probe = r'''
import dataclasses, json, pathlib, sys
repo = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(repo))
from flyff_farming_simulator.farming.map_profile import LIVE_TOWER_PROFILE, SIM_TOWER_PROFILE, TowerMapProfile
immutable = True
try:
    LIVE_TOWER_PROFILE.obstacle_radius_cells = 99
except (dataclasses.FrozenInstanceError, AttributeError):
    pass
else:
    immutable = False
print(json.dumps({
    "origin": str(pathlib.Path(sys.modules[TowerMapProfile.__module__].__file__).resolve()),
    "class": TowerMapProfile.__name__,
    "immutable": immutable,
    "live": [LIVE_TOWER_PROFILE.obstacle_radius_cells, LIVE_TOWER_PROFILE.teleport_radius_cells],
    "live_types": [type(LIVE_TOWER_PROFILE.obstacle_radius_cells).__name__, type(LIVE_TOWER_PROFILE.teleport_radius_cells).__name__],
    "simulator": [SIM_TOWER_PROFILE.obstacle_radius_cells, SIM_TOWER_PROFILE.teleport_radius_cells],
    "simulator_types": [type(SIM_TOWER_PROFILE.obstacle_radius_cells).__name__, type(SIM_TOWER_PROFILE.teleport_radius_cells).__name__],
}, sort_keys=True))
'''
    evidence = _probe(repo, probe)
    expected = {
        "class": "TowerMapProfile",
        "immutable": True,
        "live": [2, 2.0],
        "live_types": ["int", "float"],
        "simulator": [0, 2],
        "simulator_types": ["int", "int"],
    }
    failures = [
        f"profile {key} differs: {evidence.get(key)!r} != {value!r}"
        for key, value in expected.items()
        if evidence.get(key) != value
    ]
    expected_origin = (repo / "flyff_farming_simulator/farming/map_profile.py").resolve()
    if Path(evidence["origin"]).resolve() != expected_origin:
        failures.append(f"profile origin is not canonical: {evidence['origin']}")
    return failures, evidence


def _import_aliases(path: Path) -> dict[tuple[str, str], str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        (node.module or "", alias.name): alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }


def check_wiring(repo: Path) -> tuple[list[str], dict[str, Any]]:
    live_path = repo / "foreground_vision_bot/farming/map_context.py"
    simulator_path = repo / "flyff_farming_simulator/simulator/map_model.py"
    live_source = live_path.read_text(encoding="utf-8")
    simulator_source = simulator_path.read_text(encoding="utf-8")
    live_aliases = _import_aliases(live_path)
    simulator_aliases = _import_aliases(simulator_path)
    failures: list[str] = []
    if live_aliases.get(("map_profile", "LIVE_TOWER_PROFILE")) != "_LIVE_TOWER_PROFILE":
        failures.append("live loader does not privately import LIVE_TOWER_PROFILE")
    if simulator_aliases.get(
        ("farming.map_profile", "SIM_TOWER_PROFILE")
    ) != "_SIM_TOWER_PROFILE":
        failures.append("simulator loader does not privately import SIM_TOWER_PROFILE")
    if live_source.count("_LIVE_TOWER_PROFILE.") != 2:
        failures.append("live profile is not the source of exactly both loader defaults")
    if simulator_source.count("_SIM_TOWER_PROFILE.") != 3:
        failures.append("simulator profile is not the source of exactly three packaged-load radii")

    live_override_probe = r'''
import json, pathlib, sys
import numpy as np
repo = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(repo / "foreground_vision_bot"))
import farming.map_context as module
loaded = module.FarmingMapContext.load("Tower AoE", obstacle_buffer_radius_cells=0, teleport_buffer_radius_cells=1.0)
expected = module.inflate_map_masks(loaded.features.traversable, loaded.features.forbidden, obstacle_radius_cells=0, teleport_radius_cells=1)
print(json.dumps({"safe_exact": bool(np.array_equal(loaded.features.safe_traversable, expected.safe_traversable)), "teleport": loaded.features.teleport_buffer_radius_cells}))
'''
    simulator_override_probe = r'''
import json, pathlib, sys
import numpy as np
repo = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(repo / "flyff_farming_simulator"))
from farming.map_masks import inflate_map_masks
from simulator.map_model import MapModel
walkable = np.ones((11, 11), dtype=bool)
blocked = np.zeros_like(walkable); blocked[5, 5] = True
loaded = MapModel.from_arrays(walkable, forbidden=blocked, obstacle_radius_cells=1, teleport_radius_cells=3)
expected = inflate_map_masks(walkable, blocked, obstacle_radius_cells=1, teleport_radius_cells=3)
print(json.dumps({"safe_exact": bool(np.array_equal(loaded.features.safe_traversable, expected.safe_traversable)), "teleport": loaded.features.teleport_buffer_radius_cells}))
'''
    live_override = _probe(repo, live_override_probe)
    simulator_override = _probe(repo, simulator_override_probe)
    if live_override != {"safe_exact": True, "teleport": 1.0}:
        failures.append(f"live explicit override lost precedence: {live_override}")
    if simulator_override != {"safe_exact": True, "teleport": 3.0}:
        failures.append(f"simulator explicit override lost precedence: {simulator_override}")
    return failures, {
        "live_profile_reference_count": live_source.count("_LIVE_TOWER_PROFILE."),
        "simulator_profile_reference_count": simulator_source.count("_SIM_TOWER_PROFILE."),
        "live_override": live_override,
        "simulator_override": simulator_override,
    }


def check_g11(repo: Path) -> tuple[list[str], dict[str, Any]]:
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "docs/migration/tools/phase2_fingerprints.py"),
            "g11",
            "--repo",
            str(repo),
        ],
        cwd=repo,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        return [f"G11 command failed ({result.returncode}): {result.stdout}{result.stderr}"], {}
    evidence = json.loads(result.stdout)["g11"]["evidence"]
    failures: list[str] = []
    for path, actual in evidence["hashes"].items():
        expected = G11_HASHES[Path(path).name]
        if actual != expected:
            failures.append(f"G11 hash differs for {path}: {actual} != {expected}")
    if not all(evidence["pairs"].values()):
        failures.append(f"G11 raw Tower pairs differ: {evidence['pairs']}")
    if not evidence["marker_present"]:
        failures.append("G11 .skip_legacy_import marker is absent")
    return failures, evidence


def check_g12(repo: Path) -> tuple[list[str], dict[str, Any]]:
    phase3 = _load_phase3(repo)
    with tempfile.TemporaryDirectory(prefix="flyffrl-phase6-g12-") as raw:
        actual = phase3.capture_maps(repo, Path(raw))
    matches = {
        name: payload == (repo / FIXTURE_ROOT / name).read_bytes()
        for name, payload in actual.items()
    }
    failures = [
        f"G12 {name} does not reproduce its committed golden"
        for name in ("map_live.json", "map_simulator.json")
        if not matches[name]
    ]
    live = json.loads(actual["map_live.json"])
    simulator = json.loads(actual["map_simulator.json"])
    diagnostic = json.loads(actual["map6_diagnostic.json"])
    expected_counts = {
        "live": {"traversable": 59_818, "forbidden": 49, "safe_traversable": 52_071},
        "simulator": {"traversable": 59_818, "forbidden": 49, "safe_traversable": 59_726},
    }
    if live["obstacle_radius_cells"] != 2 or live["counts"] != expected_counts["live"]:
        failures.append(f"G12 live profile changed: radius={live['obstacle_radius_cells']}, counts={live['counts']}")
    if simulator["obstacle_radius_cells"] != 0 or simulator["counts"] != expected_counts["simulator"]:
        failures.append(f"G12 simulator profile changed: radius={simulator['obstacle_radius_cells']}, counts={simulator['counts']}")
    return failures, {
        "fixture_matches": matches,
        "live": {"obstacle_radius_cells": live["obstacle_radius_cells"], "teleport_radius_cells": live["teleport_radius_cells"], "counts": live["counts"]},
        "simulator": {"obstacle_radius_cells": simulator["obstacle_radius_cells"], "teleport_radius_cells": simulator["teleport_radius_cells"], "counts": simulator["counts"]},
        "map6": {"diagnostic_only": "DIAGNOSTIC ONLY" in diagnostic["label"], "safe_mask_xor_count": diagnostic["safe_mask_xor_count"], "fixture_match": matches["map6_diagnostic.json"]},
    }


def check_all(repo: Path) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    evidence: dict[str, Any] = {}
    for name, gate in (
        ("profiles", check_profiles),
        ("wiring", check_wiring),
        ("G11", check_g11),
        ("G12", check_g12),
    ):
        gate_failures, gate_evidence = gate(repo)
        failures.extend(gate_failures)
        evidence[name] = gate_evidence
    return failures, evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    arguments = parser.parse_args()
    problems, details = check_all(arguments.repo.resolve())
    print(json.dumps({"ok": not problems, "failures": problems, "evidence": details}, indent=2, sort_keys=True))
    raise SystemExit(bool(problems))
