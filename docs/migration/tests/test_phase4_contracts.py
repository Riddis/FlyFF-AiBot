from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "docs/migration/tools/phase4_contracts.py"
SPEC = importlib.util.spec_from_file_location("phase4_contracts", TOOL)
assert SPEC is not None and SPEC.loader is not None
contracts = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = contracts
SPEC.loader.exec_module(contracts)


def test_revised_g3_replays_exact_frozen_live_contract() -> None:
    failures, evidence = contracts.check_observation(REPO)
    assert failures == [], json.dumps(failures, indent=2)
    assert evidence["observation_count"] == 10_016
    assert evidence["direct_case_count"] == 4_126
    assert evidence["direct_hypot_mismatch_count"] == 0
    assert evidence["signed_diagonal_nextabove_exact"] is True


def test_ggeo_preserves_independent_point_and_field_results() -> None:
    failures, evidence = contracts.check_geodesic(REPO)
    assert failures == [], json.dumps(failures, indent=2)
    assert evidence["comparison_count"] == 526
    assert evidence["mismatch_count"] == 108
    assert evidence["classification"] == {
        "one_ulp": 105,
        "two_ulp": 1,
        "field_absent_point_finite": 2,
    }


def test_observation_contract_import_is_stdlib_only() -> None:
    probe = """
import json, sys
sys.path.insert(0, sys.argv[1])
from farming.observation_contract import OBSERVATION_SCHEMA_HASH, OBSERVATION_SCHEMA_ID
blocked = sorted(name for name in sys.modules if name == 'numpy' or name.startswith(('numpy.', 'gymnasium', 'stable_baselines3', 'torch')))
print(json.dumps({'id': OBSERVATION_SCHEMA_ID, 'hash': OBSERVATION_SCHEMA_HASH, 'blocked': blocked}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(REPO)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload == {
        "id": "native-unified-923-v4",
        "hash": "F2D568C1C4A4B5F577C9C2E36A37B1C5533C2CE28D415846C3B68EC293C84609",
        "blocked": [],
    }


def test_b1_origins_bot_only_visibility_and_shim_api() -> None:
    failures, evidence = contracts.check_b1(REPO)
    assert failures == [], json.dumps(failures, indent=2)
    assert set(evidence["contexts"]) == {
        "bot-app",
        "bot-test",
        "recorder-app",
        "recorder-test",
        "simulator",
    }
    assert set(evidence["contexts"]["bot-app"]["bot_only"]) == set(contracts.BOT_ONLY_MODULES)
    assert all(not item["has_behavior_definitions"] for item in evidence["shims"].values())


def test_canonical_package_preserves_bot_public_api_lazily() -> None:
    """foreground_vision_bot/farming/__init__.py was removed in the
    2026-08-21 repository cleanup (ADR 0005's TEST_CONTRACT_RETIREMENT
    condition) -- its historical __all__ content is read via the frozen
    legacy-roots-pre-removal-20260821 tag instead of the live
    filesystem. The live-import assertions below (farming.__all__ ==
    that historical set, plus the two __module__ identity checks) are
    still a genuine current invariant, checked live against canonical
    farming/ exactly as before."""
    result = subprocess.run(
        ["git", "show", "legacy-roots-pre-removal-20260821:foreground_vision_bot/farming/__init__.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    tree = ast.parse(result.stdout)
    expected = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    )
    sys.path.insert(0, str(REPO))
    try:
        import farming

        assert farming.FarmingAction.__module__ == "farming.actions"
        assert farming.ObservationBuilder.__module__ == "farming.observation"
        assert farming.__all__ == expected
    finally:
        sys.path.pop(0)


def test_phase7_conftest_merge_preserves_the_two_required_behaviors() -> None:
    root_source = (REPO / "conftest.py").read_text(encoding="utf-8")
    suite_source = (REPO / "tests/conftest.py").read_text(encoding="utf-8")

    assert "def pytest_configure" in root_source
    assert "config.option.basetemp" in root_source
    assert 'ROOT = Path(__file__).resolve().parents[1]' in suite_source
    assert "sys.path.insert(0, str(ROOT))" in suite_source
    assert not (REPO / "flyff_farming_recorder/tests/conftest.py").exists()
    assert not (REPO / "foreground_vision_bot/tests/conftest.py").exists()


def test_phase7_canonical_origins_hold_in_all_historical_test_contexts() -> None:
    expected = {
        "farming": "farming/__init__.py",
        "farming.observation": "farming/observation.py",
        "farming.model_contract": "farming/model_contract.py",
        "farming.map_features": "farming/map_features.py",
        "farming.map_profile": "farming/map_profile.py",
        "position": "position/__init__.py",
        "position.IndependentNativeReader": "position/IndependentNativeReader.py",
        "position.NativeTraceTargets": "position/NativeTraceTargets.py",
        "position.native_process_service": "position/native_process_service.py",
        "simulator": "simulator/__init__.py",
        "simulator.split_branch_policy": "simulator/split_branch_policy.py",
        "simulator.navigation_history": "simulator/navigation_history.py",
        "navigation": "navigation/__init__.py",
        "navigation.kinodynamic_route_planner": "navigation/kinodynamic_route_planner.py",
        "navigation.movement_kernel": "navigation/movement_kernel.py",
        "navigation.navigation_evidence": "navigation/navigation_evidence.py",
        "devtools.recorder": "devtools/recorder/__init__.py",
        "runtime.runtime_bus": "runtime/runtime_bus.py",
        "mapper": "mapper/__init__.py",
        "mapper.Mapper": "mapper/Mapper.py",
    }
    probe = r"""
import importlib.util, json, pathlib, sys
repo = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo))
names = json.loads(sys.argv[2])
origins = {}
for name in names:
    spec = importlib.util.find_spec(name)
    origins[name] = None if spec is None or spec.origin is None else str(pathlib.Path(spec.origin).resolve())
print(json.dumps(origins, sort_keys=True))
"""
    # foreground_vision_bot/, flyff_farming_recorder/, and
    # flyff_farming_simulator/ were all removed entirely in the
    # 2026-08-21 repository cleanup (per ADR 0005's TEST_CONTRACT_
    # RETIREMENT condition -- their historical facade content is
    # proven separately via LEGACY_ROOTS_TAG in check_b1/check_b2, not
    # by requiring these directories to exist). No cwd context can be a
    # directory that no longer exists; REPO and two other real,
    # still-current subdirectories preserve the original intent of
    # proving import resolution is cwd-independent.
    for name in ("foreground_vision_bot", "flyff_farming_recorder", "flyff_farming_simulator"):
        assert not (REPO / name).exists()
    for context in (REPO, REPO / "tests", REPO / "simulator"):
        result = subprocess.run(
            [sys.executable, "-I", "-c", probe, str(REPO), json.dumps(list(expected))],
            cwd=context,
            capture_output=True,
            text=True,
            check=True,
        )
        origins = json.loads(result.stdout)
        assert origins == {
            name: str((REPO / relative).resolve()) for name, relative in expected.items()
        }
