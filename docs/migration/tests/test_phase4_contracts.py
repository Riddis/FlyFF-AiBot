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
        [sys.executable, "-I", "-c", probe, str(REPO / "flyff_farming_simulator")],
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


def test_canonical_package_preserves_bot_public_api_lazily() -> None:
    tree = ast.parse(
        (REPO / "foreground_vision_bot/farming/__init__.py").read_text(encoding="utf-8")
    )
    expected = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    )
    sys.path.insert(0, str(REPO / "flyff_farming_simulator"))
    try:
        import farming

        assert farming.FarmingAction.__module__ == "farming.actions"
        assert farming.ObservationBuilder.__module__ == "farming.observation"
        assert farming.__all__ == expected
    finally:
        sys.path.pop(0)
