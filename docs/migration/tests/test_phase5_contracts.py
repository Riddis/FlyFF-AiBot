from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "docs/migration/tools/phase5_contracts.py"
SPEC = importlib.util.spec_from_file_location("phase5_contracts", TOOL)
assert SPEC is not None and SPEC.loader is not None
contracts = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = contracts
SPEC.loader.exec_module(contracts)


def test_g1_canonical_api_is_the_structural_union() -> None:
    failures, evidence = contracts.check_g1(REPO)
    assert failures == [], json.dumps(failures, indent=2)
    assert evidence["historical_module_count"] == 23
    assert evidence["missing"] == {}


def test_np_layering_and_narrow_promotion_contract() -> None:
    failures, evidence = contracts.check_np(REPO)
    assert failures == [], json.dumps(failures, indent=2)
    assert evidence["layer1_profiling_importers"] == []
    assert evidence["live_import"]["profiling"] == []


def test_g9_policy_resolved_configuration_matches_frozen_baseline() -> None:
    failures, evidence = contracts.check_g9(REPO)
    assert failures == [], json.dumps(failures, indent=2)
    assert all(evidence["presence_values"].values())


def test_b2_resolves_shared_position_to_one_physical_owner() -> None:
    failures, evidence = contracts.check_b2(REPO)
    assert failures == [], json.dumps(failures, indent=2)
    assert len(evidence["origins"]) == 5
