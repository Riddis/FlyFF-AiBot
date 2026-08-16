from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "docs/migration/tools/phase6_map_profiles.py"
SPEC = importlib.util.spec_from_file_location("phase6_map_profiles", TOOL)
assert SPEC is not None and SPEC.loader is not None
profiles = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = profiles
SPEC.loader.exec_module(profiles)


def test_profiles_are_canonical_typed_immutable_and_exact() -> None:
    failures, evidence = profiles.check_profiles(REPO)
    assert failures == [], json.dumps(failures, indent=2)
    assert evidence["live"] == [2, 2.0]
    assert evidence["live_types"] == ["int", "float"]
    assert evidence["simulator"] == [0, 2]
    assert evidence["simulator_types"] == ["int", "int"]
    assert evidence["immutable"] is True


def test_production_loaders_use_named_profiles_and_overrides_still_win() -> None:
    failures, evidence = profiles.check_wiring(REPO)
    assert failures == [], json.dumps(failures, indent=2)
    assert evidence["live_profile_reference_count"] == 2
    assert evidence["simulator_profile_reference_count"] == 3
    assert evidence["live_override"] == {"safe_exact": True, "teleport": 1.0}
    assert evidence["simulator_override"] == {"safe_exact": True, "teleport": 3.0}


def test_g11_preserves_raw_tower_copies_and_marker() -> None:
    failures, evidence = profiles.check_g11(REPO)
    assert failures == [], json.dumps(failures, indent=2)
    assert all(evidence["pairs"].values())
    assert evidence["marker_present"] is True
    assert {Path(path).name: value for path, value in evidence["hashes"].items()} == profiles.G11_HASHES


def test_g12_preserves_both_goldens_and_map6_stays_diagnostic() -> None:
    failures, evidence = profiles.check_g12(REPO)
    assert failures == [], json.dumps(failures, indent=2)
    assert evidence["fixture_matches"]["map_live.json"] is True
    assert evidence["fixture_matches"]["map_simulator.json"] is True
    assert evidence["live"] == {
        "obstacle_radius_cells": 2,
        "teleport_radius_cells": 2,
        "counts": {"forbidden": 49, "safe_traversable": 52_071, "traversable": 59_818},
    }
    assert evidence["simulator"] == {
        "obstacle_radius_cells": 0,
        "teleport_radius_cells": 2,
        "counts": {"forbidden": 49, "safe_traversable": 59_726, "traversable": 59_818},
    }
    assert evidence["map6"] == {
        "diagnostic_only": True,
        "safe_mask_xor_count": 7_655,
        "fixture_match": True,
    }
