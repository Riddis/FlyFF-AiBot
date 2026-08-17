"""Phase-9 pickle-compatibility hardening: an independent, deliberately
hardcoded pin of the Phase-2 G4 contract literals.

``test_phase2_fingerprints.py`` intentionally hardcodes none of these
values -- it checks that the current source agrees with
``docs/migration/PHASE2_FINGERPRINTS.toml``, the single frozen-value file,
so there is exactly one place to look for a pin. That design is correct
for its purpose, but it cannot catch the one scenario this file exists
for: if the TOML pin and the source were ever changed *together*, silently
drifting to a new, wrong value, ``test_g4_contract_fingerprints_match_
current_source`` would still pass (source and TOML would still agree with
each other) even though the actual contract had moved.

This file exists to close exactly that gap for the values Phase-7's root
collapse and Phase-9's navigation extraction moved across the most source
paths: it recomputes each value directly from the live source (never from
the TOML) and compares it against literals written directly into this
file's own source code -- the true historical values, unaffected by
Phase-7/9 renaming their owning modules. If this file's own hardcoded
values are ever legitimately changed (a genuine model-contract change,
not a path move), that change must be as visible and deliberate in code
review as touching the frozen TOML itself -- which is the point."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "docs/migration/tools/phase2_fingerprints.py"
SPEC = importlib.util.spec_from_file_location("phase9_g4_literal_hardening_fingerprints", TOOL)
assert SPEC is not None and SPEC.loader is not None
fingerprints = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fingerprints
SPEC.loader.exec_module(fingerprints)

# Hardcoded independently of docs/migration/PHASE2_FINGERPRINTS.toml on
# purpose -- see module docstring.
EXPECTED_OBSERVATION_SCHEMA_ID = "native-unified-923-v4"
EXPECTED_OBSERVATION_SCHEMA_HASH = "F2D568C1C4A4B5F577C9C2E36A37B1C5533C2CE28D415846C3B68EC293C84609"
EXPECTED_RAW_OBSERVATION_SIZE = 923
EXPECTED_POLICY_ACTION_NVECS = [3, 3]
EXPECTED_SIDECAR_SIZE = 5
EXPECTED_POLICY_INPUT_SIZE = 928
EXPECTED_MODEL_CONTRACT_METADATA_VERSION = 2
EXPECTED_PHYSICS_VERSION = "live_calibrated_arc"


def test_g4_literals_match_hardcoded_historical_values_independent_of_the_toml_pin() -> None:
    probe = fingerprints.probe_root(REPO, ".")

    assert probe["OBSERVATION_SCHEMA_ID"] == EXPECTED_OBSERVATION_SCHEMA_ID
    assert probe["OBSERVATION_SCHEMA_HASH"] == EXPECTED_OBSERVATION_SCHEMA_HASH
    # Recomputed (not merely declared) hash must also match -- catches
    # semantic descriptor drift, not just a copy-pasted constant.
    assert probe["OBSERVATION_SCHEMA_HASH_RECOMPUTED"] == EXPECTED_OBSERVATION_SCHEMA_HASH
    assert probe["RAW_OBSERVATION_SIZE"] == EXPECTED_RAW_OBSERVATION_SIZE
    assert list(probe["POLICY_ACTION_NVECS"]) == EXPECTED_POLICY_ACTION_NVECS
    assert probe["SIDECAR_SIZE"] == EXPECTED_SIDECAR_SIZE
    assert probe["POLICY_INPUT_SIZE"] == EXPECTED_POLICY_INPUT_SIZE
    assert probe["RAW_OBSERVATION_SIZE"] + probe["SIDECAR_SIZE"] == EXPECTED_POLICY_INPUT_SIZE
    assert probe["MODEL_CONTRACT_METADATA_VERSION"] == EXPECTED_MODEL_CONTRACT_METADATA_VERSION
    assert probe["MOVEMENT_PHYSICS_MODEL_ID"] == EXPECTED_PHYSICS_VERSION


def test_g4_hardcoded_pin_agrees_with_the_frozen_toml_pin() -> None:
    """The two pins (this file's hardcoded literals and
    PHASE2_FINGERPRINTS.toml) are independent by construction, but they
    must still describe the same contract -- if they ever diverge, that is
    itself a finding, not something to silently reconcile."""
    fp = fingerprints.load_fingerprints(REPO)
    g4 = fp["g4"]
    assert g4["observation_schema_id"]["value"] == EXPECTED_OBSERVATION_SCHEMA_ID
    assert g4["observation_schema_hash"]["value"] == EXPECTED_OBSERVATION_SCHEMA_HASH
    assert list(g4["policy_action_nvecs"]["value"]) == EXPECTED_POLICY_ACTION_NVECS
    assert g4["sidecar_size"]["value"] == EXPECTED_SIDECAR_SIZE
    assert g4["policy_input_size"]["value"] == EXPECTED_POLICY_INPUT_SIZE
    assert g4["raw_observation_size"]["value"] == EXPECTED_RAW_OBSERVATION_SIZE
    assert g4["model_contract_metadata_version"]["value"] == EXPECTED_MODEL_CONTRACT_METADATA_VERSION
    assert g4["physics_version"]["value"] == EXPECTED_PHYSICS_VERSION
