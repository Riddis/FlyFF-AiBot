"""Proves what the REAL Basic-stage checkpoint save call sites
(`simulator/tools/RUN_CANONICAL_BASIC.py`'s bootstrap and milestone saves,
both via `simulator.basic_training.save_checkpoint_with_provenance`)
actually record in provenance -- not a synthetic/generic manifest.

Regression target: before this fix, `save_checkpoint_with_provenance` never
passed an `architecture_contract` through to `build_run_manifest`, so every
Basic checkpoint's `.provenance.json` silently inherited `build_run_manifest`'s
own historical default (`SplitSteeringNavigationPolicy`, 928-value
navigation-sidecar input) even though Basic actually trains
`SplitFarmingTargetEventPolicy` over the raw 923-value observation with a
`MultiDiscrete([13, 3])` action contract and no steering action at all.
"""

from __future__ import annotations

import json
from pathlib import Path

from farming.actions import FarmingEvent
from farming.observation import DIRECT_ACTOR_SLOTS
from navigation.navigation_evidence import RAW_OBSERVATION_SIZE
from simulator.basic_training import build_fresh_basic_policy, save_checkpoint_with_provenance
from simulator.navigation_subpolicy import (
    FROZEN_NAVIGATION_CHECKPOINT_PATH,
    FROZEN_NAVIGATION_CHECKPOINT_SHA256,
    farming_policy_architecture_contract,
)


def _save_real_basic_checkpoint(tmp_path: Path, *, milestone: str = "bootstrap", starting_checkpoint: str | None = None) -> dict:
    """Exercises the exact call shape RUN_CANONICAL_BASIC.py uses: a real
    `build_fresh_basic_policy` model, saved through
    `save_checkpoint_with_provenance` with `architecture_contract=
    farming_policy_architecture_contract()` -- not a hand-built manifest
    dict, so this proves what the real save call site actually emits."""
    model = build_fresh_basic_policy(seed=0, device="cpu")
    checkpoint_path = tmp_path / f"canonical_basic_{milestone}.zip"
    save_checkpoint_with_provenance(
        model, checkpoint_path, stage="basic", milestone=milestone, seeds=0, config={},
        architecture_contract=farming_policy_architecture_contract(),
        starting_checkpoint=starting_checkpoint,
    )
    manifest_path = checkpoint_path.with_suffix("").with_suffix(".provenance.json")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def test_basic_bootstrap_save_records_the_actual_farming_target_event_architecture(tmp_path: Path) -> None:
    manifest = _save_real_basic_checkpoint(tmp_path, milestone="bootstrap")
    contract = manifest["architecture_contract"]
    assert contract["policy_class"] == "SplitFarmingTargetEventPolicy"
    assert contract["raw_observation_size"] == RAW_OBSERVATION_SIZE == 923
    assert f"MultiDiscrete([{DIRECT_ACTOR_SLOTS + 1}, {len(FarmingEvent)}])" in contract["action_contract"]


def test_basic_bootstrap_save_records_frozen_navigator_path_and_sha(tmp_path: Path) -> None:
    manifest = _save_real_basic_checkpoint(tmp_path, milestone="bootstrap")
    contract = manifest["architecture_contract"]
    assert contract["navigation_checkpoint_path"] == str(FROZEN_NAVIGATION_CHECKPOINT_PATH)
    assert contract["navigation_checkpoint_sha256"] == FROZEN_NAVIGATION_CHECKPOINT_SHA256


def test_basic_bootstrap_save_does_not_report_the_retired_steering_navigation_policy(tmp_path: Path) -> None:
    manifest = _save_real_basic_checkpoint(tmp_path, milestone="bootstrap")
    contract = manifest["architecture_contract"]
    assert contract["policy_class"] != "SplitSteeringNavigationPolicy"
    assert contract["raw_observation_size"] != 928
    assert contract.get("policy_input_size") != 928
    assert "MultiDiscrete([3, 3])" not in contract.get("action_contract", "")


def test_basic_milestone_round_save_also_records_the_correct_architecture_contract(tmp_path: Path) -> None:
    """The milestone/DAgger-round save site is a second, independent call to
    save_checkpoint_with_provenance -- proves it was fixed too, not just
    bootstrap."""
    prior = tmp_path / "canonical_basic_bootstrap.zip"
    manifest = _save_real_basic_checkpoint(tmp_path, milestone="milestone_001", starting_checkpoint=str(prior))
    contract = manifest["architecture_contract"]
    assert contract["policy_class"] == "SplitFarmingTargetEventPolicy"
    assert contract["raw_observation_size"] == 923
    assert manifest["starting_checkpoint"] == str(prior)
    assert manifest["fresh_initialization"] is False
