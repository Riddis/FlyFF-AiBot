from __future__ import annotations

import json
from pathlib import Path

from simulator.movement_kernel import MOVEMENT_PHYSICS_MODEL_ID
from simulator.navigation_history import POLICY_INPUT_SIZE, RAW_OBSERVATION_SIZE
from simulator.run_provenance import build_run_manifest, capture_git_state, write_run_manifest


def test_fresh_initialization_is_recorded_explicitly_true_when_no_starting_checkpoint() -> None:
    manifest = build_run_manifest(stage="basic", milestone="bootstrap", seeds=[0], config={}, starting_checkpoint=None)
    assert manifest["fresh_initialization"] is True
    assert manifest["starting_checkpoint"] is None


def test_fresh_initialization_is_false_when_continuing_a_checkpoint() -> None:
    manifest = build_run_manifest(
        stage="beginner", milestone="ppo_chunk", seeds=0, config={}, starting_checkpoint="models/canonical_basic_milestone_003.zip",
    )
    assert manifest["fresh_initialization"] is False
    assert manifest["starting_checkpoint"] == "models/canonical_basic_milestone_003.zip"


def test_manifest_includes_architecture_contract_and_git_state() -> None:
    manifest = build_run_manifest(stage="basic", milestone="bootstrap", seeds=[0], config={})
    assert manifest["architecture_contract"]["policy_input_size"] == POLICY_INPUT_SIZE
    assert manifest["architecture_contract"]["raw_observation_size"] == RAW_OBSERVATION_SIZE
    assert "commit" in manifest["git"]


def test_manifest_records_the_live_movement_physics_model() -> None:
    """2026-08-13: every new manifest must explicitly record which physics
    kernel produced the run, so old (legacy_recorded_iid, by absence) and
    new (live_calibrated_arc) results are never conflated when comparing
    checkpoints later."""
    manifest = build_run_manifest(stage="basic", milestone="bootstrap", seeds=[0], config={})
    assert manifest["architecture_contract"]["movement_physics_model"] == MOVEMENT_PHYSICS_MODEL_ID
    assert MOVEMENT_PHYSICS_MODEL_ID == "live_calibrated_arc"


def test_capture_git_state_never_raises() -> None:
    state = capture_git_state()
    assert "available" in state


def test_write_run_manifest_writes_next_to_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "canonical_basic_bootstrap.zip"
    checkpoint.write_bytes(b"not a real checkpoint, just a path to write next to")
    manifest = build_run_manifest(stage="basic", milestone="bootstrap", seeds=[0], config={"epochs": 2})
    manifest_path = write_run_manifest(checkpoint, manifest)
    assert manifest_path.name == "canonical_basic_bootstrap.provenance.json"
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["config"]["epochs"] == 2


def test_manifest_includes_seeds_as_a_list_even_when_given_a_single_int() -> None:
    manifest = build_run_manifest(stage="basic", milestone="bootstrap", seeds=42, config={})
    assert manifest["seeds"] == [42]
