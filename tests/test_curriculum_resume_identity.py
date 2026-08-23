"""Proves the resume-identity gate (`simulator.curriculum_resume_identity`)
actually rejects incompatible historical state rather than resuming from it
merely because a same-named file exists on disk -- the blocker: legacy
Beginner/Intermediate/Advanced run summaries and evaluation caches could be
silently reused (inheriting `consecutive_passes`, a `carried_forward_
checkpoint` path, or a cached evaluation) even when they belonged to an
incompatible historical lineage (steering+event, event-only, or any earlier
action/observation contract) rather than the CURRENT learned-target-
selection lineage (`MultiDiscrete([13, 3])`, raw 923-value observation).

Also proves each canonical runner's own resume block actually calls through
to this gate (not just that the gate works in isolation)."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from simulator.curriculum_resume_identity import (
    archive_legacy_artifact,
    current_curriculum_generation_identity,
    identity_mismatch_reason,
    load_cached_report_if_current,
    load_resumable_round_reports,
    with_current_generation_identity,
)
from simulator.navigation_subpolicy import FROZEN_NAVIGATION_CHECKPOINT_SHA256

BEGINNER = importlib.import_module("simulator.tools.RUN_CANONICAL_BEGINNER")
INTERMEDIATE = importlib.import_module("simulator.tools.RUN_CANONICAL_INTERMEDIATE")
ADVANCED = importlib.import_module("simulator.tools.RUN_CANONICAL_ADVANCED")


def _obsolete_legacy_round(*, consecutive_passes: int = 2) -> dict:
    """Shaped like a REAL tracked pre-2026-08-23 round entry: no
    generation_identity at all, and an obsolete flyff_farming_simulator/
    checkpoint path from before the 2026-08-21 project-root collapse."""
    return {
        "round": 4,
        "consecutive_passes": consecutive_passes,
        "carried_forward_checkpoint": r"C:\Users\Ridd\Documents\Repos\Flyff RL\flyff_farming_simulator\models\canonical_beginner_ppo_040k_rehearsed.zip",
        "round_passed_absolute_bar": True,
        "bar_failure_reasons": [],
        "aggregates": {},
    }


def _current_round(*, consecutive_passes: int = 1, checkpoint: str = "models/canonical_beginner_ppo_010k.zip", **extra) -> dict:
    return with_current_generation_identity({
        "round": 1,
        "consecutive_passes": consecutive_passes,
        "carried_forward_checkpoint": checkpoint,
        "round_passed_absolute_bar": True,
        "bar_failure_reasons": [],
        "aggregates": {},
        **extra,
    })


# --- identity_mismatch_reason -------------------------------------------


def test_current_identity_matches_itself():
    assert identity_mismatch_reason(current_curriculum_generation_identity()) is None


def test_missing_identity_is_a_mismatch():
    assert identity_mismatch_reason(None) is not None
    assert identity_mismatch_reason({}) is not None


def test_wrong_action_schema_is_a_mismatch():
    """Legacy/event-only/steering-event action schema must never be
    accepted as current -- e.g. the retired Discrete(len(FarmingEvent))
    event-only contract, or the old MultiDiscrete([3, 3]) steering+event
    contract."""
    stale = current_curriculum_generation_identity()
    stale["policy_action_nvec"] = [3, 3]  # retired steering+event contract
    reason = identity_mismatch_reason(stale)
    assert reason is not None
    assert "policy_action_nvec" in reason


def test_wrong_observation_size_is_a_mismatch():
    stale = current_curriculum_generation_identity()
    stale["raw_observation_size"] = 928  # retired navigation-sidecar width
    reason = identity_mismatch_reason(stale)
    assert reason is not None
    assert "raw_observation_size" in reason


def test_frozen_navigation_sha_mismatch_is_rejected():
    stale = current_curriculum_generation_identity()
    stale["navigation_checkpoint_sha256"] = "0" * 64
    reason = identity_mismatch_reason(stale)
    assert reason is not None
    assert "navigation_checkpoint_sha256" in reason


def test_frozen_navigation_sha_in_identity_matches_the_real_frozen_checkpoint():
    assert current_curriculum_generation_identity()["navigation_checkpoint_sha256"] == FROZEN_NAVIGATION_CHECKPOINT_SHA256


def test_parent_checkpoint_mismatch_is_rejected():
    current = current_curriculum_generation_identity(declared_parent_checkpoint="models/canonical_basic_graduated.zip")
    reason = identity_mismatch_reason(current, expected_declared_parent_checkpoint="models/some_other_quarantined_checkpoint.zip")
    assert reason is not None
    assert "declared_parent_checkpoint" in reason


def test_matching_parent_checkpoint_is_accepted():
    current = current_curriculum_generation_identity(declared_parent_checkpoint="models/canonical_basic_graduated.zip")
    reason = identity_mismatch_reason(current, expected_declared_parent_checkpoint="models/canonical_basic_graduated.zip")
    assert reason is None


# --- load_resumable_round_reports ----------------------------------------


def test_legacy_summary_missing_identity_is_not_resumed(tmp_path: Path):
    summary_path = tmp_path / "canonical_beginner_run_summary.json"
    summary_path.write_text(json.dumps([_obsolete_legacy_round(consecutive_passes=2)]), encoding="utf-8")
    logs: list[str] = []
    result = load_resumable_round_reports(summary_path, log=logs.append)
    assert result == [], "a round with no generation_identity must never be resumed"
    assert any("generation_identity" in msg for msg in logs)


def test_legacy_consecutive_passes_of_two_does_not_carry_forward(tmp_path: Path):
    """Direct proof of the task's own example: consecutive_passes=2 in a
    legacy (identity-mismatched) summary must not seed the current run --
    it must start at 0."""
    summary_path = tmp_path / "canonical_beginner_run_summary.json"
    summary_path.write_text(json.dumps([_obsolete_legacy_round(consecutive_passes=2)]), encoding="utf-8")
    result = load_resumable_round_reports(summary_path, log=lambda _msg: None)
    assert result == []
    # The caller (RUN_CANONICAL_BEGINNER.main) derives consecutive_passes
    # from round_reports[-1]; an empty list means it stays at its
    # initialized value of 0 -- reproduce that derivation here directly.
    consecutive_passes = result[-1]["consecutive_passes"] if result else 0
    assert consecutive_passes == 0


def test_legacy_summary_is_archived_not_mutated(tmp_path: Path):
    summary_path = tmp_path / "canonical_beginner_run_summary.json"
    original_bytes = json.dumps([_obsolete_legacy_round()]).encode("utf-8")
    summary_path.write_bytes(original_bytes)
    load_resumable_round_reports(summary_path, log=lambda _msg: None)
    # The original file must survive completely untouched.
    assert summary_path.read_bytes() == original_bytes
    # A byte-identical archive copy must exist alongside it.
    archives = list(tmp_path.glob("canonical_beginner_run_summary.legacy-*.json"))
    assert len(archives) == 1
    assert archives[0].read_bytes() == original_bytes


def test_current_generation_summary_resumes_normally(tmp_path: Path):
    summary_path = tmp_path / "canonical_beginner_run_summary.json"
    summary_path.write_text(json.dumps([_current_round(consecutive_passes=1)]), encoding="utf-8")
    result = load_resumable_round_reports(summary_path, log=lambda _msg: None)
    assert len(result) == 1
    assert result[0]["consecutive_passes"] == 1


def test_summary_with_stale_action_schema_is_not_resumed(tmp_path: Path):
    """Simulates a retired event-only/steering-event lineage's own summary
    -- has SOME generation_identity (not merely absent), but the wrong
    action contract."""
    stale_round = _current_round()
    stale_round["generation_identity"]["policy_action_nvec"] = [3, 3]
    summary_path = tmp_path / "canonical_beginner_run_summary.json"
    summary_path.write_text(json.dumps([stale_round]), encoding="utf-8")
    result = load_resumable_round_reports(summary_path, log=lambda _msg: None)
    assert result == []


def test_summary_with_frozen_navigation_sha_mismatch_is_not_resumed(tmp_path: Path):
    stale_round = _current_round()
    stale_round["generation_identity"]["navigation_checkpoint_sha256"] = "f" * 64
    summary_path = tmp_path / "canonical_beginner_run_summary.json"
    summary_path.write_text(json.dumps([stale_round]), encoding="utf-8")
    result = load_resumable_round_reports(summary_path, log=lambda _msg: None)
    assert result == []


def test_summary_with_parent_checkpoint_mismatch_is_not_resumed(tmp_path: Path):
    matching_round = with_current_generation_identity(
        {"round": 1, "consecutive_passes": 1, "carried_forward_checkpoint": "x"},
        declared_parent_checkpoint="models/canonical_basic_graduated.zip",
    )
    summary_path = tmp_path / "canonical_beginner_run_summary.json"
    summary_path.write_text(json.dumps([matching_round]), encoding="utf-8")
    result = load_resumable_round_reports(
        summary_path, log=lambda _msg: None, declared_parent_checkpoint="models/some_other_checkpoint.zip",
    )
    assert result == [], "a chain descending from a DIFFERENT declared parent checkpoint must not be resumed"


def test_summary_with_matching_parent_checkpoint_resumes(tmp_path: Path):
    matching_round = with_current_generation_identity(
        {"round": 1, "consecutive_passes": 1, "carried_forward_checkpoint": "x"},
        declared_parent_checkpoint="models/canonical_basic_graduated.zip",
    )
    summary_path = tmp_path / "canonical_beginner_run_summary.json"
    summary_path.write_text(json.dumps([matching_round]), encoding="utf-8")
    result = load_resumable_round_reports(
        summary_path, log=lambda _msg: None, declared_parent_checkpoint="models/canonical_basic_graduated.zip",
    )
    assert len(result) == 1


def test_missing_summary_file_returns_empty_without_error(tmp_path: Path):
    assert load_resumable_round_reports(tmp_path / "does_not_exist.json", log=lambda _msg: None) == []


# --- load_cached_report_if_current (zero-shot / rehearsal eval caches) ---


def test_legacy_cached_evaluation_is_not_reused(tmp_path: Path):
    path = tmp_path / "canonical_beginner_zero_shot_diagnostic.json"
    path.write_text(json.dumps({"per_layout": {}}), encoding="utf-8")  # no generation_identity at all
    result = load_cached_report_if_current(path, log=lambda _msg: None)
    assert result is None


def test_current_cached_evaluation_is_reused(tmp_path: Path):
    path = tmp_path / "canonical_beginner_zero_shot_diagnostic.json"
    report = with_current_generation_identity({"per_layout": {}})
    path.write_text(json.dumps(report), encoding="utf-8")
    result = load_cached_report_if_current(path, log=lambda _msg: None)
    assert result is not None
    assert result["per_layout"] == {}


def test_missing_cache_file_returns_none(tmp_path: Path):
    assert load_cached_report_if_current(tmp_path / "does_not_exist.json", log=lambda _msg: None) is None


def test_legacy_cache_is_archived_not_mutated(tmp_path: Path):
    path = tmp_path / "canonical_beginner_zero_shot_diagnostic.json"
    original_bytes = json.dumps({"per_layout": {}}).encode("utf-8")
    path.write_bytes(original_bytes)
    load_cached_report_if_current(path, log=lambda _msg: None)
    assert path.read_bytes() == original_bytes
    archives = list(tmp_path.glob("canonical_beginner_zero_shot_diagnostic.legacy-*.json"))
    assert len(archives) == 1


# --- archive_legacy_artifact ----------------------------------------------


def test_archive_legacy_artifact_is_byte_identical_copy(tmp_path: Path):
    path = tmp_path / "some_report.json"
    path.write_bytes(b'{"legacy": true}')
    archive_legacy_artifact(path, log=lambda _msg: None)
    archives = list(tmp_path.glob("some_report.legacy-*.json"))
    assert len(archives) == 1
    assert archives[0].read_bytes() == b'{"legacy": true}'
    assert path.read_bytes() == b'{"legacy": true}', "the original file must be untouched"


def test_archive_legacy_artifact_is_a_noop_when_file_does_not_exist(tmp_path: Path):
    archive_legacy_artifact(tmp_path / "does_not_exist.json", log=lambda _msg: None)
    assert list(tmp_path.glob("*.legacy-*")) == []


# --- each canonical runner actually wires the gate in --------------------


@pytest.mark.parametrize("module,expected_declared_parent_attr", [
    (BEGINNER, "GRADUATED_BASIC_CHECKPOINT"),
    (INTERMEDIATE, "GRADUATED_BEGINNER_CHECKPOINT"),
    (ADVANCED, "GRADUATED_INTERMEDIATE_CHECKPOINT"),
])
def test_runner_source_calls_the_resume_identity_gate(module, expected_declared_parent_attr):
    """Source-level proof (not just that the shared gate works in
    isolation) that each canonical runner's main() actually resume-gates
    through it, rather than the raw json.loads(...) it used to use."""
    import inspect

    source = inspect.getsource(module.main)
    assert "load_resumable_round_reports" in source
    assert "load_cached_report_if_current" in source
    assert "with_current_generation_identity" in source
    assert f"declared_parent_checkpoint={expected_declared_parent_attr}" in source
