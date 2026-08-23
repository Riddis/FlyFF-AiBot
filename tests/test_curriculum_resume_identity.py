"""Proves the resume-identity gate (`simulator.curriculum_resume_identity`)
rejects incompatible historical AND same-generation-but-wrong state, not
just a different architecture generation.

Blocker A (final pre-merge remediation, 2026-08-23): an earlier version of
this gate only checked architecture-generation-level facts (action schema,
observation size, frozen-navigation SHA) -- a cache from the SAME
architecture but a DIFFERENT checkpoint, parent, curriculum stage, manifest,
or evaluation configuration was still silently reusable. This module now
also validates checkpoint identity by CONTENT (SHA-256, never path alone),
parent identity, curriculum stage, manifest content, and a canonical
evaluation-config fingerprint -- and writes/reads current-generation state
under a namespaced path (`current_generation_path`) so historical evidence
under the old filename is never read, written, or archived-then-overwritten
by current code at all.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from simulator.curriculum_resume_identity import (
    checkpoint_identity,
    current_generation_path,
    evaluation_cache_identity,
    identity_mismatch_reason,
    load_cached_evaluation_if_current,
    load_resumable_round_reports,
    manifest_identity,
    round_identity,
    round_record_validity_reason,
    sha256_of_file,
    stable_config_fingerprint,
    with_evaluation_cache_identity,
    with_round_identity,
)
from simulator.navigation_subpolicy import FROZEN_NAVIGATION_CHECKPOINT_SHA256

BEGINNER = importlib.import_module("simulator.tools.RUN_CANONICAL_BEGINNER")
INTERMEDIATE = importlib.import_module("simulator.tools.RUN_CANONICAL_INTERMEDIATE")
ADVANCED = importlib.import_module("simulator.tools.RUN_CANONICAL_ADVANCED")


def _write(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    return path


# --- basic content-identity primitives ------------------------------------


def test_sha256_of_file_is_content_based(tmp_path: Path):
    a = _write(tmp_path / "a.zip", b"checkpoint bytes v1")
    b = _write(tmp_path / "b.zip", b"checkpoint bytes v1")  # same content, different path
    c = _write(tmp_path / "c.zip", b"checkpoint bytes v2")  # same-ish name pattern, different content
    assert sha256_of_file(a) == sha256_of_file(b)
    assert sha256_of_file(a) != sha256_of_file(c)


def test_checkpoint_identity_is_content_based_not_path_alone(tmp_path: Path):
    path = tmp_path / "canonical_beginner_ppo_010k.zip"
    _write(path, b"original bytes")
    identity_v1 = checkpoint_identity(path)
    _write(path, b"replaced bytes -- same path, different content")
    identity_v2 = checkpoint_identity(path)
    assert identity_v1["path"] == identity_v2["path"]
    assert identity_v1["sha256"] != identity_v2["sha256"], "a same-named file with different bytes must have a different identity"


def test_checkpoint_identity_is_none_for_missing_or_not_yet_materialized_path(tmp_path: Path):
    assert checkpoint_identity(None) is None
    assert checkpoint_identity(tmp_path / "does_not_exist.zip") is None


def test_manifest_identity_is_content_based(tmp_path: Path):
    m1 = _write(tmp_path / "heldout.json", b'{"layouts": {}}')
    id1 = manifest_identity(m1)
    _write(m1, b'{"layouts": {"new_layout": {}}}')
    id2 = manifest_identity(m1)
    assert id1["content_sha256"] != id2["content_sha256"]


def test_stable_config_fingerprint_is_order_independent():
    a = stable_config_fingerprint({"episode_seconds": 150.0, "seeds": [0, 1]})
    b = stable_config_fingerprint({"seeds": [0, 1], "episode_seconds": 150.0})
    assert a == b


def test_stable_config_fingerprint_changes_with_content():
    a = stable_config_fingerprint({"episode_seconds": 150.0})
    b = stable_config_fingerprint({"episode_seconds": 999.0})
    assert a != b


def test_current_generation_path_inserts_namespace_before_suffix():
    p = current_generation_path("simulator/evaluations/canonical_beginner_run_summary.json")
    assert p.name == "canonical_beginner_run_summary.target_event_v1.json"


def test_frozen_navigation_sha_matches_the_real_frozen_checkpoint():
    identity = round_identity(stage="early", declared_parent_checkpoint="x", current_checkpoint="y")
    assert identity["navigation_checkpoint_sha256"] == FROZEN_NAVIGATION_CHECKPOINT_SHA256


# --- round identity: identity_mismatch_reason + round_record_validity_reason ---


@pytest.fixture
def parent_checkpoint(tmp_path: Path) -> Path:
    return _write(tmp_path / "canonical_basic_graduated.zip", b"parent checkpoint bytes")


@pytest.fixture
def current_checkpoint(tmp_path: Path) -> Path:
    return _write(tmp_path / "canonical_beginner_ppo_010k.zip", b"round 1 checkpoint bytes")


def _valid_round_record(parent_checkpoint: Path, current_checkpoint: Path, *, stage: str = "early") -> dict:
    return with_round_identity(
        {"round": 1, "consecutive_passes": 1, "carried_forward_checkpoint": str(current_checkpoint.resolve())},
        stage=stage, declared_parent_checkpoint=parent_checkpoint, current_checkpoint=current_checkpoint,
    )


def test_matching_round_record_is_valid(parent_checkpoint, current_checkpoint):
    record = _valid_round_record(parent_checkpoint, current_checkpoint)
    reason = round_record_validity_reason(record, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert reason is None


def test_round_record_with_wrong_stage_is_rejected(parent_checkpoint, current_checkpoint):
    record = _valid_round_record(parent_checkpoint, current_checkpoint, stage="early")
    reason = round_record_validity_reason(record, stage="intermediate", declared_parent_checkpoint=parent_checkpoint)
    assert reason is not None
    assert "curriculum_stage" in reason


def test_round_record_with_wrong_parent_path_is_rejected(parent_checkpoint, current_checkpoint, tmp_path):
    record = _valid_round_record(parent_checkpoint, current_checkpoint)
    other_parent = _write(tmp_path / "some_other_parent.zip", b"a different parent entirely")
    reason = round_record_validity_reason(record, stage="early", declared_parent_checkpoint=other_parent)
    assert reason is not None
    assert "declared_parent_checkpoint" in reason


def test_round_record_with_wrong_parent_sha_is_rejected(parent_checkpoint, current_checkpoint):
    """SAME parent path, but the parent checkpoint's bytes changed since
    this round was recorded (e.g. the graduated Basic checkpoint was
    regenerated/quarantined) -- must reject even though the path matches."""
    record = _valid_round_record(parent_checkpoint, current_checkpoint)
    _write(parent_checkpoint, b"parent checkpoint bytes -- REPLACED")
    reason = round_record_validity_reason(record, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert reason is not None
    assert "declared_parent_checkpoint_sha256" in reason


def test_round_record_with_stale_checkpoint_path_is_rejected(parent_checkpoint, current_checkpoint):
    record = _valid_round_record(parent_checkpoint, current_checkpoint)
    current_checkpoint.unlink()
    reason = round_record_validity_reason(record, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert reason is not None
    assert "no longer exists" in reason


def test_round_record_with_changed_checkpoint_bytes_is_rejected(parent_checkpoint, current_checkpoint):
    """SAME path, DIFFERENT bytes -- the exact scenario a path-only resume
    check cannot catch."""
    record = _valid_round_record(parent_checkpoint, current_checkpoint)
    _write(current_checkpoint, b"round 1 checkpoint bytes -- REPLACED WITH DIFFERENT MODEL")
    reason = round_record_validity_reason(record, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert reason is not None
    assert "bytes changed" in reason


def test_round_record_with_legacy_action_schema_is_rejected(parent_checkpoint, current_checkpoint):
    record = _valid_round_record(parent_checkpoint, current_checkpoint)
    record["identity"]["policy_action_nvec"] = [3, 3]  # retired steering+event contract
    reason = round_record_validity_reason(record, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert reason is not None
    assert "policy_action_nvec" in reason


def test_round_record_with_no_identity_is_rejected(parent_checkpoint, current_checkpoint):
    record = {"round": 4, "consecutive_passes": 2, "carried_forward_checkpoint": str(current_checkpoint)}
    reason = round_record_validity_reason(record, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert reason is not None
    assert "no identity recorded" in reason


def test_round_record_missing_carried_forward_checkpoint_is_rejected(parent_checkpoint, current_checkpoint):
    record = _valid_round_record(parent_checkpoint, current_checkpoint)
    del record["carried_forward_checkpoint"]
    reason = round_record_validity_reason(record, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert reason is not None


# --- load_resumable_round_reports: real temporary files -------------------


def test_load_resumable_round_reports_accepts_a_valid_matching_chain(tmp_path, parent_checkpoint, current_checkpoint):
    summary_path = current_generation_path(tmp_path / "canonical_beginner_run_summary.json")
    summary_path.write_text(json.dumps([_valid_round_record(parent_checkpoint, current_checkpoint)]), encoding="utf-8")
    result = load_resumable_round_reports(summary_path, log=lambda _m: None, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert len(result) == 1


def test_load_resumable_round_reports_rejects_orphaned_current_checkpoint(tmp_path, parent_checkpoint, current_checkpoint):
    """Task section 13: a validated round record whose checkpoint file has
    since vanished must not silently resume."""
    record = _valid_round_record(parent_checkpoint, current_checkpoint)
    current_checkpoint.unlink()
    summary_path = current_generation_path(tmp_path / "canonical_beginner_run_summary.json")
    summary_path.write_text(json.dumps([record]), encoding="utf-8")
    result = load_resumable_round_reports(summary_path, log=lambda _m: None, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert result == []


def test_load_resumable_round_reports_never_mutates_the_file_on_rejection(tmp_path, parent_checkpoint, current_checkpoint):
    record = {"round": 4, "consecutive_passes": 2, "carried_forward_checkpoint": str(current_checkpoint)}  # no identity: legacy shape
    summary_path = current_generation_path(tmp_path / "canonical_beginner_run_summary.json")
    original_bytes = json.dumps([record]).encode("utf-8")
    summary_path.write_bytes(original_bytes)
    load_resumable_round_reports(summary_path, log=lambda _m: None, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert summary_path.read_bytes() == original_bytes, "rejection must never mutate the file it read"


def test_load_resumable_round_reports_missing_file_returns_empty(tmp_path, parent_checkpoint):
    result = load_resumable_round_reports(tmp_path / "does_not_exist.json", log=lambda _m: None, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert result == []


def test_legacy_filename_at_old_path_is_never_read_by_the_current_loader(tmp_path, parent_checkpoint, current_checkpoint):
    """The un-namespaced historical filename is a completely different
    path from what load_resumable_round_reports is ever called with in
    the real runners (current_generation_path(...)) -- proves the
    separation directly: writing legacy-shaped content to the OLD path
    and calling the loader with the NEW (namespaced, nonexistent) path
    returns empty, never touching or even opening the old file."""
    legacy_path = tmp_path / "canonical_beginner_run_summary.json"
    legacy_path.write_text(json.dumps([{"round": 4, "consecutive_passes": 2}]), encoding="utf-8")
    current_path = current_generation_path(legacy_path)
    assert not current_path.exists()
    result = load_resumable_round_reports(current_path, log=lambda _m: None, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert result == []
    assert legacy_path.read_text(encoding="utf-8") == json.dumps([{"round": 4, "consecutive_passes": 2}])


# --- evaluation-cache identity: real temporary files -----------------------


@pytest.fixture
def heldout_manifest(tmp_path: Path) -> Path:
    return _write(tmp_path / "early_heldout.json", b'{"layouts": {"L1": {}}}')


def _cache_identity_kwargs(parent_checkpoint, evaluated_checkpoint, heldout_manifest, *, stage="early", role="heldout", episode_seconds=150.0):
    return dict(
        stage=stage, declared_parent_checkpoint=parent_checkpoint, evaluated_checkpoint=evaluated_checkpoint,
        evaluation_role=role, manifests={"heldout": str(heldout_manifest)},
        config={"episode_seconds": episode_seconds, "seeds": [0, 1]},
    )


def test_matching_cache_identity_is_accepted(tmp_path, parent_checkpoint, current_checkpoint, heldout_manifest):
    kwargs = _cache_identity_kwargs(parent_checkpoint, current_checkpoint, heldout_manifest)
    cache_path = current_generation_path(tmp_path / "canonical_x_pre_rehearsal_heldout.json")
    report = with_evaluation_cache_identity({"n_episodes": 4}, **kwargs)
    cache_path.write_text(json.dumps(report), encoding="utf-8")
    expected = evaluation_cache_identity(**kwargs)
    result = load_cached_evaluation_if_current(cache_path, log=lambda _m: None, expected_identity=expected)
    assert result is not None
    assert result["n_episodes"] == 4


def test_cache_with_wrong_evaluated_checkpoint_sha_is_rejected(tmp_path, parent_checkpoint, current_checkpoint, heldout_manifest):
    kwargs = _cache_identity_kwargs(parent_checkpoint, current_checkpoint, heldout_manifest)
    cache_path = current_generation_path(tmp_path / "canonical_x_pre_rehearsal_heldout.json")
    report = with_evaluation_cache_identity({"n_episodes": 4}, **kwargs)
    cache_path.write_text(json.dumps(report), encoding="utf-8")
    _write(current_checkpoint, b"round 1 checkpoint bytes -- REPLACED")  # same path, different bytes
    expected = evaluation_cache_identity(**kwargs)  # recomputed fresh -- picks up the new SHA
    result = load_cached_evaluation_if_current(cache_path, log=lambda _m: None, expected_identity=expected)
    assert result is None


def test_cache_with_wrong_parent_is_rejected(tmp_path, parent_checkpoint, current_checkpoint, heldout_manifest, tmp_path_factory):
    kwargs = _cache_identity_kwargs(parent_checkpoint, current_checkpoint, heldout_manifest)
    cache_path = current_generation_path(tmp_path / "canonical_x_pre_rehearsal_heldout.json")
    cache_path.write_text(json.dumps(with_evaluation_cache_identity({"n_episodes": 4}, **kwargs)), encoding="utf-8")
    other_parent = _write(tmp_path / "different_parent.zip", b"a genuinely different parent")
    expected = evaluation_cache_identity(**{**kwargs, "declared_parent_checkpoint": other_parent})
    result = load_cached_evaluation_if_current(cache_path, log=lambda _m: None, expected_identity=expected)
    assert result is None


def test_cache_with_wrong_parent_sha_is_rejected(tmp_path, parent_checkpoint, current_checkpoint, heldout_manifest):
    kwargs = _cache_identity_kwargs(parent_checkpoint, current_checkpoint, heldout_manifest)
    cache_path = current_generation_path(tmp_path / "canonical_x_pre_rehearsal_heldout.json")
    cache_path.write_text(json.dumps(with_evaluation_cache_identity({"n_episodes": 4}, **kwargs)), encoding="utf-8")
    _write(parent_checkpoint, b"parent checkpoint bytes -- REPLACED")
    expected = evaluation_cache_identity(**kwargs)  # fresh -- picks up new parent SHA
    result = load_cached_evaluation_if_current(cache_path, log=lambda _m: None, expected_identity=expected)
    assert result is None


def test_cache_with_wrong_stage_is_rejected(tmp_path, parent_checkpoint, current_checkpoint, heldout_manifest):
    kwargs = _cache_identity_kwargs(parent_checkpoint, current_checkpoint, heldout_manifest, stage="early")
    cache_path = current_generation_path(tmp_path / "canonical_x_pre_rehearsal_heldout.json")
    cache_path.write_text(json.dumps(with_evaluation_cache_identity({"n_episodes": 4}, **kwargs)), encoding="utf-8")
    expected = evaluation_cache_identity(**{**kwargs, "stage": "intermediate"})
    result = load_cached_evaluation_if_current(cache_path, log=lambda _m: None, expected_identity=expected)
    assert result is None


def test_cache_with_wrong_manifest_hash_is_rejected(tmp_path, parent_checkpoint, current_checkpoint, heldout_manifest):
    kwargs = _cache_identity_kwargs(parent_checkpoint, current_checkpoint, heldout_manifest)
    cache_path = current_generation_path(tmp_path / "canonical_x_pre_rehearsal_heldout.json")
    cache_path.write_text(json.dumps(with_evaluation_cache_identity({"n_episodes": 4}, **kwargs)), encoding="utf-8")
    _write(heldout_manifest, b'{"layouts": {"L1": {}, "L2": {}}}')  # manifest content changed
    expected = evaluation_cache_identity(**kwargs)  # fresh -- picks up new manifest hash
    result = load_cached_evaluation_if_current(cache_path, log=lambda _m: None, expected_identity=expected)
    assert result is None


def test_cache_with_wrong_evaluation_config_is_rejected(tmp_path, parent_checkpoint, current_checkpoint, heldout_manifest):
    kwargs = _cache_identity_kwargs(parent_checkpoint, current_checkpoint, heldout_manifest, episode_seconds=150.0)
    cache_path = current_generation_path(tmp_path / "canonical_x_pre_rehearsal_heldout.json")
    cache_path.write_text(json.dumps(with_evaluation_cache_identity({"n_episodes": 4}, **kwargs)), encoding="utf-8")
    expected = evaluation_cache_identity(**{**kwargs, "config": {"episode_seconds": 999.0, "seeds": [0, 1]}})
    result = load_cached_evaluation_if_current(cache_path, log=lambda _m: None, expected_identity=expected)
    assert result is None


def test_cache_with_wrong_evaluation_role_is_rejected(tmp_path, parent_checkpoint, current_checkpoint, heldout_manifest):
    """A cache produced for 'heldout' must never satisfy a 'challenge' (or
    any other role) lookup, even against the identical checkpoint."""
    kwargs = _cache_identity_kwargs(parent_checkpoint, current_checkpoint, heldout_manifest, role="heldout")
    cache_path = current_generation_path(tmp_path / "canonical_x_pre_rehearsal_heldout.json")
    cache_path.write_text(json.dumps(with_evaluation_cache_identity({"n_episodes": 4}, **kwargs)), encoding="utf-8")
    expected = evaluation_cache_identity(**{**kwargs, "evaluation_role": "challenge"})
    result = load_cached_evaluation_if_current(cache_path, log=lambda _m: None, expected_identity=expected)
    assert result is None


def test_cache_with_no_identity_is_rejected(tmp_path, parent_checkpoint, current_checkpoint, heldout_manifest):
    cache_path = current_generation_path(tmp_path / "canonical_x_pre_rehearsal_heldout.json")
    cache_path.write_text(json.dumps({"n_episodes": 4}), encoding="utf-8")  # no "identity" key at all
    expected = evaluation_cache_identity(**_cache_identity_kwargs(parent_checkpoint, current_checkpoint, heldout_manifest))
    result = load_cached_evaluation_if_current(cache_path, log=lambda _m: None, expected_identity=expected)
    assert result is None


def test_cache_with_legacy_event_only_action_schema_is_rejected(tmp_path, parent_checkpoint, current_checkpoint, heldout_manifest):
    kwargs = _cache_identity_kwargs(parent_checkpoint, current_checkpoint, heldout_manifest)
    cache_path = current_generation_path(tmp_path / "canonical_x_pre_rehearsal_heldout.json")
    report = with_evaluation_cache_identity({"n_episodes": 4}, **kwargs)
    report["identity"]["policy_action_nvec"] = [3, 3]
    cache_path.write_text(json.dumps(report), encoding="utf-8")
    expected = evaluation_cache_identity(**kwargs)
    result = load_cached_evaluation_if_current(cache_path, log=lambda _m: None, expected_identity=expected)
    assert result is None


def test_missing_cache_file_returns_none(tmp_path, parent_checkpoint, current_checkpoint, heldout_manifest):
    expected = evaluation_cache_identity(**_cache_identity_kwargs(parent_checkpoint, current_checkpoint, heldout_manifest))
    assert load_cached_evaluation_if_current(tmp_path / "does_not_exist.json", log=lambda _m: None, expected_identity=expected) is None


# --- each canonical runner actually wires the new API in -----------------


@pytest.mark.parametrize("module,expected_declared_parent_attr", [
    (BEGINNER, "GRADUATED_BASIC_CHECKPOINT"),
    (INTERMEDIATE, "GRADUATED_BEGINNER_CHECKPOINT"),
    (ADVANCED, "GRADUATED_INTERMEDIATE_CHECKPOINT"),
])
def test_runner_source_calls_the_new_resume_identity_gate(module, expected_declared_parent_attr):
    """Source-level proof (not just that the shared gate works in
    isolation) that each canonical runner resume-gates through the CURRENT
    (content-based, generation-namespaced) API via its own
    `_resume_round_state` helper (which `main()` calls), and no longer
    contains the removed orphan-checkpoint glob-resume mechanism anywhere
    in the module."""
    import inspect

    whole_module_source = inspect.getsource(module)
    assert "load_resumable_round_reports" in whole_module_source
    assert "load_cached_evaluation_if_current" in whole_module_source
    assert "current_generation_path" in whole_module_source
    assert "_resume_round_state" in inspect.getsource(module.main)
    resume_state_source = inspect.getsource(module._resume_round_state)
    assert f"declared_parent_checkpoint={expected_declared_parent_attr}" in inspect.getsource(module.main)
    assert "load_resumable_round_reports" in resume_state_source
    assert "existing_rounds" not in whole_module_source, "orphan-checkpoint glob resume must be fully removed"
    assert "MODELS_DIR.glob" not in resume_state_source, "_resume_round_state must never inspect the filesystem for checkpoints"
