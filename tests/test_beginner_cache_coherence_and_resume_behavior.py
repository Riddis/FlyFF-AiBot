"""Behavior-level tests (not source-string inspection) driving the ACTUAL
canonical-runner resume/cache-coherence functions -- `_resume_round_state`
and (Beginner-only) `_load_coherent_evaluation_set` -- against real
temporary files, monkeypatching only `EVAL_DIR`/`MODELS_DIR` so the tests
never touch the real repository's `simulator/evaluations/`/`models/`.

Task section 12: Beginner's heldout/unseen/challenge caches must be treated
as ONE coherent set -- a valid heldout + stale unseen + valid challenge
must never be silently combined into "one valid evaluation."

Task section 13: for each of Beginner/Intermediate/Advanced, a same-named
PPO checkpoint file that merely exists on disk (no validated round record)
must NEVER be resumed from -- the stage must start from its explicitly
declared graduated parent checkpoint. A VALID matching round record (same
path AND same content SHA) must resume correctly.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from simulator.curriculum_resume_identity import current_generation_path, with_round_identity

BEGINNER = importlib.import_module("simulator.tools.RUN_CANONICAL_BEGINNER")
INTERMEDIATE = importlib.import_module("simulator.tools.RUN_CANONICAL_INTERMEDIATE")
ADVANCED = importlib.import_module("simulator.tools.RUN_CANONICAL_ADVANCED")


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


# --- section 13: no orphan checkpoint resume, all three stages -----------


@pytest.mark.parametrize("module,parent_name,stage", [
    (BEGINNER, "canonical_basic_graduated.zip", "early"),
    (INTERMEDIATE, "canonical_beginner_graduated.zip", "intermediate"),
    (ADVANCED, "canonical_intermediate_graduated.zip", "advanced"),
])
def test_orphan_checkpoint_with_no_valid_summary_is_never_resumed(tmp_path, module, parent_name, stage):
    declared_parent = _write(tmp_path / parent_name, b"declared parent checkpoint bytes")
    # A same-named, plausible-looking PPO checkpoint file exists on disk --
    # but there is NO round-summary file at all vouching for it.
    orphan_checkpoint = _write(tmp_path / f"canonical_{stage}_ppo_010k.zip", b"an orphan checkpoint nobody validated")
    summary_path = current_generation_path(tmp_path / f"canonical_{stage}_run_summary.json")
    assert not summary_path.exists()

    round_reports, consecutive_passes, current_checkpoint = module._resume_round_state(
        summary_path, declared_parent_checkpoint=declared_parent,
    )

    assert round_reports == []
    assert consecutive_passes == 0
    assert current_checkpoint == declared_parent, (
        f"must start from the explicitly declared graduated parent, not the orphan checkpoint {orphan_checkpoint}"
    )


@pytest.mark.parametrize("module,parent_name,stage", [
    (BEGINNER, "canonical_basic_graduated.zip", "early"),
    (INTERMEDIATE, "canonical_beginner_graduated.zip", "intermediate"),
    (ADVANCED, "canonical_intermediate_graduated.zip", "advanced"),
])
def test_orphan_checkpoint_with_stale_summary_is_never_resumed(tmp_path, module, parent_name, stage):
    """A round-summary file DOES exist, but carries no identity at all
    (the pre-2026-08-23 legacy shape) -- still must not resume, and must
    still fall back to the declared parent rather than the orphan file."""
    declared_parent = _write(tmp_path / parent_name, b"declared parent checkpoint bytes")
    orphan_checkpoint = _write(tmp_path / f"canonical_{stage}_ppo_040k.zip", b"a stale round's leftover checkpoint")
    summary_path = current_generation_path(tmp_path / f"canonical_{stage}_run_summary.json")
    legacy_record = {
        "round": 4, "consecutive_passes": 2, "carried_forward_checkpoint": str(orphan_checkpoint),
    }
    summary_path.write_text(json.dumps([legacy_record]), encoding="utf-8")

    round_reports, consecutive_passes, current_checkpoint = module._resume_round_state(
        summary_path, declared_parent_checkpoint=declared_parent,
    )

    assert round_reports == []
    assert consecutive_passes == 0, "the legacy consecutive_passes=2 must never carry forward"
    assert current_checkpoint == declared_parent


@pytest.mark.parametrize("module,parent_name,stage", [
    (BEGINNER, "canonical_basic_graduated.zip", "early"),
    (INTERMEDIATE, "canonical_beginner_graduated.zip", "intermediate"),
    (ADVANCED, "canonical_intermediate_graduated.zip", "advanced"),
])
def test_valid_matching_round_record_resumes_correctly(tmp_path, module, parent_name, stage):
    declared_parent = _write(tmp_path / parent_name, b"declared parent checkpoint bytes")
    real_checkpoint = _write(tmp_path / f"canonical_{stage}_ppo_010k.zip", b"round 1's real checkpoint bytes")
    summary_path = current_generation_path(tmp_path / f"canonical_{stage}_run_summary.json")
    record = with_round_identity(
        {"round": 1, "round_passed_absolute_bar": True, "consecutive_passes": 1, "carried_forward_checkpoint": str(real_checkpoint.resolve())},
        stage=stage, declared_parent_checkpoint=declared_parent, current_checkpoint=real_checkpoint,
    )
    summary_path.write_text(json.dumps([record]), encoding="utf-8")

    round_reports, consecutive_passes, current_checkpoint = module._resume_round_state(
        summary_path, declared_parent_checkpoint=declared_parent,
    )

    assert len(round_reports) == 1
    assert consecutive_passes == 1
    assert current_checkpoint == real_checkpoint


@pytest.mark.parametrize("module,parent_name,stage", [
    (BEGINNER, "canonical_basic_graduated.zip", "early"),
    (INTERMEDIATE, "canonical_beginner_graduated.zip", "intermediate"),
    (ADVANCED, "canonical_intermediate_graduated.zip", "advanced"),
])
def test_valid_summary_but_checkpoint_bytes_changed_is_not_resumed(tmp_path, module, parent_name, stage):
    """SAME path as a validly-recorded round's checkpoint, but the bytes
    changed since -- must not resume even though the summary itself looks
    otherwise valid."""
    declared_parent = _write(tmp_path / parent_name, b"declared parent checkpoint bytes")
    real_checkpoint = _write(tmp_path / f"canonical_{stage}_ppo_010k.zip", b"round 1's real checkpoint bytes")
    summary_path = current_generation_path(tmp_path / f"canonical_{stage}_run_summary.json")
    record = with_round_identity(
        {"round": 1, "round_passed_absolute_bar": True, "consecutive_passes": 1, "carried_forward_checkpoint": str(real_checkpoint.resolve())},
        stage=stage, declared_parent_checkpoint=declared_parent, current_checkpoint=real_checkpoint,
    )
    summary_path.write_text(json.dumps([record]), encoding="utf-8")
    _write(real_checkpoint, b"REPLACED -- different model entirely, same filename")

    round_reports, consecutive_passes, current_checkpoint = module._resume_round_state(
        summary_path, declared_parent_checkpoint=declared_parent,
    )

    assert round_reports == []
    assert consecutive_passes == 0
    assert current_checkpoint == declared_parent


# --- section 12: Beginner's heldout/unseen/challenge coherent cache set --


@pytest.fixture
def beginner_env(tmp_path, monkeypatch):
    """Points RUN_CANONICAL_BEGINNER's module-level EVAL_DIR at a temp dir
    and its manifest constants at real temp manifest files, so
    `_load_coherent_evaluation_set` (which reads these module constants
    internally) operates against controlled, isolated state."""
    eval_dir = tmp_path / "evaluations"
    eval_dir.mkdir()
    monkeypatch.setattr(BEGINNER, "EVAL_DIR", eval_dir)
    heldout_manifest = _write(tmp_path / "early_heldout.json", b'{"layouts": {"L1": {}}}')
    unseen_manifest = _write(tmp_path / "early_heldout_unseen.json", b'{"layouts": {"L2": {}}}')
    challenge_manifest = _write(tmp_path / "early_challenge.json", b'{"fixed_regression_scenarios": {}, "challenge_family_layouts": {}}')
    monkeypatch.setattr(BEGINNER, "EARLY_HELDOUT_MANIFEST", str(heldout_manifest))
    monkeypatch.setattr(BEGINNER, "EARLY_HELDOUT_UNSEEN_MANIFEST", str(unseen_manifest))
    monkeypatch.setattr(BEGINNER, "EARLY_CHALLENGE_MANIFEST", str(challenge_manifest))
    checkpoint = _write(tmp_path / "canonical_beginner_ppo_010k.zip", b"the checkpoint being evaluated")
    parent = _write(tmp_path / "canonical_basic_graduated.zip", b"declared parent bytes")
    monkeypatch.setattr(BEGINNER, "GRADUATED_BASIC_CHECKPOINT", parent)
    return {"eval_dir": eval_dir, "checkpoint": checkpoint, "label": "ppo_010k_pre_rehearsal"}


def _write_role_cache(module, env, role: str, *, manifest_attr: str, n_episodes: int) -> Path:
    from simulator.curriculum_resume_identity import evaluation_cache_identity, with_evaluation_cache_identity

    path = current_generation_path(env["eval_dir"] / f"canonical_{env['label']}_{role if role != 'unseen_templates' else 'unseen'}.json")
    manifest_path = getattr(module, manifest_attr)
    report = with_evaluation_cache_identity(
        {"n_episodes": n_episodes},
        stage="early", declared_parent_checkpoint=module.GRADUATED_BASIC_CHECKPOINT, evaluated_checkpoint=env["checkpoint"],
        evaluation_role=role, manifests={role: manifest_path}, config=module._HELDOUT_EVAL_CONFIG if role != "challenge" else module._CHALLENGE_EVAL_CONFIG,
    )
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_all_three_roles_valid_reuses_the_coherent_set(beginner_env):
    _write_role_cache(BEGINNER, beginner_env, "heldout", manifest_attr="EARLY_HELDOUT_MANIFEST", n_episodes=4)
    _write_role_cache(BEGINNER, beginner_env, "unseen_templates", manifest_attr="EARLY_HELDOUT_UNSEEN_MANIFEST", n_episodes=4)
    _write_role_cache(BEGINNER, beginner_env, "challenge", manifest_attr="EARLY_CHALLENGE_MANIFEST", n_episodes=4)

    result = BEGINNER._load_coherent_evaluation_set(beginner_env["checkpoint"], label=beginner_env["label"])

    assert result is not None
    heldout, unseen, challenge = result
    assert heldout["n_episodes"] == unseen["n_episodes"] == challenge["n_episodes"] == 4


def test_heldout_valid_unseen_stale_challenge_valid_is_not_treated_as_one_valid_evaluation(beginner_env):
    """The exact scenario task section 12 names: heldout valid, unseen
    stale, challenge valid must NOT result in a mixed "valid" evaluation
    set -- the whole set must be recomputed together."""
    _write_role_cache(BEGINNER, beginner_env, "heldout", manifest_attr="EARLY_HELDOUT_MANIFEST", n_episodes=4)
    stale_unseen_path = current_generation_path(beginner_env["eval_dir"] / f"canonical_{beginner_env['label']}_unseen.json")
    stale_unseen_path.write_text(json.dumps({"n_episodes": 999}), encoding="utf-8")  # no identity at all: stale/legacy shape
    _write_role_cache(BEGINNER, beginner_env, "challenge", manifest_attr="EARLY_CHALLENGE_MANIFEST", n_episodes=4)

    result = BEGINNER._load_coherent_evaluation_set(beginner_env["checkpoint"], label=beginner_env["label"])

    assert result is None, "one stale member must invalidate the whole cache set, never a partial mix"


def test_all_three_roles_missing_recomputes_together(beginner_env):
    result = BEGINNER._load_coherent_evaluation_set(beginner_env["checkpoint"], label=beginner_env["label"])
    assert result is None


def test_heldout_and_unseen_from_a_different_checkpoint_than_challenge_is_rejected(beginner_env):
    """A subtler mixed-checkpoint scenario: heldout+unseen were computed
    against checkpoint A, challenge against a DIFFERENT checkpoint B (same
    label by coincidence/bug) -- must still reject the set."""
    _write_role_cache(BEGINNER, beginner_env, "heldout", manifest_attr="EARLY_HELDOUT_MANIFEST", n_episodes=4)
    _write_role_cache(BEGINNER, beginner_env, "unseen_templates", manifest_attr="EARLY_HELDOUT_UNSEEN_MANIFEST", n_episodes=4)
    # Challenge cache written against a DIFFERENT checkpoint's identity.
    from simulator.curriculum_resume_identity import with_evaluation_cache_identity

    other_checkpoint = beginner_env["eval_dir"].parent / "some_other_checkpoint.zip"
    _write(other_checkpoint, b"a different checkpoint entirely")
    challenge_path = current_generation_path(beginner_env["eval_dir"] / f"canonical_{beginner_env['label']}_challenge.json")
    challenge_report = with_evaluation_cache_identity(
        {"n_episodes": 4}, stage="early", declared_parent_checkpoint=BEGINNER.GRADUATED_BASIC_CHECKPOINT,
        evaluated_checkpoint=other_checkpoint, evaluation_role="challenge",
        manifests={"challenge": BEGINNER.EARLY_CHALLENGE_MANIFEST}, config=BEGINNER._CHALLENGE_EVAL_CONFIG,
    )
    challenge_path.write_text(json.dumps(challenge_report), encoding="utf-8")

    result = BEGINNER._load_coherent_evaluation_set(beginner_env["checkpoint"], label=beginner_env["label"])
    assert result is None
