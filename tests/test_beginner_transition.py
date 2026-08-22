from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from simulator.basic_environment import collect_basic_dagger_dataset, save_basic_dagger_dataset
from simulator.basic_training import (
    build_fresh_basic_policy,
    build_human_bootstrap_dataset,
    bootstrap_policy_from_human_recordings,
)
from simulator.beginner_transition import graduate_basic_to_beginner, rehearse_beginner_on_basic_data
from simulator.demonstrations import export_demonstrations
from simulator.map_model import MapModel
from simulator.navigation_dataset import MiningConfig
from simulator.navigation_history import NavigationHistoryWrapper
from simulator.navigation_ppo import balanced_training_vec_env_phase2
from simulator.navigation_subpolicy import FrozenNavigationSteering
from simulator.synthetic import generate_curriculum_from_plan, iter_variant_environments
from tests.test_simulator_core import _synthetic_recording


def _tiny_curriculum(tmp_path: Path) -> Path:
    return generate_curriculum_from_plan(
        tmp_path / "curriculum", [("early", "open_field", "typical", "fast", 0)], seed=555003, overwrite=True,
    )


def _basic_checkpoint(tmp_path: Path, curriculum_path: Path) -> Path:
    entry, base_env = next(iter(iter_variant_environments(
        str(curriculum_path), stage="early", seed=0, episode_steps=5, episode_seconds=3.0,
    )))
    env = NavigationHistoryWrapper(base_env)
    model = build_fresh_basic_policy(env, seed=0, device="cpu")
    env.close()
    checkpoint = tmp_path / "canonical_basic_bootstrap.zip"
    model.save(str(checkpoint))
    return checkpoint


def test_balanced_training_vec_env_phase2_never_wraps_recovery(tmp_path: Path) -> None:
    """The structural property the whole recovery/PPO design depends on:
    Beginner's PPO training env chain must never contain RecoveryController,
    regardless of future refactors -- assert it directly on the actual
    wrapper chain, not just by code inspection."""
    curriculum_path = _tiny_curriculum(tmp_path)
    vec_env, _names = balanced_training_vec_env_phase2(
        str(curriculum_path), stage="early", seed=0, episode_seconds=3.0, max_actions=5,
    )
    try:
        chain = []
        cursor = vec_env.envs[0]
        for _ in range(10):
            chain.append(type(cursor).__name__)
            cursor = getattr(cursor, "env", None)
            if cursor is None:
                break
        assert not any("Recovery" in name for name in chain), f"RecoveryController found in Beginner's PPO env chain: {chain}"
    finally:
        vec_env.close()


def test_graduate_basic_to_beginner_runs_and_updates_parameters(tmp_path: Path) -> None:
    curriculum_path = _tiny_curriculum(tmp_path)
    checkpoint = _basic_checkpoint(tmp_path, curriculum_path)
    before = PPO.load(str(checkpoint), device="cpu")
    before_value_weight = None
    for name, param in before.policy.named_parameters():
        if "value" in name and param.dim() == 2:
            before_value_weight = param.detach().clone()
            break

    output = tmp_path / "canonical_beginner_ppo_smoke.zip"
    result = graduate_basic_to_beginner(
        checkpoint, output, curriculum=str(curriculum_path), timesteps=32,
        stage="early", seed=0, episode_seconds=3.0, max_actions=20,
    )
    assert Path(result["checkpoint_out"]).exists()

    after = PPO.load(str(output), device="cpu")
    for name, param in after.policy.named_parameters():
        assert not torch.isnan(param).any(), f"NaN in {name} after Beginner PPO chunk"

    manifest_path = output.with_suffix("").with_suffix(".provenance.json")
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["fresh_initialization"] is False
    assert manifest["starting_checkpoint"] == str(checkpoint.resolve())
    assert manifest["recovery_config"]["enabled"] is False


def test_zero_shot_raw_diagnostic_parallel_matches_sequential(tmp_path: Path) -> None:
    from simulator.beginner_transition import zero_shot_raw_diagnostic, zero_shot_raw_diagnostic_parallel
    from simulator.curriculum_manifests import HeldoutManifest, save_manifest

    curriculum_path = _tiny_curriculum(tmp_path)
    checkpoint = _basic_checkpoint(tmp_path, curriculum_path)
    manifest_path = save_manifest(
        HeldoutManifest(stage="early", curriculum_path=str(curriculum_path),
                         layouts=("01_early_open_field_typical_fast",)),
        tmp_path / "heldout.json",
    )

    sequential = zero_shot_raw_diagnostic(
        checkpoint, heldout_manifest_path=str(manifest_path), seeds=[0, 1], episode_seconds=8.0, max_actions=40,
    )
    parallel = zero_shot_raw_diagnostic_parallel(
        checkpoint, heldout_manifest_path=str(manifest_path), seeds=[0, 1], episode_seconds=8.0, max_actions=40,
        n_workers=2,
    )
    assert sequential["per_layout"] == parallel["per_layout"]


def test_rehearse_beginner_on_basic_data_combines_human_and_dagger_data(tmp_path: Path) -> None:
    curriculum_path = _tiny_curriculum(tmp_path)
    checkpoint = _basic_checkpoint(tmp_path, curriculum_path)

    # human bootstrap dataset
    map_data = MapModel.load()
    session_dir = tmp_path / "rec"
    session_dir.mkdir()
    recording = _synthetic_recording(session_dir, map_data)
    demo_path = export_demonstrations([recording], tmp_path / "demos.npz", map_model=map_data)
    bootstrap_path = build_human_bootstrap_dataset(demo_path, tmp_path / "bootstrap.npz")

    # basic DAgger dataset
    model = PPO.load(str(checkpoint), device="cpu")
    config = MiningConfig(max_events_per_layout_seed=4, max_events_per_episode=2, max_samples_per_event=1)
    mined = collect_basic_dagger_dataset(
        str(curriculum_path), ["01_early_open_field_typical_fast"], seeds=[0], model=model,
        navigation_steering=FrozenNavigationSteering.load_frozen(device="cpu"),
        episode_seconds=8.0, max_actions=40, config=config,
    )
    dagger_path = save_basic_dagger_dataset(mined, str(tmp_path / "dagger.npz"))

    output = tmp_path / "canonical_beginner_rehearsed_smoke.zip"
    result = rehearse_beginner_on_basic_data(
        checkpoint, output, basic_dataset_paths=[bootstrap_path, dagger_path], epochs=1, batch_size=2, seed=0,
    )
    assert output.exists()
    reloaded = PPO.load(str(output), device="cpu")
    for name, param in reloaded.policy.named_parameters():
        assert not torch.isnan(param).any(), f"NaN in {name} after rehearsal"
    assert result["train_samples"] > 0

    manifest_path = output.with_suffix("").with_suffix(".provenance.json")
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["milestone"] == "rehearsal"
    assert manifest["starting_checkpoint"] == str(checkpoint.resolve())


def test_evaluate_heldout_925_matches_parallel(tmp_path: Path) -> None:
    """evaluate_heldout (milestone_evaluator's original) drives the raw
    923-value env unwrapped -- the wrong input width for a
    SplitSteeringNavigationPolicy checkpoint (confirmed: milestone_
    evaluator.py never imports NavigationHistoryWrapper and its own tests
    only exercise it against the older 923-input SplitSteeringEventPolicy).
    evaluate_heldout_925 is the fix; this locks in that its sequential and
    parallel forms agree exactly, the same equivalence bar every other
    parallel/sequential pair in this codebase is held to."""
    from simulator.beginner_transition import evaluate_heldout_925, evaluate_heldout_925_parallel
    from simulator.curriculum_manifests import HeldoutManifest

    curriculum_path = _tiny_curriculum(tmp_path)
    checkpoint = _basic_checkpoint(tmp_path, curriculum_path)
    model = PPO.load(str(checkpoint), device="cpu")
    manifest = HeldoutManifest(stage="early", curriculum_path=str(curriculum_path), layouts=("01_early_open_field_typical_fast",))

    sequential = evaluate_heldout_925(model, manifest, seeds=[0, 1], episode_seconds=8.0, max_actions=40)
    parallel = evaluate_heldout_925_parallel(checkpoint, manifest, seeds=[0, 1], episode_seconds=8.0, max_actions=40, n_workers=2)

    assert sequential["role"] == "heldout"
    layout = "01_early_open_field_typical_fast"
    assert sequential["layouts"][layout]["n_episodes"] == 2
    assert sequential["layouts"][layout] == parallel["layouts"][layout]


def test_evaluate_challenge_925_matches_parallel(tmp_path: Path) -> None:
    from simulator.beginner_transition import evaluate_challenge_925, evaluate_challenge_925_parallel
    from simulator.curriculum_manifests import ChallengeManifest, FixedRegressionScenario

    curriculum_path = _tiny_curriculum(tmp_path)
    checkpoint = _basic_checkpoint(tmp_path, curriculum_path)
    model = PPO.load(str(checkpoint), device="cpu")
    manifest = ChallengeManifest(
        stage="early",
        fixed_regression_scenarios=(
            FixedRegressionScenario(
                id="probe_case", curriculum_path=str(curriculum_path), layout="01_early_open_field_typical_fast",
                seed=0, episode_seconds=8.0, max_actions=40, expected_failure_signature="none yet -- test probe",
                discovered="2026-08-08",
            ),
        ),
        challenge_family_curriculum_path=str(curriculum_path),
        challenge_family_layouts=("01_early_open_field_typical_fast",),
    )

    sequential = evaluate_challenge_925(model, manifest, family_seeds=[0, 1], episode_seconds=8.0, max_actions=40)
    parallel = evaluate_challenge_925_parallel(checkpoint, manifest, family_seeds=[0, 1], episode_seconds=8.0, max_actions=40, n_workers=2)

    assert sequential["role"] == "challenge"
    assert "probe_case" in sequential["fixed_regression_scenarios"]
    assert sequential["fixed_regression_scenarios"]["probe_case"] == parallel["fixed_regression_scenarios"]["probe_case"]
    layout = "01_early_open_field_typical_fast"
    assert sequential["challenge_family"][layout] == parallel["challenge_family"][layout]
