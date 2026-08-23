from __future__ import annotations

import json
from pathlib import Path

import torch
from stable_baselines3 import PPO

from simulator.basic_training import build_fresh_basic_policy
from simulator.beginner_transition import (
    continue_farming_policy_ppo_chunk,
    rehearse_farming_policy_on_basic_data,
    zero_shot_raw_diagnostic,
    zero_shot_raw_diagnostic_parallel,
)
from simulator.basic_environment import collect_basic_dagger_dataset, save_basic_dagger_dataset
from simulator.basic_training import build_human_bootstrap_dataset
from simulator.demonstrations import export_demonstrations
from simulator.farming_target_policy import TARGET_ACTION_SIZE
from simulator.map_model import MapModel
from simulator.navigation_dataset import MiningConfig
from simulator.navigation_ppo import balanced_training_vec_env_farming_policy
from simulator.navigation_subpolicy import FrozenNavigationSteering
from simulator.synthetic import generate_curriculum_from_plan, iter_variant_environments
from tests.test_simulator_core import _synthetic_recording

LAYOUT = "01_early_open_field_typical_fast"


def _tiny_curriculum(tmp_path: Path) -> Path:
    return generate_curriculum_from_plan(
        tmp_path / "curriculum", [("early", "open_field", "typical", "fast", 0)], seed=555003, overwrite=True,
    )


def _basic_checkpoint(tmp_path: Path, curriculum_path: Path) -> Path:
    model = build_fresh_basic_policy(seed=0, device="cpu")
    checkpoint = tmp_path / "canonical_basic_bootstrap.zip"
    model.save(str(checkpoint))
    return checkpoint


def test_balanced_training_vec_env_farming_policy_never_wraps_recovery(tmp_path: Path) -> None:
    """The structural property the whole recovery/PPO design depends on:
    Beginner+'s PPO training env chain must never contain RecoveryController,
    regardless of future refactors -- assert it directly on the actual
    wrapper chain, not just by code inspection."""
    curriculum_path = _tiny_curriculum(tmp_path)
    vec_env, _names = balanced_training_vec_env_farming_policy(
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
        assert not any("Recovery" in name for name in chain), f"RecoveryController found in Beginner+'s PPO env chain: {chain}"
        assert any(name == "FarmingPolicyWrapper" for name in chain), f"FarmingPolicyWrapper missing from env chain: {chain}"
    finally:
        vec_env.close()


def test_continue_farming_policy_ppo_chunk_continues_basics_own_checkpoint_directly(tmp_path: Path) -> None:
    """No bridge/transfer needed: Beginner (and Intermediate/Advanced)
    continue Basic's own graduated checkpoint via a plain PPO chunk, since
    all four stages share one SplitFarmingTargetEventPolicy architecture."""
    curriculum_path = _tiny_curriculum(tmp_path)
    checkpoint = _basic_checkpoint(tmp_path, curriculum_path)
    before = PPO.load(str(checkpoint), device="cpu")
    before_value_weight = None
    for name, param in before.policy.named_parameters():
        if "value" in name and param.dim() == 2:
            before_value_weight = param.detach().clone()
            break

    output = tmp_path / "canonical_beginner_ppo_smoke.zip"
    result = continue_farming_policy_ppo_chunk(
        checkpoint, output, curriculum=str(curriculum_path), timesteps=32,
        stage="early", seed=0, episode_seconds=3.0, max_actions=20,
    )
    assert Path(result["checkpoint_out"]).exists()

    after = PPO.load(str(output), device="cpu")
    assert after.action_space == before.action_space
    for name, param in after.policy.named_parameters():
        assert not torch.isnan(param).any(), f"NaN in {name} after Beginner PPO chunk"

    manifest_path = output.with_suffix("").with_suffix(".provenance.json")
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["fresh_initialization"] is False
    assert manifest["starting_checkpoint"] == str(checkpoint.resolve())
    assert manifest["recovery_config"]["enabled"] is False
    assert manifest["architecture_contract"]["navigation_checkpoint_sha256"]
    assert manifest["architecture_contract"]["policy_class"] == "SplitFarmingTargetEventPolicy"
    assert manifest["architecture_contract"]["raw_observation_size"] == 923
    # Regression guard: farming_policy_architecture_contract() must override
    # EVERY architecture-describing key build_run_manifest's own default
    # sets, not just some of them -- a prior bug left policy_input_size=928
    # (the retired SplitSteeringNavigationPolicy's sidecar width) behind
    # after the merge even though raw_observation_size was correctly
    # overridden to 923 in the same manifest (see MISTAKES.md 2026-08-23).
    assert manifest["architecture_contract"]["policy_input_size"] == 923


def test_zero_shot_raw_diagnostic_parallel_matches_sequential(tmp_path: Path) -> None:
    from simulator.curriculum_manifests import HeldoutManifest, save_manifest

    curriculum_path = _tiny_curriculum(tmp_path)
    checkpoint = _basic_checkpoint(tmp_path, curriculum_path)
    manifest_path = save_manifest(
        HeldoutManifest(stage="early", curriculum_path=str(curriculum_path), layouts=(LAYOUT,)),
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


def test_rehearse_farming_policy_on_basic_data_combines_human_and_dagger_data(tmp_path: Path) -> None:
    curriculum_path = _tiny_curriculum(tmp_path)
    checkpoint = _basic_checkpoint(tmp_path, curriculum_path)

    map_data = MapModel.load()
    session_dir = tmp_path / "rec"
    session_dir.mkdir()
    recording = _synthetic_recording(session_dir, map_data)
    demo_path = export_demonstrations([recording], tmp_path / "demos.npz", map_model=map_data)
    bootstrap_path = build_human_bootstrap_dataset(demo_path, tmp_path / "bootstrap.npz")

    model = PPO.load(str(checkpoint), device="cpu")
    config = MiningConfig(max_events_per_layout_seed=4, max_events_per_episode=2, max_samples_per_event=1)
    mined = collect_basic_dagger_dataset(
        str(curriculum_path), [LAYOUT], seeds=[0, 1, 2, 3, 4], model=model,
        navigation_steering=FrozenNavigationSteering.load_frozen(device="cpu"),
        episode_seconds=8.0, max_actions=40, config=config,
    )
    dagger_path = save_basic_dagger_dataset(mined, str(tmp_path / "dagger.npz"))

    output = tmp_path / "canonical_beginner_rehearsed_smoke.zip"
    result = rehearse_farming_policy_on_basic_data(
        checkpoint, output, basic_dataset_paths=[bootstrap_path, dagger_path], max_epochs=2, batch_size=4, seed=0,
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


def test_farming_policy_action_space_is_multidiscrete_target_event() -> None:
    from gymnasium import spaces
    from farming.actions import FarmingEvent

    model = build_fresh_basic_policy(seed=0, device="cpu")
    assert isinstance(model.action_space, spaces.MultiDiscrete)
    assert list(model.action_space.nvec) == [TARGET_ACTION_SIZE, len(FarmingEvent)]
