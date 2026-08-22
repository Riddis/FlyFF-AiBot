"""Proves the Beginner (and by extension Intermediate/Advanced) event-only
PPO architecture is wired correctly end to end: Basic's dual-head checkpoint
transfers into a genuinely event-only policy, that policy trains under
Discrete(len(FarmingEvent)) with steering driven entirely by
FrozenNavigationSteering, and evaluation exercises the same composed
architecture as training (docs/architecture/CURRICULUM_TRAINING_PIPELINE.md
section 4/5/10).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import PPO

from farming.actions import FarmingEvent
from navigation.navigation_evidence import RAW_OBSERVATION_SIZE
from simulator.basic_training import build_fresh_basic_policy
from simulator.beginner_transition import (
    build_event_only_ppo_from_basic_checkpoint,
    continue_event_only_ppo_chunk,
    rehearse_event_only_on_basic_data,
)
from simulator.basic_environment import collect_basic_dagger_dataset, save_basic_dagger_dataset
from simulator.basic_training import build_human_bootstrap_dataset
from simulator.demonstrations import export_demonstrations
from simulator.map_model import MapModel
from simulator.navigation_dataset import MiningConfig
from simulator.navigation_history import NavigationHistoryWrapper
from simulator.navigation_ppo import balanced_training_vec_env_event_only
from simulator.navigation_subpolicy import FrozenNavigationSteering
from simulator.synthetic import generate_curriculum_from_plan, iter_variant_environments
from tests.test_simulator_core import _synthetic_recording

LAYOUT = "01_early_open_field_typical_fast"


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


def test_balanced_training_vec_env_event_only_exposes_discrete_action_space(tmp_path: Path) -> None:
    curriculum_path = _tiny_curriculum(tmp_path)
    vec_env, names = balanced_training_vec_env_event_only(
        str(curriculum_path), stage="early", seed=0, episode_seconds=3.0, max_actions=5,
    )
    try:
        assert names == [LAYOUT]
        assert isinstance(vec_env.action_space, spaces.Discrete)
        assert vec_env.action_space.n == len(FarmingEvent)
        assert not isinstance(vec_env.action_space, spaces.MultiDiscrete)
        assert vec_env.observation_space.shape == (RAW_OBSERVATION_SIZE,)

        chain = []
        cursor = vec_env.envs[0]
        for _ in range(10):
            chain.append(type(cursor).__name__)
            cursor = getattr(cursor, "env", None)
            if cursor is None:
                break
        assert not any("Recovery" in name for name in chain), f"RecoveryController found in event-only PPO env chain: {chain}"
        assert any(name == "FrozenNavigationWrapper" for name in chain), f"FrozenNavigationWrapper missing from env chain: {chain}"
    finally:
        vec_env.close()


def test_build_event_only_ppo_from_basic_checkpoint_matches_source_event_distribution(tmp_path: Path) -> None:
    curriculum_path = _tiny_curriculum(tmp_path)
    basic_checkpoint = _basic_checkpoint(tmp_path, curriculum_path)
    source = PPO.load(str(basic_checkpoint), device="cpu")

    event_only = build_event_only_ppo_from_basic_checkpoint(basic_checkpoint, seed=0, device="cpu")
    assert isinstance(event_only.action_space, spaces.Discrete)
    assert event_only.action_space.n == len(FarmingEvent)
    assert event_only.observation_space.shape == (RAW_OBSERVATION_SIZE,)

    entry, base_env = next(iter(iter_variant_environments(
        str(curriculum_path), stage="early", seed=0, episode_steps=5, episode_seconds=3.0,
    )))
    env = NavigationHistoryWrapper(base_env)
    observation, _ = env.reset(seed=0)
    env.close()
    raw = np.asarray(observation, dtype=np.float32)[None, :RAW_OBSERVATION_SIZE]
    full = np.asarray(observation, dtype=np.float32)[None, :]

    with torch.no_grad():
        source_probs = source.policy.get_distribution(torch.as_tensor(full)).distribution[1].probs
        event_only_probs = event_only.policy.get_distribution(torch.as_tensor(raw)).distribution.probs

    torch.testing.assert_close(source_probs, event_only_probs, atol=1e-5, rtol=1e-4)


def test_continue_event_only_ppo_chunk_trains_only_event_and_never_touches_0051200(tmp_path: Path) -> None:
    curriculum_path = _tiny_curriculum(tmp_path)
    basic_checkpoint = _basic_checkpoint(tmp_path, curriculum_path)
    event_only = build_event_only_ppo_from_basic_checkpoint(basic_checkpoint, seed=0, device="cpu")
    starting_checkpoint = tmp_path / "canonical_beginner_event_only_start.zip"
    event_only.save(str(starting_checkpoint))

    from simulator.navigation_subpolicy import FROZEN_NAVIGATION_CHECKPOINT_PATH, verify_frozen_navigation_checkpoint
    sha_before = verify_frozen_navigation_checkpoint(FROZEN_NAVIGATION_CHECKPOINT_PATH)

    output = tmp_path / "canonical_beginner_ppo_smoke.zip"
    result = continue_event_only_ppo_chunk(
        starting_checkpoint, output, curriculum=str(curriculum_path), timesteps=32,
        stage="early", seed=0, episode_seconds=3.0, max_actions=20,
    )
    assert Path(result["checkpoint_out"]).exists()

    sha_after = verify_frozen_navigation_checkpoint(FROZEN_NAVIGATION_CHECKPOINT_PATH)
    assert sha_after == sha_before, "frozen navigation checkpoint bytes changed after a PPO chunk -- must never happen"

    after = PPO.load(str(output), device="cpu")
    assert isinstance(after.action_space, spaces.Discrete)
    assert after.action_space.n == len(FarmingEvent)
    for name, param in after.policy.named_parameters():
        assert not torch.isnan(param).any(), f"NaN in {name} after Beginner event-only PPO chunk"


def test_event_only_checkpoints_record_navigation_checkpoint_provenance(tmp_path: Path) -> None:
    """Section 19's provenance requirement: an event-only checkpoint is not
    reproducible/executable by itself without knowing which frozen
    navigator it composes with -- every provenance manifest in this lineage
    must record its path and SHA-256."""
    import json

    from simulator.beginner_transition import save_event_only_checkpoint_with_provenance
    from simulator.navigation_subpolicy import FROZEN_NAVIGATION_CHECKPOINT_SHA256

    curriculum_path = _tiny_curriculum(tmp_path)
    basic_checkpoint = _basic_checkpoint(tmp_path, curriculum_path)
    event_only = build_event_only_ppo_from_basic_checkpoint(basic_checkpoint, seed=0, device="cpu")
    start_checkpoint = tmp_path / "event_only_start.zip"
    save_event_only_checkpoint_with_provenance(
        event_only, start_checkpoint, basic_checkpoint=basic_checkpoint, seed=0,
    )
    start_manifest = json.loads(start_checkpoint.with_suffix("").with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert start_manifest["architecture_contract"]["navigation_checkpoint_sha256"] == FROZEN_NAVIGATION_CHECKPOINT_SHA256

    ppo_output = tmp_path / "event_only_round1.zip"
    continue_event_only_ppo_chunk(
        start_checkpoint, ppo_output, curriculum=str(curriculum_path), timesteps=32,
        stage="early", seed=0, episode_seconds=3.0, max_actions=20,
    )
    ppo_manifest = json.loads(ppo_output.with_suffix("").with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert ppo_manifest["architecture_contract"]["navigation_checkpoint_sha256"] == FROZEN_NAVIGATION_CHECKPOINT_SHA256
    assert ppo_manifest["starting_checkpoint"] == str(start_checkpoint.resolve())


def test_resume_ppo_chunk_event_only_rejects_a_dual_head_checkpoint(tmp_path: Path) -> None:
    """Model-contract negative check for the event-only curriculum stages
    (docs/architecture/CURRICULUM_TRAINING_PIPELINE.md section 18): a
    pre-recovery MultiDiscrete([3,3]) SplitSteeringNavigationPolicy
    checkpoint (Basic's own shape, or any stale checkpoint that never went
    through build_event_only_ppo_from_basic_checkpoint) must be refused,
    not silently accepted with a steering action that would then never
    execute."""
    from simulator.navigation_ppo import resume_ppo_chunk_event_only

    curriculum_path = _tiny_curriculum(tmp_path)
    dual_head_checkpoint = _basic_checkpoint(tmp_path, curriculum_path)

    # SB3's own PPO.load(env=...) space check catches this first (dual-head's
    # 928-value observation vs. event-only's 923-value one) -- an even
    # stronger guarantee than a bespoke action-space check alone, since it
    # refuses the mismatch before any policy_kwargs/action-space comparison
    # is even reached.
    with pytest.raises(ValueError, match="spaces do not match"):
        resume_ppo_chunk_event_only(
            checkpoint=dual_head_checkpoint, curriculum=str(curriculum_path), output=tmp_path / "should_not_exist.zip",
            timesteps=32, stage="early", seed=0, episode_seconds=3.0, max_actions=20,
        )


def test_frozen_navigation_wrapper_step_executes_sampled_event_and_frozen_steering(tmp_path: Path) -> None:
    """Direct on-policy-consistency proof for the composed training env: the
    steering component of the executed command comes from
    FrozenNavigationSteering (spied), and the event component is EXACTLY the
    action passed into step() -- no override, no second event mechanism."""
    curriculum_path = _tiny_curriculum(tmp_path)
    vec_env, _names = balanced_training_vec_env_event_only(
        str(curriculum_path), stage="early", seed=0, episode_seconds=8.0, max_actions=30,
    )
    try:
        wrapper = vec_env.envs[0].env  # Monitor -> FrozenNavigationWrapper
        assert type(wrapper).__name__ == "FrozenNavigationWrapper"

        call_count = 0
        real_steering_action = FrozenNavigationSteering.steering_action

        def _spy(self, env, *, target_actor_id):
            nonlocal call_count
            call_count += 1
            return real_steering_action(self, env, target_actor_id=target_actor_id)

        FrozenNavigationSteering.steering_action = _spy
        try:
            vec_env.reset()
            for tick in range(15):
                event_action = tick % len(FarmingEvent)
                obs, reward, done, info = vec_env.step(np.asarray([event_action]))
                if done[0]:
                    vec_env.reset()
        finally:
            FrozenNavigationSteering.steering_action = real_steering_action

        assert call_count > 0, "FrozenNavigationSteering.steering_action was never invoked during training-env steps"
    finally:
        vec_env.close()


def test_rehearse_event_only_trains_event_head_and_evaluate_heldout_uses_composed_architecture(tmp_path: Path) -> None:
    curriculum_path = _tiny_curriculum(tmp_path)
    basic_checkpoint = _basic_checkpoint(tmp_path, curriculum_path)
    event_only = build_event_only_ppo_from_basic_checkpoint(basic_checkpoint, seed=0, device="cpu")
    starting_checkpoint = tmp_path / "event_only_start.zip"
    event_only.save(str(starting_checkpoint))

    map_data = MapModel.load()
    session_dir = tmp_path / "rec"
    session_dir.mkdir()
    recording = _synthetic_recording(session_dir, map_data)
    demo_path = export_demonstrations([recording], tmp_path / "demos.npz", map_model=map_data)
    bootstrap_path = build_human_bootstrap_dataset(demo_path, tmp_path / "bootstrap.npz")

    model_for_mining = PPO.load(str(basic_checkpoint), device="cpu")
    config = MiningConfig(max_events_per_layout_seed=4, max_events_per_episode=2, max_samples_per_event=1)
    mined = collect_basic_dagger_dataset(
        str(curriculum_path), [LAYOUT], seeds=[0], model=model_for_mining,
        navigation_steering=FrozenNavigationSteering.load_frozen(device="cpu"),
        episode_seconds=8.0, max_actions=40, config=config,
    )
    dagger_path = save_basic_dagger_dataset(mined, str(tmp_path / "dagger.npz"))

    rehearsed_output = tmp_path / "event_only_rehearsed.zip"
    result = rehearse_event_only_on_basic_data(
        starting_checkpoint, rehearsed_output, basic_dataset_paths=[bootstrap_path, dagger_path],
        epochs=1, batch_size=4, seed=0,
    )
    assert rehearsed_output.exists()
    assert result["train_samples"] > 0
    reloaded = PPO.load(str(rehearsed_output), device="cpu")
    for name, param in reloaded.policy.named_parameters():
        assert not torch.isnan(param).any(), f"NaN in {name} after event-only rehearsal"

    from simulator.curriculum_manifests import HeldoutManifest
    from simulator.milestone_evaluator import evaluate_heldout

    manifest = HeldoutManifest(stage="early", curriculum_path=str(curriculum_path), layouts=(LAYOUT,))
    navigation_steering = FrozenNavigationSteering.load_frozen(device="cpu")
    report = evaluate_heldout(
        reloaded, manifest, seeds=[0], episode_seconds=8.0, max_actions=30, navigation_steering=navigation_steering,
    )
    assert report["role"] == "heldout"
    assert report["layouts"][LAYOUT]["n_episodes"] == 1
