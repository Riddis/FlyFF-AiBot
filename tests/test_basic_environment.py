from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from navigation.navigation_evidence import POLICY_INPUT_SIZE
from simulator.basic_environment import collect_basic_dagger_dataset, save_basic_dagger_dataset
from simulator.basic_training import build_fresh_basic_policy
from simulator.navigation_dataset import CATEGORY_PRECEDENCE, MiningConfig
from simulator.navigation_history import NavigationHistoryWrapper
from simulator.navigation_subpolicy import FrozenNavigationSteering
from simulator.synthetic import generate_curriculum_from_plan, iter_variant_environments


@pytest.fixture(scope="module")
def navigation_steering():
    return FrozenNavigationSteering.load_frozen(device="cpu")


def _tiny_curriculum(tmp_path: Path) -> Path:
    return generate_curriculum_from_plan(
        tmp_path / "curriculum",
        [("early", "open_field", "typical", "fast", 0), ("early", "irregular_plain", "typical", "fast", 0)],
        seed=555002, overwrite=True,
    )


def _fresh_model(curriculum_path: Path):
    entry, base_env = next(iter(iter_variant_environments(
        str(curriculum_path), stage="early", seed=0, episode_steps=5, episode_seconds=3.0,
    )))
    env = NavigationHistoryWrapper(base_env)
    model = build_fresh_basic_policy(env, seed=0, device="cpu")
    env.close()
    return model


def test_collect_basic_dagger_dataset_produces_valid_shapes_and_categories(tmp_path: Path, navigation_steering) -> None:
    curriculum_path = _tiny_curriculum(tmp_path)
    model = _fresh_model(curriculum_path)
    config = MiningConfig(max_events_per_layout_seed=6, max_events_per_episode=3, max_samples_per_event=1)

    mined = collect_basic_dagger_dataset(
        str(curriculum_path), ["01_early_open_field_typical_fast", "02_early_irregular_plain_typical_fast"],
        seeds=[0, 1], model=model, navigation_steering=navigation_steering,
        episode_seconds=8.0, max_actions=40, config=config,
    )

    assert mined["observations"].shape[1] == POLICY_INPUT_SIZE
    assert mined["actions"].shape == (mined["observations"].shape[0], 2)
    assert set(mined["categories"]) <= set(CATEGORY_PRECEDENCE)
    assert len(mined["episode_summaries"]) == 4
    for summary in mined["episode_summaries"]:
        assert summary["intervention_count"] >= 0
        assert summary["final_state"] in ("normal", "recovering", "given_up")


def test_collect_basic_dagger_dataset_never_touches_a_ppo_buffer(tmp_path: Path, navigation_steering) -> None:
    """Structural guard: collect_basic_dagger_dataset's return value must
    never contain PPO-buffer-shaped fields (log_prob, value, advantage,
    reward) -- if this ever starts looking like a rollout buffer, that is
    exactly the regression the recovery/PPO design forbids."""
    curriculum_path = _tiny_curriculum(tmp_path)
    model = _fresh_model(curriculum_path)
    config = MiningConfig(max_events_per_layout_seed=4, max_events_per_episode=2, max_samples_per_event=1)
    mined = collect_basic_dagger_dataset(
        str(curriculum_path), ["01_early_open_field_typical_fast"], seeds=[0], model=model,
        navigation_steering=navigation_steering, episode_seconds=8.0, max_actions=40, config=config,
    )
    forbidden_keys = {"log_prob", "value", "advantage", "reward", "return"}
    assert forbidden_keys.isdisjoint(mined.keys())


def test_collect_basic_dagger_dataset_parallel_matches_sequential(tmp_path: Path, navigation_steering) -> None:
    """The parallel (checkpoint_path + n_workers>1) path pre-rolls every
    episode out-of-process instead of skipping some via the sequential
    per-layout event cap short-circuit, then runs the identical
    mining/capping logic over the results -- must produce a byte-for-byte
    identical mined dataset to the sequential path, not just "similar"."""
    curriculum_path = _tiny_curriculum(tmp_path)
    model = _fresh_model(curriculum_path)
    checkpoint = tmp_path / "model.zip"
    model.save(str(checkpoint))
    config = MiningConfig(max_events_per_layout_seed=3, max_events_per_episode=2, max_samples_per_event=1)
    layouts = ["01_early_open_field_typical_fast", "02_early_irregular_plain_typical_fast"]

    from stable_baselines3 import PPO
    sequential_model = PPO.load(str(checkpoint), device="cpu")
    sequential = collect_basic_dagger_dataset(
        str(curriculum_path), layouts, seeds=[0, 1, 2], model=sequential_model,
        navigation_steering=navigation_steering, episode_seconds=8.0, max_actions=40, config=config,
    )
    parallel_model = PPO.load(str(checkpoint), device="cpu")
    parallel = collect_basic_dagger_dataset(
        str(curriculum_path), layouts, seeds=[0, 1, 2], model=parallel_model,
        navigation_steering=navigation_steering, episode_seconds=8.0, max_actions=40, config=config,
        checkpoint_path=str(checkpoint), n_workers=2,
    )

    np.testing.assert_array_equal(sequential["observations"], parallel["observations"])
    np.testing.assert_array_equal(sequential["actions"], parallel["actions"])
    assert sequential["categories"] == parallel["categories"]
    assert sequential["category_counts"] == parallel["category_counts"]
    assert len(sequential["episode_summaries"]) == len(parallel["episode_summaries"])


def test_save_basic_dagger_dataset_round_trips_and_marks_steering_fully_valid(tmp_path: Path, navigation_steering) -> None:
    curriculum_path = _tiny_curriculum(tmp_path)
    model = _fresh_model(curriculum_path)
    config = MiningConfig(max_events_per_layout_seed=4, max_events_per_episode=2, max_samples_per_event=1)
    mined = collect_basic_dagger_dataset(
        str(curriculum_path), ["01_early_open_field_typical_fast"], seeds=[0], model=model,
        navigation_steering=navigation_steering, episode_seconds=8.0, max_actions=40, config=config,
    )
    path = save_basic_dagger_dataset(mined, str(tmp_path / "dagger.npz"))
    data = np.load(path, allow_pickle=False)
    assert data["observations"].shape == mined["observations"].shape
    assert np.all(data["steering_label_valid"])
    assert data["session_index"].shape[0] == mined["observations"].shape[0]
