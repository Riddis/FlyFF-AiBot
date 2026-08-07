from __future__ import annotations

from pathlib import Path

import numpy as np

from simulator.basic_environment import collect_basic_dagger_dataset, save_basic_dagger_dataset
from simulator.basic_training import build_fresh_basic_policy
from simulator.navigation_dataset import CATEGORY_PRECEDENCE, MiningConfig
from simulator.navigation_history import NavigationHistoryWrapper
from simulator.synthetic import generate_curriculum_from_plan, iter_variant_environments


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


def test_collect_basic_dagger_dataset_produces_valid_shapes_and_categories(tmp_path: Path) -> None:
    curriculum_path = _tiny_curriculum(tmp_path)
    model = _fresh_model(curriculum_path)
    config = MiningConfig(max_events_per_layout_seed=6, max_events_per_episode=3, max_samples_per_event=1)

    mined = collect_basic_dagger_dataset(
        str(curriculum_path), ["01_early_open_field_typical_fast", "02_early_irregular_plain_typical_fast"],
        seeds=[0, 1], model=model, episode_seconds=8.0, max_actions=40, config=config,
    )

    assert mined["observations"].shape[1] == 925
    assert mined["actions"].shape == (mined["observations"].shape[0], 2)
    assert set(mined["categories"]) <= set(CATEGORY_PRECEDENCE)
    assert len(mined["episode_summaries"]) == 4
    for summary in mined["episode_summaries"]:
        assert summary["intervention_count"] >= 0
        assert summary["final_state"] in ("normal", "recovering", "given_up")


def test_collect_basic_dagger_dataset_never_touches_a_ppo_buffer(tmp_path: Path) -> None:
    """Structural guard: collect_basic_dagger_dataset's return value must
    never contain PPO-buffer-shaped fields (log_prob, value, advantage,
    reward) -- if this ever starts looking like a rollout buffer, that is
    exactly the regression the recovery/PPO design forbids."""
    curriculum_path = _tiny_curriculum(tmp_path)
    model = _fresh_model(curriculum_path)
    config = MiningConfig(max_events_per_layout_seed=4, max_events_per_episode=2, max_samples_per_event=1)
    mined = collect_basic_dagger_dataset(
        str(curriculum_path), ["01_early_open_field_typical_fast"], seeds=[0], model=model,
        episode_seconds=8.0, max_actions=40, config=config,
    )
    forbidden_keys = {"log_prob", "value", "advantage", "reward", "return"}
    assert forbidden_keys.isdisjoint(mined.keys())


def test_save_basic_dagger_dataset_round_trips_and_marks_steering_fully_valid(tmp_path: Path) -> None:
    curriculum_path = _tiny_curriculum(tmp_path)
    model = _fresh_model(curriculum_path)
    config = MiningConfig(max_events_per_layout_seed=4, max_events_per_episode=2, max_samples_per_event=1)
    mined = collect_basic_dagger_dataset(
        str(curriculum_path), ["01_early_open_field_typical_fast"], seeds=[0], model=model,
        episode_seconds=8.0, max_actions=40, config=config,
    )
    path = save_basic_dagger_dataset(mined, str(tmp_path / "dagger.npz"))
    data = np.load(path, allow_pickle=False)
    assert data["observations"].shape == mined["observations"].shape
    assert np.all(data["steering_label_valid"])
    assert data["session_index"].shape[0] == mined["observations"].shape[0]
