from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.utils import get_schedule_fn

from navigation.navigation_evidence import POLICY_INPUT_SIZE, RAW_OBSERVATION_SIZE, SIDECAR_SIZE
from simulator.factorized_v193_training import fine_tune_steering_branch_v193
from simulator.split_branch_policy import SplitSteeringNavigationPolicy
from simulator.synthetic import iter_variant_environments


class _FakeModel:
    def __init__(self, policy) -> None:
        self.policy = policy


def _build_policy() -> SplitSteeringNavigationPolicy:
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(POLICY_INPUT_SIZE,), dtype=np.float32)
    act_space = spaces.MultiDiscrete([3, 3])
    return SplitSteeringNavigationPolicy(
        obs_space, act_space, get_schedule_fn(3e-4),
        steering_net_arch=[16, 8], event_net_arch=[32, 16], vf_net_arch=[32, 16],
    )


def _real_raw_observations(n: int) -> np.ndarray:
    entry, env = next(iter(iter_variant_environments(
        "synthetic_curriculum/curriculum.json", stage="early", seed=0, episode_steps=n + 5, episode_seconds=30.0,
    )))
    observation, _ = env.reset(seed=0)
    observations = [np.asarray(observation, dtype=np.float32)]
    for _ in range(n - 1):
        observation, _r, term, trunc, _info = env.step(np.asarray([0, 0], dtype=np.int64))
        observations.append(np.asarray(observation, dtype=np.float32))
        if term or trunc:
            break
    env.close()
    return np.stack(observations[:n])


def _build_synthetic_dataset(path: Path, n_per_layout: int = 60) -> None:
    rng = np.random.default_rng(0)
    raw = _real_raw_observations(n_per_layout * 2)
    temporal = rng.uniform(0.0, 1.0, size=(raw.shape[0], 2)).astype(np.float32)
    # A valid previous-steering one-hot per sample (not just uniform noise --
    # this feeds the same steering_net columns a real deployment would see).
    previous_steering_index = rng.integers(0, 3, size=raw.shape[0])
    one_hot = np.zeros((raw.shape[0], 3), dtype=np.float32)
    one_hot[np.arange(raw.shape[0]), previous_steering_index] = 1.0
    sidecar = np.concatenate([temporal, one_hot], axis=1)
    assert sidecar.shape[1] == SIDECAR_SIZE
    observations = np.concatenate([raw, sidecar], axis=1)
    # A deliberately learnable pattern: steering label depends on sidecar[0]
    # (recent_progress), with a STRAIGHT band in the middle so all three
    # steering values and both event values are represented (required by
    # _layout_stratified_episode_split's coverage check).
    steering_labels = np.where(
        sidecar[:, 0] > 0.66, 1, np.where(sidecar[:, 0] < 0.33, 2, 0)
    ).astype(np.int64)
    event_labels = np.where(sidecar[:, 1] > 0.9, 1, 0).astype(np.int64)  # rare CAST_EVA
    actions = np.stack([steering_labels, event_labels], axis=1)
    layout_index = np.concatenate([np.zeros(n_per_layout, dtype=np.int64), np.ones(n_per_layout, dtype=np.int64)])
    episode_index = np.concatenate([
        np.repeat(np.arange(6), n_per_layout // 6 + 1)[:n_per_layout],
        np.repeat(np.arange(6, 12), n_per_layout // 6 + 1)[:n_per_layout],
    ])
    np.savez_compressed(
        path, observations=observations, actions=actions,
        layout_index=layout_index, episode_index=episode_index,
    )


def test_fine_tune_updates_steering_but_not_event_or_value(tmp_path: Path):
    dataset_path = tmp_path / "dataset.npz"
    _build_synthetic_dataset(dataset_path)

    policy = _build_policy()
    model = _FakeModel(policy)

    # Snapshot event_net/vf_net/value_net weights before.
    before_event = {n: p.detach().clone() for n, p in policy.named_parameters() if "event" in n or "vf_net" in n}
    before_value = {n: p.detach().clone() for n, p in policy.named_parameters() if n.startswith("value_net")}

    result = fine_tune_steering_branch_v193(
        model, dataset_path, epochs=10, learning_rate=1e-2, batch_size=32, validation_fraction=0.3, seed=0,
    )

    assert result["train_samples"] > 0
    assert result["validation_samples"] > 0
    assert result["event_head_unaffected"] is True

    for name, param in policy.named_parameters():
        if name in before_event:
            torch.testing.assert_close(param.detach(), before_event[name])
        if name in before_value:
            torch.testing.assert_close(param.detach(), before_value[name])

    # Steering accuracy on the learnable synthetic pattern should improve materially.
    assert result["after"]["steering"]["samples"] > 0


def test_all_params_trainable_after_fine_tune(tmp_path: Path):
    dataset_path = tmp_path / "dataset.npz"
    _build_synthetic_dataset(dataset_path)
    policy = _build_policy()
    model = _FakeModel(policy)
    fine_tune_steering_branch_v193(model, dataset_path, epochs=1, batch_size=32, validation_fraction=0.3, seed=0)
    assert all(p.requires_grad for p in policy.parameters())


def test_raises_for_wrong_policy_type(tmp_path: Path):
    import pytest

    from simulator.split_branch_policy import SplitSteeringEventPolicy

    dataset_path = tmp_path / "dataset.npz"
    _build_synthetic_dataset(dataset_path)

    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(923,), dtype=np.float32)
    act_space = spaces.MultiDiscrete([3, 3])
    old_policy = SplitSteeringEventPolicy(obs_space, act_space, get_schedule_fn(3e-4))
    model = _FakeModel(old_policy)
    with pytest.raises(ValueError, match="SplitSteeringNavigationPolicy"):
        fine_tune_steering_branch_v193(model, dataset_path, epochs=1)


def test_raises_for_wrong_observation_width(tmp_path: Path):
    import pytest

    dataset_path = tmp_path / "dataset.npz"
    _build_synthetic_dataset(dataset_path)
    with np.load(dataset_path) as data:
        d = dict(data)
    d["observations"] = d["observations"][:, :-1]  # wrong width
    np.savez_compressed(dataset_path, **d)

    policy = _build_policy()
    model = _FakeModel(policy)
    with pytest.raises(ValueError, match="raw\\+sidecar"):
        fine_tune_steering_branch_v193(model, dataset_path, epochs=1)
