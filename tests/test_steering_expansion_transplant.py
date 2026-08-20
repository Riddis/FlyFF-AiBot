from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.utils import get_schedule_fn

from navigation.navigation_evidence import POLICY_INPUT_SIZE, RAW_OBSERVATION_SIZE, SIDECAR_SIZE
from simulator.factorized_v193_training import expand_steering_input_from_checkpoint
from simulator.navigation_history import NavigationHistoryWrapper
from simulator.split_branch_policy import SplitSteeringEventPolicy, SplitSteeringNavigationPolicy
from simulator.synthetic import iter_variant_environments


def _build_old_policy() -> SplitSteeringEventPolicy:
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(RAW_OBSERVATION_SIZE,), dtype=np.float32)
    act_space = spaces.MultiDiscrete([3, 3])
    return SplitSteeringEventPolicy(
        obs_space, act_space, get_schedule_fn(3e-4),
        steering_net_arch=[32, 16], event_net_arch=[64, 32], vf_net_arch=[64, 32],
    )


def _build_new_policy() -> SplitSteeringNavigationPolicy:
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(POLICY_INPUT_SIZE,), dtype=np.float32)
    act_space = spaces.MultiDiscrete([3, 3])
    return SplitSteeringNavigationPolicy(
        obs_space, act_space, get_schedule_fn(3e-4),
        steering_net_arch=[32, 16], event_net_arch=[64, 32], vf_net_arch=[64, 32],
    )


def _real_raw_observations(n: int) -> np.ndarray:
    entry, env = next(iter(iter_variant_environments(
        "curricula/synthetic_curriculum/curriculum.json", stage="early", seed=0, episode_steps=n + 5, episode_seconds=30.0,
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


def test_transplant_reproduces_old_policy_exactly_with_nonzero_new_features():
    torch.manual_seed(0)
    old_policy = _build_old_policy()
    new_policy = _build_new_policy()
    expand_steering_input_from_checkpoint(old_policy, new_policy)

    raw = _real_raw_observations(8)
    # Deliberately NONZERO, realistic-looking sidecar/navigation values --
    # an all-zero probe would pass even with broken (randomly-initialized)
    # new columns, since random_weight * 0 == 0 regardless of the weight.
    # [recent_progress, recent_contact, prev_straight, prev_left, prev_right]
    # -- a valid one-hot (prev_left), not all-zero, so this exercises the
    # 2026-08-13 previous-steering sidecar columns too.
    assert SIDECAR_SIZE == 5
    sidecar = np.tile(np.asarray([0.37, 0.4, 0.0, 1.0, 0.0], dtype=np.float32), (raw.shape[0], 1))
    augmented = np.concatenate([raw, sidecar], axis=1)

    old_obs = torch.as_tensor(raw, dtype=torch.float32)
    new_obs = torch.as_tensor(augmented, dtype=torch.float32)

    with torch.no_grad():
        old_dist = old_policy.get_distribution(old_obs).distribution
        new_dist = new_policy.get_distribution(new_obs).distribution
        old_value = old_policy.predict_values(old_obs)
        new_value = new_policy.predict_values(new_obs)

    old_steering_probs = old_dist[0].probs
    new_steering_probs = new_dist[0].probs
    old_event_probs = old_dist[1].probs
    new_event_probs = new_dist[1].probs

    torch.testing.assert_close(old_steering_probs, new_steering_probs, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(old_event_probs, new_event_probs, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(old_value, new_value, atol=1e-5, rtol=1e-4)


def test_new_columns_are_exactly_zero_after_transplant():
    old_policy = _build_old_policy()
    new_policy = _build_new_policy()
    expand_steering_input_from_checkpoint(old_policy, new_policy)

    old_in = old_policy.mlp_extractor.steering_net[0].in_features
    new_weight = new_policy.mlp_extractor.steering_net[0].weight
    assert torch.all(new_weight[:, old_in:] == 0.0)
    torch.testing.assert_close(
        new_weight[:, :old_in], old_policy.mlp_extractor.steering_net[0].weight
    )


def test_expansion_rejects_non_expanding_target():
    old_policy = _build_old_policy()
    with_same_shape = _build_old_policy()  # same steering input size, not an expansion
    import pytest

    with pytest.raises(ValueError):
        expand_steering_input_from_checkpoint(old_policy, with_same_shape)
