"""Verifies simulator.factorized_v193_training.transfer_event_head_to_event_only_policy:
after transfer, the fresh event-only policy's event-action distribution
must be provably identical to the source SplitSteeringNavigationPolicy's
own event branch on real observations -- the Basic -> Beginner continuation
point for the recovered frozen-navigation-sub-policy architecture (see
docs/architecture/CURRICULUM_TRAINING_PIPELINE.md section 4).
"""
from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_schedule_fn

from farming.actions import FarmingEvent
from navigation.navigation_evidence import POLICY_INPUT_SIZE, RAW_OBSERVATION_SIZE
from simulator.factorized_v193_training import transfer_event_head_to_event_only_policy
from simulator.split_branch_policy import SplitSteeringNavigationPolicy
from simulator.synthetic import iter_variant_environments

NET_ARCH = [256, 128]


def _build_split_policy() -> SplitSteeringNavigationPolicy:
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(POLICY_INPUT_SIZE,), dtype=np.float32)
    act_space = spaces.MultiDiscrete([3, 3])
    return SplitSteeringNavigationPolicy(
        obs_space, act_space, get_schedule_fn(3e-4),
        steering_net_arch=[32, 16], event_net_arch=NET_ARCH, vf_net_arch=NET_ARCH,
    )


def _build_plain_event_only_policy() -> ActorCriticPolicy:
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(RAW_OBSERVATION_SIZE,), dtype=np.float32)
    act_space = spaces.Discrete(len(FarmingEvent))
    return ActorCriticPolicy(
        obs_space, act_space, get_schedule_fn(3e-4), net_arch=dict(pi=list(NET_ARCH), vf=list(NET_ARCH)),
    )


def _real_raw_observations(n: int) -> np.ndarray:
    entry, env = next(iter(iter_variant_environments(
        "simulator/curricula/synthetic_curriculum/curriculum.json", stage="early", seed=0,
        episode_steps=n + 5, episode_seconds=30.0,
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


def test_transfer_reproduces_source_event_distribution_exactly():
    torch.manual_seed(0)
    split_policy = _build_split_policy()
    plain_policy = _build_plain_event_only_policy()
    transfer_event_head_to_event_only_policy(split_policy, plain_policy)

    raw = _real_raw_observations(8)[:, :RAW_OBSERVATION_SIZE]
    # Deliberately nonzero sidecar values in the SPLIT policy's own 928-dim
    # input -- event_net never reads the sidecar slice at all (only the raw
    # 923), so this also proves the transfer is correct regardless of what
    # sidecar values the source saw.
    sidecar = np.tile(np.asarray([0.6, 0.9, 1.0, 0.0, 0.0], dtype=np.float32), (raw.shape[0], 1))
    split_input = torch.as_tensor(np.concatenate([raw, sidecar], axis=1), dtype=torch.float32)
    plain_input = torch.as_tensor(raw, dtype=torch.float32)

    with torch.no_grad():
        split_dist = split_policy.get_distribution(split_input).distribution
        plain_dist = plain_policy.get_distribution(plain_input).distribution
        split_value = split_policy.predict_values(split_input)
        plain_value = plain_policy.predict_values(plain_input)

    split_event_probs = split_dist[1].probs
    plain_event_probs = plain_dist.probs

    torch.testing.assert_close(split_event_probs, plain_event_probs, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(split_value, plain_value, atol=1e-5, rtol=1e-4)


def test_transfer_raises_on_shape_mismatch():
    split_policy = _build_split_policy()
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(RAW_OBSERVATION_SIZE,), dtype=np.float32)
    act_space = spaces.Discrete(len(FarmingEvent))
    mismatched_policy = ActorCriticPolicy(
        obs_space, act_space, get_schedule_fn(3e-4), net_arch=dict(pi=[64, 32], vf=[64, 32]),
    )
    try:
        transfer_event_head_to_event_only_policy(split_policy, mismatched_policy)
        assert False, "expected a shape-mismatch failure"
    except RuntimeError:
        pass
