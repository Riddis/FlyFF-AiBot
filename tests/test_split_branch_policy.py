from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from farming.model_contract import ModelContractMetadata
from simulator.factorized_training import atomic_save_policy
from simulator.split_branch_policy import SplitSteeringEventPolicy
from simulator.synthetic import iter_variant_environments

_CURRICULUM = "curricula/synthetic_curriculum/curriculum.json"


def _build_model(env):
    from stable_baselines3 import PPO

    return PPO(
        SplitSteeringEventPolicy,
        env,
        n_steps=32,
        batch_size=16,
        seed=0,
        device="cpu",
        policy_kwargs={
            "steering_net_arch": [32, 16],
            "event_net_arch": [128, 64],
            "vf_net_arch": [128, 64],
        },
    )


def test_split_policy_produces_correctly_shaped_multidiscrete_distribution() -> None:
    entry, env = next(
        iter(iter_variant_environments(_CURRICULUM, stage="early", episode_steps=10, episode_seconds=4.0))
    )
    del entry
    model = _build_model(env)
    net = model.policy

    obs, _ = env.reset(seed=0)
    obs_tensor = torch.as_tensor(np.stack([obs, obs]).astype(np.float32))

    distribution = net.get_distribution(obs_tensor).distribution
    assert len(distribution) == 2
    assert distribution[0].probs.shape == (2, 3)
    assert distribution[1].probs.shape == (2, 3)

    action, value, log_prob = net(obs_tensor)
    assert action.shape == (2, 2)
    assert value.shape == (2, 1)
    env.close()


def test_split_policy_isolates_gradients_between_branches() -> None:
    entry, env = next(
        iter(iter_variant_environments(_CURRICULUM, stage="early", episode_steps=10, episode_seconds=4.0))
    )
    del entry
    model = _build_model(env)
    net = model.policy
    obs, _ = env.reset(seed=0)
    obs_tensor = torch.as_tensor(np.stack([obs, obs, obs]).astype(np.float32))
    env.close()

    def grad_norm(param: torch.nn.Parameter) -> float:
        return float(param.grad.abs().sum()) if param.grad is not None else 0.0

    net.zero_grad()
    event_loss = net.get_distribution(obs_tensor).distribution[1].logits.sum()
    event_loss.backward()
    assert grad_norm(net.action_net.steering_out.weight) == 0.0
    assert grad_norm(net.action_net.event_out.weight) > 0.0
    for param in net.mlp_extractor.steering_net.parameters():
        assert grad_norm(param) == 0.0

    net.zero_grad()
    steering_loss = net.get_distribution(obs_tensor).distribution[0].logits.sum()
    steering_loss.backward()
    assert grad_norm(net.action_net.event_out.weight) == 0.0
    assert grad_norm(net.action_net.steering_out.weight) > 0.0
    for param in net.mlp_extractor.event_net.parameters():
        assert grad_norm(param) == 0.0


def test_apply_prior_bias_correction_handles_split_head() -> None:
    from simulator.factorized_v193_training import _apply_prior_bias_correction

    entry, env = next(
        iter(iter_variant_environments(_CURRICULUM, stage="early", episode_steps=10, episode_seconds=4.0))
    )
    del entry
    model = _build_model(env)
    env.close()
    net = model.policy
    before_steering = net.action_net.steering_out.bias.detach().clone()
    before_event = net.action_net.event_out.bias.detach().clone()

    result = _apply_prior_bias_correction(
        net,
        steering_target=np.asarray([0.5, 0.3, 0.2]),
        event_target=np.asarray([0.8, 0.15, 0.05]),
        event_sampling_fractions={0: 0.55, 1: 0.45},
    )

    assert result["applied"] is True
    after_steering = net.action_net.steering_out.bias.detach().clone()
    after_event = net.action_net.event_out.bias.detach().clone()
    assert not torch.allclose(before_steering, after_steering)
    assert not torch.allclose(before_event, after_event)


def test_split_policy_save_and_load_round_trips_exactly(tmp_path: Path) -> None:
    from stable_baselines3 import PPO

    entry, env = next(
        iter(iter_variant_environments(_CURRICULUM, stage="early", episode_steps=10, episode_seconds=4.0))
    )
    del entry
    model = _build_model(env)
    setattr(model, "farming_contract_metadata", ModelContractMetadata.current().as_dict())
    obs, _ = env.reset(seed=0)
    obs_tensor = torch.as_tensor(np.stack([obs]).astype(np.float32))
    env.close()

    before = model.policy.get_distribution(obs_tensor).distribution[0].probs.detach().clone()
    path = atomic_save_policy(model, tmp_path / "checkpoint")
    reloaded = PPO.load(str(path), device="cpu")
    after = reloaded.policy.get_distribution(obs_tensor).distribution[0].probs.detach().clone()

    assert torch.allclose(before, after)
    assert isinstance(reloaded.policy, SplitSteeringEventPolicy)
    assert reloaded.policy.action_net.steering_out.in_features == model.policy.action_net.steering_out.in_features
