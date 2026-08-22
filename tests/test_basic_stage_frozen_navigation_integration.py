"""Proves Basic's rollout loop (simulator/basic_environment.py) actually
routes steering through the production navigation stack -- frozen 0051200 +
select_persistent_waypoint -- and never through the trainable policy's own
(deliberately untrained) steering head. This is the "no direct-target
bypass remains" proof for the Basic stage specifically, mirroring the
mismatch discovered and documented in docs/architecture/
CURRICULUM_TRAINING_PIPELINE.md before this integration existed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import simulator.basic_environment as basic_environment
from simulator.basic_environment import _roll_basic_episode, collect_basic_dagger_dataset, save_basic_dagger_dataset
from simulator.basic_training import bootstrap_event_head, build_fresh_basic_policy
from simulator.navigation_dataset import MiningConfig
from simulator.navigation_history import NavigationHistoryWrapper
from simulator.navigation_subpolicy import FrozenNavigationSteering
from simulator.synthetic import iter_variant_environments

ROOT = Path(__file__).resolve().parents[1]
DAGGER_CURRICULUM = str(ROOT / "simulator" / "curricula" / "synthetic_curriculum_phase2_dagger_siblings_v2" / "curriculum.json")
LAYOUT = "01_early_open_field_typical_fast"


@pytest.fixture(scope="module")
def fresh_model():
    entry, probe_env = next(iter(iter_variant_environments(
        DAGGER_CURRICULUM, stage="early", seed=0, episode_steps=5, episode_seconds=5.0, variant_name=LAYOUT,
    )))
    wrapped = NavigationHistoryWrapper(probe_env)
    model = build_fresh_basic_policy(wrapped, seed=0, device="cpu")
    wrapped.close()
    return model


@pytest.fixture(scope="module")
def navigation_steering():
    return FrozenNavigationSteering.load_frozen(device="cpu")


def test_roll_basic_episode_calls_frozen_navigation_steering_not_the_net(fresh_model, navigation_steering, monkeypatch):
    call_count = 0
    real_steering_action = FrozenNavigationSteering.steering_action

    def _spy(self, env, *, target_actor_id):
        nonlocal call_count
        call_count += 1
        return real_steering_action(self, env, target_actor_id=target_actor_id)

    monkeypatch.setattr(FrozenNavigationSteering, "steering_action", _spy)

    records, summary = _roll_basic_episode(
        DAGGER_CURRICULUM, LAYOUT, seed=0, model=fresh_model, navigation_steering=navigation_steering,
        episode_seconds=15.0, max_actions=30, history_window=20, expected_clear_path_displacement=1.0,
    )

    assert len(records) > 0
    assert call_count > 0, "FrozenNavigationSteering.steering_action was never called -- steering bypassed the router"
    assert call_count <= len(records)


def test_roll_basic_episode_ignores_the_nets_own_steering_output(fresh_model, navigation_steering, monkeypatch):
    """Directly proves record.policy_steering never reflects the trainable
    net's own (deliberately untrained) steering head: force _policy_forward
    to return an impossible sentinel steering value every tick (keeping its
    real event output intact) and confirm that sentinel never appears in the
    recorded steering -- it can only have come from FrozenNavigationSteering."""
    SENTINEL_STEERING = 999
    real_policy_forward = basic_environment._policy_forward

    def _rigged_policy_forward(net, observation):
        _real_steering, event = real_policy_forward(net, observation)
        return SENTINEL_STEERING, event

    monkeypatch.setattr(basic_environment, "_policy_forward", _rigged_policy_forward)

    records, _ = _roll_basic_episode(
        DAGGER_CURRICULUM, LAYOUT, seed=0, model=fresh_model, navigation_steering=navigation_steering,
        episode_seconds=15.0, max_actions=20, history_window=20, expected_clear_path_displacement=1.0,
    )

    assert len(records) > 0
    assert all(r.policy_steering != SENTINEL_STEERING for r in records), (
        "the trainable net's own rigged steering sentinel leaked into the executed/recorded steering -- "
        "steering must come entirely from FrozenNavigationSteering, never from the net's own head"
    )
    assert all(r.policy_steering in (0, 1, 2) for r in records)


def test_collect_basic_dagger_dataset_runs_end_to_end_with_frozen_navigation(fresh_model, navigation_steering):
    mined = collect_basic_dagger_dataset(
        DAGGER_CURRICULUM, [LAYOUT], seeds=[0], model=fresh_model, navigation_steering=navigation_steering,
        episode_seconds=15.0, max_actions=30,
        config=MiningConfig(max_events_per_layout_seed=5, max_events_per_episode=3, max_samples_per_event=1),
    )
    assert mined["observations"].shape[0] == mined["actions"].shape[0]
    assert mined["actions"].shape[1] == 2


_STEERING_PARAM_KEYS = ("mlp_extractor.steering_net", "action_net.steering_out")
_EVENT_PARAM_KEYS = ("mlp_extractor.event_net", "action_net.event_out")


def _perturb_steering_head(model, *, scale: float = 25.0, seed: int = 0) -> None:
    """Corrupts ONLY the trainable steering head's weights, far outside any
    plausible trained range -- if mining or loss secretly depended on it,
    this would change the result."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.policy.named_parameters():
            if any(key in name for key in _STEERING_PARAM_KEYS):
                param.copy_(torch.randn(param.shape, generator=generator) * scale)


def _mine_small_dataset(model, navigation_steering, seeds: list[int] | None = None):
    return collect_basic_dagger_dataset(
        DAGGER_CURRICULUM, [LAYOUT], seeds=seeds or [0], model=model, navigation_steering=navigation_steering,
        episode_seconds=15.0, max_actions=30,
        config=MiningConfig(max_events_per_layout_seed=5, max_events_per_episode=3, max_samples_per_event=1),
    )


def _fresh_model_copy(seed: int = 0):
    entry, probe_env = next(iter(iter_variant_environments(
        DAGGER_CURRICULUM, stage="early", seed=seed, episode_steps=5, episode_seconds=5.0, variant_name=LAYOUT,
    )))
    wrapped = NavigationHistoryWrapper(probe_env)
    model = build_fresh_basic_policy(wrapped, seed=seed, device="cpu")
    wrapped.close()
    return model


def test_dagger_sample_selection_is_independent_of_the_nets_steering_head():
    """The unused trainable steering-head logits must not be able to change
    WHICH ticks get mined, their categories, or their labels -- proves
    section 2.A of the frozen-navigation-sub-policy audit: mining depends
    only on FrozenNavigationSteering's output and the net's event head,
    never on the net's own (deliberately untrained) steering head. Uses two
    independently-constructed models (not the shared module fixture) so
    corrupting one's weights cannot leak into other tests."""
    baseline_model = _fresh_model_copy()
    baseline_navigation_steering = FrozenNavigationSteering.load_frozen(device="cpu")
    baseline = _mine_small_dataset(baseline_model, baseline_navigation_steering)

    perturbed_model = _fresh_model_copy()
    perturbed_model.policy.load_state_dict(baseline_model.policy.state_dict())
    _perturb_steering_head(perturbed_model)
    perturbed_navigation_steering = FrozenNavigationSteering.load_frozen(device="cpu")
    perturbed = _mine_small_dataset(perturbed_model, perturbed_navigation_steering)

    assert perturbed["categories"] == baseline["categories"]
    np.testing.assert_array_equal(perturbed["actions"], baseline["actions"])
    np.testing.assert_allclose(perturbed["observations"], baseline["observations"])


def test_basic_round_supervised_update_never_touches_steering_head(fresh_model, navigation_steering, tmp_path):
    """The only supervised update RUN_CANONICAL_BASIC.py's round loop runs
    against mined DAgger data is basic_training.bootstrap_event_head --
    proves section 2.C of the frozen-navigation-sub-policy audit: that call
    leaves the steering head bit-for-bit unchanged while genuinely updating
    the event head (so event learning is not accidentally disabled too)."""
    mined = _mine_small_dataset(fresh_model, navigation_steering, seeds=[0, 1, 2, 3, 4])
    dagger_path = tmp_path / "dagger_round.npz"
    save_basic_dagger_dataset(mined, str(dagger_path))

    before = {
        name: param.clone()
        for name, param in fresh_model.policy.named_parameters()
        if any(key in name for key in _STEERING_PARAM_KEYS + _EVENT_PARAM_KEYS)
    }

    bootstrap_event_head(fresh_model, [dagger_path], max_epochs=2, seed=0, progress_every_seconds=999.0)

    steering_changed = False
    event_changed = False
    for name, param in fresh_model.policy.named_parameters():
        if any(key in name for key in _STEERING_PARAM_KEYS):
            assert torch.equal(param, before[name]), f"{name} changed from a supervised update that must be event-only"
            steering_changed = steering_changed or not torch.equal(param, before[name])
        elif any(key in name for key in _EVENT_PARAM_KEYS):
            event_changed = event_changed or not torch.equal(param, before[name])
    assert not steering_changed
    assert event_changed, "event head did not update -- event learning must remain trainable"
