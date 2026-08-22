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

import simulator.basic_environment as basic_environment
from simulator.basic_environment import _roll_basic_episode, collect_basic_dagger_dataset
from simulator.basic_training import build_fresh_basic_policy
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
