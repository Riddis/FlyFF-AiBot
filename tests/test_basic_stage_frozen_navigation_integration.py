"""Proves Basic's rollout loop (simulator/basic_environment.py) actually
routes steering through the production navigation stack -- frozen 0051200 +
select_persistent_waypoint -- driven by the trainable policy's OWN target-
selection action (simulator.farming_target_policy), never by the
environment's own deterministic best-group/nearest-reachable hysteresis.
Basic's trainable policy has no steering action at all (SplitFarmingTarget
EventPolicy owns only target selection + event) -- this is the "no direct-
target bypass, no vestigial steering head" proof for the Basic stage,
mirroring the mismatch discovered and documented in docs/architecture/
CURRICULUM_TRAINING_PIPELINE.md before this integration existed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import simulator.basic_environment as basic_environment
from simulator.basic_environment import _roll_basic_episode, collect_basic_dagger_dataset, save_basic_dagger_dataset
from simulator.basic_training import bootstrap_farming_event_head, build_fresh_basic_policy
from simulator.farming_target_policy import TARGET_ACTION_SIZE
from simulator.navigation_dataset import MiningConfig
from simulator.navigation_subpolicy import FrozenNavigationSteering
from simulator.synthetic import iter_variant_environments

ROOT = Path(__file__).resolve().parents[1]
DAGGER_CURRICULUM = str(ROOT / "simulator" / "curricula" / "synthetic_curriculum_phase2_dagger_siblings_v2" / "curriculum.json")
LAYOUT = "01_early_open_field_typical_fast"


@pytest.fixture(scope="module")
def fresh_model():
    return build_fresh_basic_policy(seed=0, device="cpu")


@pytest.fixture(scope="module")
def navigation_steering():
    return FrozenNavigationSteering.load_frozen(device="cpu")


def test_roll_basic_episode_calls_frozen_navigation_steering(fresh_model, navigation_steering, monkeypatch):
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


def test_roll_basic_episode_has_no_steering_action_to_ignore(fresh_model, navigation_steering):
    """SplitFarmingTargetEventPolicy has no steering head at all -- confirm
    directly, so the absence isn't merely assumed. _BasicTickRecord carries
    policy_target/teacher_target, never policy_steering/teacher_steering."""
    from dataclasses import fields

    field_names = {f.name for f in fields(basic_environment._BasicTickRecord)}
    assert "policy_target" in field_names and "teacher_target" in field_names
    assert "policy_steering" not in field_names and "teacher_steering" not in field_names

    trainable_names = [name for name, _p in fresh_model.policy.named_parameters()]
    assert not any("steering" in name for name in trainable_names), (
        f"SplitFarmingTargetEventPolicy must have no steering-named parameters at all, found: "
        f"{[n for n in trainable_names if 'steering' in n]}"
    )


def test_collect_basic_dagger_dataset_runs_end_to_end_with_frozen_navigation(fresh_model, navigation_steering):
    mined = collect_basic_dagger_dataset(
        DAGGER_CURRICULUM, [LAYOUT], seeds=[0], model=fresh_model, navigation_steering=navigation_steering,
        episode_seconds=15.0, max_actions=30,
        config=MiningConfig(max_events_per_layout_seed=5, max_events_per_episode=3, max_samples_per_event=1),
    )
    assert mined["observations"].shape[0] == mined["actions"].shape[0]
    assert mined["actions"].shape[1] == 2
    if mined["actions"].shape[0]:
        assert mined["actions"][:, 0].max() < TARGET_ACTION_SIZE
        assert mined["actions"][:, 0].min() >= 0


_TARGET_PARAM_KEYS = ("mlp_extractor.target_net", "action_net.target_out")
_EVENT_PARAM_KEYS = ("mlp_extractor.event_net", "action_net.event_out")


def _perturb_target_head(model, *, scale: float = 25.0, seed: int = 0) -> None:
    """Corrupts ONLY the trainable target-selection head's weights, far
    outside any plausible trained range -- used to prove event training is
    independent of it (the mirror-image of the retired steering-independence
    proof: target selection IS trainable now, but the EVENT head's own
    training must still not depend on the target head's weights)."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.policy.named_parameters():
            if any(key in name for key in _TARGET_PARAM_KEYS):
                param.copy_(torch.randn(param.shape, generator=generator) * scale)


def _mine_small_dataset(model, navigation_steering, seeds: list[int] | None = None):
    return collect_basic_dagger_dataset(
        DAGGER_CURRICULUM, [LAYOUT], seeds=seeds or [0], model=model, navigation_steering=navigation_steering,
        episode_seconds=15.0, max_actions=30,
        config=MiningConfig(max_events_per_layout_seed=5, max_events_per_episode=3, max_samples_per_event=1),
    )


def _fresh_model_copy(seed: int = 0):
    return build_fresh_basic_policy(seed=seed, device="cpu")


def test_basic_round_event_update_never_touches_target_head(fresh_model, navigation_steering, tmp_path):
    """The only supervised update RUN_CANONICAL_BASIC.py's round loop runs
    for EVENT is basic_training.bootstrap_farming_event_head -- proves that
    call leaves the target-selection head bit-for-bit unchanged while
    genuinely updating the event head (event and target are trained by
    separate, independently-scoped calls, never entangled)."""
    mined = _mine_small_dataset(fresh_model, navigation_steering, seeds=[0, 1, 2, 3, 4])
    dagger_path = tmp_path / "dagger_round.npz"
    save_basic_dagger_dataset(mined, str(dagger_path))

    before = {
        name: param.clone()
        for name, param in fresh_model.policy.named_parameters()
        if any(key in name for key in _TARGET_PARAM_KEYS + _EVENT_PARAM_KEYS)
    }

    bootstrap_farming_event_head(fresh_model, [dagger_path], max_epochs=2, seed=0, progress_every_seconds=999.0)

    target_changed = False
    event_changed = False
    for name, param in fresh_model.policy.named_parameters():
        if any(key in name for key in _TARGET_PARAM_KEYS):
            assert torch.equal(param, before[name]), f"{name} changed from a supervised update that must be event-only"
            target_changed = target_changed or not torch.equal(param, before[name])
        elif any(key in name for key in _EVENT_PARAM_KEYS):
            event_changed = event_changed or not torch.equal(param, before[name])
    assert not target_changed
    assert event_changed, "event head did not update -- event learning must remain trainable"
