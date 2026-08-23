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
from farming.actions import FarmingEvent
from navigation.movement_kernel import SteeringDirection
from simulator.basic_environment import _roll_basic_episode, collect_basic_dagger_dataset, save_basic_dagger_dataset
from simulator.basic_training import bootstrap_farming_event_head, build_fresh_basic_policy
from simulator.environment import RecordedFarmingEnv
from simulator.farming_target_policy import KEEP_CURRENT_TARGET_ACTION, TARGET_ACTION_SIZE
from simulator.navigation_dataset import MiningConfig
from simulator.navigation_subpolicy import FrozenNavigationSteering, SteeringTickResult
from simulator.synthetic import iter_variant_environments
from tests.helpers.router_qualification_harness import build_multi_wall_world

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


def test_roll_basic_episode_planner_failure_invalidates_target_and_resets_navigation_state(
    fresh_model, navigation_steering, monkeypatch,
):
    """Production-path proof (pre-merge blocker remediation, 2026-08-23):
    _roll_basic_episode is Basic's REAL rollout entrypoint -- used by both
    training (collect_basic_dagger_dataset) and Basic milestone evaluation
    (basic_milestone_evaluator.py's evaluate_basic_milestone) -- not
    PersistentFarmingTarget.invalidate() or FarmingPolicyWrapper.step in
    isolation. Forces the SECOND steering_action call (on whatever target
    the untrained policy picks) to report planner_failure=True; the FIRST
    call is real, so it establishes genuine route/controller state on
    navigation_steering that this test then proves gets cleared."""
    real_steering_action = FrozenNavigationSteering.steering_action
    call_count = 0

    def _first_call_real_then_forced_failure(self, env, *, target_actor_id):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return real_steering_action(self, env, target_actor_id=target_actor_id)
        return SteeringTickResult(
            steering=int(SteeringDirection.NONE), waypoint=(env.unwrapped.player_x, env.unwrapped.player_z),
            replanned=False, planner_failure=True,
        )

    monkeypatch.setattr(FrozenNavigationSteering, "steering_action", _first_call_real_then_forced_failure)

    records, summary = _roll_basic_episode(
        DAGGER_CURRICULUM, LAYOUT, seed=0, model=fresh_model, navigation_steering=navigation_steering,
        episode_seconds=15.0, max_actions=30, history_window=20, expected_clear_path_displacement=1.0,
    )

    assert call_count >= 2, "test setup: need at least one real call followed by a forced-failure call"
    assert summary["target_invalidations_by_planner_failure"] >= 1
    # The forced-failure call's invalidation must have actually cleared the
    # real state the first call established -- not merely incremented a
    # counter while leaving stale state behind.
    assert navigation_steering._route is None
    assert navigation_steering._controller is None
    assert navigation_steering._target_id is None
    assert navigation_steering._snapshot_pos is None
    assert len(records) > 0


def test_roll_basic_episode_keep_after_invalidation_stays_none_then_explicit_reselection_reaches_router(
    fresh_model, navigation_steering, monkeypatch,
):
    """Tightly-related companion to the test above, split out because full
    determinism (a specific target A, then an explicit reselection to a
    specific target B) requires controlling actor placement, which requires
    intercepting _roll_basic_episode's internal env.reset() -- monkeypatches
    RecordedFarmingEnv.reset to place exactly two known actors after the
    real reset runs (real reset erases any pre-placed actors), and
    basic_environment._policy_forward to drive a fixed tick-by-tick action
    sequence (still through the REAL _roll_basic_episode entrypoint, not a
    synthetic re-implementation of its logic)."""
    map_model, world = build_multi_wall_world([], population=8)
    base_env = RecordedFarmingEnv(world, map_model=map_model, episode_steps=30)
    a_pos = map_model.layout_to_native(46, 20)
    b_pos = map_model.layout_to_native(20, 45)
    a_id = 1  # RecordedFarmingEnv.reset() assigns actor ids 1..N in spawn order
    b_id = 2

    real_env_reset = RecordedFarmingEnv.reset

    def _reset_with_fixed_actors(self, *, seed=None, options=None):
        real_env_reset(self, seed=seed, options=options)
        for index, actor in enumerate(self.actors):
            if index == 0:
                actor.x, actor.z, actor.alive = a_pos[0], a_pos[1], True
            elif index == 1:
                actor.x, actor.z, actor.alive = b_pos[0], b_pos[1], True
            else:
                actor.alive = False
        observation = self._observation()
        info = self._info(kills=0, reward_components={})
        return observation, info

    def _fake_iter_variant_environments(*_args, **_kwargs):
        yield (None, base_env)

    tick = [-1]

    def _fixed_policy_forward(net, raw):
        tick[0] += 1
        t = tick[0]
        slot_ids = list(base_env._direct_actor_slot_ids)
        if t == 0:
            return slot_ids.index(a_id) + 1, int(FarmingEvent.NONE)
        if t in (1, 2):
            return KEEP_CURRENT_TARGET_ACTION, int(FarmingEvent.NONE)
        if t == 3:
            return slot_ids.index(b_id) + 1, int(FarmingEvent.NONE)
        return KEEP_CURRENT_TARGET_ACTION, int(FarmingEvent.NONE)

    real_steering_action = FrozenNavigationSteering.steering_action
    call_count = 0

    def _fail_only_the_second_call(self, env, *, target_actor_id):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            return SteeringTickResult(
                steering=int(SteeringDirection.NONE), waypoint=(env.unwrapped.player_x, env.unwrapped.player_z),
                replanned=False, planner_failure=True,
            )
        return real_steering_action(self, env, target_actor_id=target_actor_id)

    monkeypatch.setattr(RecordedFarmingEnv, "reset", _reset_with_fixed_actors)
    monkeypatch.setattr(basic_environment, "iter_variant_environments", _fake_iter_variant_environments)
    monkeypatch.setattr(basic_environment, "_policy_forward", _fixed_policy_forward)
    monkeypatch.setattr(FrozenNavigationSteering, "steering_action", _fail_only_the_second_call)

    records, summary = _roll_basic_episode(
        DAGGER_CURRICULUM, LAYOUT, seed=0, model=fresh_model, navigation_steering=navigation_steering,
        episode_seconds=15.0, max_actions=4, history_window=20, expected_clear_path_displacement=1.0,
    )

    # Tick 0: explicit select A -- real call (#1) succeeds.
    # Tick 1: KEEP persists A -- forced-failure call (#2) invalidates it.
    # Tick 2: KEEP resolves to NO target (invalidated) -- no steering_action
    #         call at all, proving KEEP never silently reacquires anything
    #         (not A again, not a heuristic substitute).
    # Tick 3: explicit select B -- real call (#3) reaches the router and
    #         succeeds, proving the policy retains full ownership of the
    #         next target decision.
    assert call_count == 3, f"expected exactly 3 real steering_action invocations (A, forced-fail, B), got {call_count}"
    assert summary["target_invalidations_by_planner_failure"] == 1
    assert navigation_steering._target_id == b_id, "final navigation state must point at B, not stale/absent A"
    assert len(records) == 4
