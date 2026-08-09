"""Tests for the 2026-08-09 PPO ablation's pure-navigation wrapper,
specifically the observation-masking fix: derive_geometry_features
independently recomputes its own target selection from the raw
observation's direct-actor slots and does NOT consult
_nearest_reachable_actor_id/_best_group_actor_id, so stable_waypoint mode
must mask the observation itself, not just set env-level hysteresis state.
"""
from __future__ import annotations

import numpy as np

from farming.observation import ACTOR_FEATURES, DIRECT_ACTOR_SLOTS, DIRECT_ACTOR_START, _DIRECT_ACTOR_FIELD_NAMES
from simulator.navigation_history import NavigationHistoryWrapper
from simulator.pure_navigation_env import PureNavigationWrapper
from simulator.synthetic import iter_variant_environments

_ACTIVE_OFFSET = _DIRECT_ACTOR_FIELD_NAMES.index("active")


def _make_env(mode: str, *, episode_steps: int = 60):
    entry, raw_env = next(iter(iter_variant_environments(
        "synthetic_curriculum/curriculum.json", stage="early", seed=0,
        episode_steps=episode_steps, episode_seconds=30.0,
    )))
    return PureNavigationWrapper(NavigationHistoryWrapper(raw_env), target_mode=mode), raw_env


def _active_slot_count(obs: np.ndarray) -> int:
    return sum(
        1 for slot in range(DIRECT_ACTOR_SLOTS)
        if obs[DIRECT_ACTOR_START + slot * ACTOR_FEATURES + _ACTIVE_OFFSET] > 0.5
    )


class TestStableWaypointMasking:
    def test_at_most_one_active_actor_slot_throughout_a_rollout(self):
        env, raw_env = _make_env("stable_waypoint")
        obs, _info = env.reset(seed=0)
        assert _active_slot_count(obs) <= 1
        for _tick in range(40):
            obs, _r, term, trunc, _info = env.step(np.asarray([0, 0], dtype=np.int64))
            assert _active_slot_count(obs) <= 1
            if term or trunc:
                break
        env.close()

    def test_target_identity_persists_across_ticks_absent_death(self):
        """The env's own sticky target ID (already stabilized by the Step 2
        hysteresis fix under an effectively-infinite margin) should not
        change every tick -- some persistence is expected, unlike the
        pervasive ~30/100-tick thrashing rate measured on the unmodified
        default."""
        env, raw_env = _make_env("stable_waypoint")
        env.reset(seed=0)
        target_ids = [raw_env._nearest_reachable_actor_id]
        for _tick in range(20):
            _obs, _r, term, trunc, _info = env.step(np.asarray([0, 0], dtype=np.int64))
            target_ids.append(raw_env._nearest_reachable_actor_id)
            if term or trunc:
                break
        env.close()
        # Not asserting zero switches (a target can legitimately die or
        # leave vision range), but it must not switch on every single tick.
        switches = sum(1 for a, b in zip(target_ids, target_ids[1:]) if a != b)
        assert switches < len(target_ids) - 1


class TestNormalTargetUnmasked:
    def test_maybe_mask_is_a_no_op_for_normal_target_mode(self):
        """normal_target must reflect today's real, unmodified observation
        pipeline. Direct invariant check: _maybe_mask must return an
        observation identical to its input for this mode (not merely
        'usually looks similar')."""
        env, _raw_env = _make_env("normal_target")
        obs, _info = env.reset(seed=0)
        masked = env._maybe_mask(obs)
        assert np.array_equal(obs, masked)
        env.close()
