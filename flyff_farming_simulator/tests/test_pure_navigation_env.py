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


def _make_env(mode: str, *, episode_steps: int = 60, reward_mode: str = "safety"):
    entry, raw_env = next(iter(iter_variant_environments(
        "synthetic_curriculum/curriculum.json", stage="early", seed=0,
        episode_steps=episode_steps, episode_seconds=30.0,
    )))
    return (
        PureNavigationWrapper(NavigationHistoryWrapper(raw_env), target_mode=mode, reward_mode=reward_mode),
        raw_env,
    )


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


class TestSteeringOnly:
    def test_event_action_is_always_forced_to_none(self):
        """2026-08-10 correction: EVA/jump must never reach the underlying
        env, regardless of what the policy's event head requests -- EVA's
        cast-lock movement suppression could otherwise become an accidental
        collision-avoidance mechanism, contaminating the steering-only
        question this experiment is meant to answer."""
        env, raw_env = _make_env("normal_target")
        env.reset(seed=0)
        total_valid_eva_before = raw_env.total_valid_eva_casts
        for _tick in range(15):
            # Request CAST_EVA (event=1) every tick; the wrapper must
            # override this to NONE (0) before it reaches the env.
            _obs, _r, term, trunc, _info = env.step(np.asarray([0, 1], dtype=np.int64))
            if term or trunc:
                break
        env.close()
        assert raw_env.total_valid_eva_casts == total_valid_eva_before


class TestGoalRewardMode:
    def test_reward_matches_internally_tracked_distance_reduction(self):
        """Under reward_mode='goal', reward must equal the per-tick
        REDUCTION in distance to the CURRENT target, with both the "before"
        and "after" distance computed relative to the SAME (current)
        target -- not raw displacement, and not a naive before/after-target
        mismatch (see the 2026-08-10 correction: comparing distance to
        whatever target was active a tick ago against distance to a
        possibly-DIFFERENT current target injects a spurious reward spike
        on every target switch)."""
        env, raw_env = _make_env("stable_waypoint", reward_mode="goal")
        env.reset(seed=0)
        prev_x, prev_z = env._prev_player_x, env._prev_player_z

        _obs, reward, term, _trunc, _info = env.step(np.asarray([0, 0], dtype=np.int64))  # STRAIGHT

        if not term:
            # Recompute independently using the CURRENT target for both
            # terms, matching the wrapper's own (fixed) methodology.
            dist_before = env._distance_to_current_target(prev_x, prev_z)
            dist_after = env._distance_to_current_target(raw_env.player_x, raw_env.player_z)
            assert dist_before is not None and dist_after is not None
            assert abs(reward - (dist_before - dist_after)) < 1.0e-6
        env.close()

    def test_target_switch_does_not_produce_reward_discontinuity(self):
        """Directly forces a target switch mid-episode (by killing the
        current target) and confirms the reward on that exact tick is still
        a small, sane progress value -- not a large spurious spike from
        comparing distance-to-old-target against distance-to-new-target."""
        env, raw_env = _make_env("normal_target", reward_mode="goal", episode_steps=100)
        env.reset(seed=0)
        for _tick in range(10):
            _obs, _r, term, trunc, _info = env.step(np.asarray([0, 0], dtype=np.int64))
            if term or trunc:
                return  # collided or ended before we could force a switch; inconclusive, not a failure

        target_id = raw_env._nearest_reachable_actor_id
        for actor in raw_env.actors:
            if actor.actor_id == target_id:
                actor.alive = False
                break

        _obs, reward, term, _trunc, _info = env.step(np.asarray([0, 0], dtype=np.int64))
        if not term:
            # One tick of native-unit movement is bounded by the movement
            # model's distance scale (a few cells at most) -- a genuine
            # progress reward this large would indicate a discontinuity,
            # not real navigation progress.
            assert abs(reward) < 20.0
        env.close()
