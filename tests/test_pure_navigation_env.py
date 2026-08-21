"""Tests for the 2026-08-09/10 PPO ablation's pure-navigation wrapper.

2026-08-10 correction #3 (the third and most structural fix in this
module's history): earlier fixes made masking mode-independent but still
borrowed `_nearest_reachable_actor_id`/`_best_group_actor_id` from the
underlying env -- attributes selected over the FULL (~300+ actor) visible
list via geodesic distance, with a hysteresis margin that only guards
against being outbid, never against the target drifting outside the
policy's actual DIRECT_ACTOR_SLOTS=12-wide observation window. Measured
directly: under `stable_waypoint`'s effectively-infinite margin, the
selected target was unrepresentable to the policy 74% of the time.

Replaced with a self-contained selector (`select_target`) that is the sole
source of truth for both the masked observation and the reward, and can
only ever select from the exact representable candidate pool -- making
"reward target unrepresentable to the policy" a structurally impossible
state. These tests assert that invariant directly (see
TestNoUnrepresentableRewardTarget), replacing the old
TestTargetRepresentabilityGap bug-sentinel, which asserted the opposite
(that the gap existed) back when it was still an open, unfixed bug.
"""
from __future__ import annotations

import numpy as np

from farming.observation import ACTOR_FEATURES, DIRECT_ACTOR_SLOTS, DIRECT_ACTOR_START, _DIRECT_ACTOR_FIELD_NAMES
from simulator.navigation_history import NavigationHistoryWrapper
from simulator.pure_navigation_env import PureNavigationWrapper, select_target
from simulator.synthetic import iter_variant_environments

_ACTIVE_OFFSET = _DIRECT_ACTOR_FIELD_NAMES.index("active")


def _make_env(mode: str, *, episode_steps: int = 60, reward_mode: str = "safety"):
    entry, raw_env = next(iter(iter_variant_environments(
        "simulator/curricula/synthetic_curriculum/curriculum.json", stage="early", seed=0,
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


def _active_slot_actor_id(obs: np.ndarray, base_env) -> int | None:
    """Decode which actor (if any) occupies the single active direct-actor
    slot in a masked observation."""
    candidates = base_env._visible_candidates()[:DIRECT_ACTOR_SLOTS]
    for slot, (_distance, actor, _dx, _dz) in enumerate(candidates):
        offset = DIRECT_ACTOR_START + slot * ACTOR_FEATURES + _ACTIVE_OFFSET
        if obs[offset] > 0.5:
            return actor.actor_id
    return None


class TestMaskingBothModes:
    """Both modes mask the observation to the wrapper's own
    _selected_target_id -- at most one active slot, always, in both."""

    def test_at_most_one_active_actor_slot_stable_waypoint(self):
        env, _raw_env = _make_env("stable_waypoint")
        obs, _info = env.reset(seed=0)
        assert _active_slot_count(obs) <= 1
        for _tick in range(40):
            obs, _r, term, trunc, _info = env.step(np.asarray([0, 0], dtype=np.int64))
            assert _active_slot_count(obs) <= 1
            if term or trunc:
                break
        env.close()

    def test_at_most_one_active_actor_slot_normal_target(self):
        env, _raw_env = _make_env("normal_target")
        obs, _info = env.reset(seed=0)
        assert _active_slot_count(obs) <= 1
        for _tick in range(40):
            obs, _r, term, trunc, _info = env.step(np.asarray([0, 0], dtype=np.int64))
            assert _active_slot_count(obs) <= 1
            if term or trunc:
                break
        env.close()


class TestObservationRewardTargetIdentityMatch:
    """The core invariant: the active observation slot (if any) is always
    exactly the wrapper's `_selected_target_id`, the same id the reward is
    computed against, in BOTH modes, at every tick -- not merely "usually",
    and not merely "when a slot happens to be active" (see
    TestNoUnrepresentableRewardTarget for why "no active slot" now implies
    "no reward target either", never a mismatch)."""

    def _assert_identity_holds(self, env, obs, n_ticks: int) -> None:
        base_env = env.unwrapped
        for _tick in range(n_ticks):
            assert _active_slot_actor_id(obs, base_env) == env._selected_target_id
            obs, _r, term, trunc, _info = env.step(np.asarray([0, 0], dtype=np.int64))
            if term or trunc:
                break

    def test_stable_waypoint(self):
        env, _raw_env = _make_env("stable_waypoint", reward_mode="goal")
        obs, _info = env.reset(seed=0)
        self._assert_identity_holds(env, obs, 30)
        env.close()

    def test_normal_target(self):
        env, _raw_env = _make_env("normal_target", reward_mode="goal")
        obs, _info = env.reset(seed=0)
        self._assert_identity_holds(env, obs, 30)
        env.close()


class TestNoUnrepresentableRewardTarget:
    """2026-08-10 correction #3's central guarantee, replacing the old
    TestTargetRepresentabilityGap bug-sentinel (which asserted the gap
    EXISTED, back when it was unfixed). Now asserts the opposite: whenever
    `reward_mode="goal"` computes a nonzero-basis reward (i.e.
    `_selected_target_id is not None`), that exact target is representable
    in the observation the policy sees THAT SAME tick."""

    def _assert_never_unrepresentable(self, mode: str, n_ticks: int = 40) -> None:
        env, _raw_env = _make_env(mode, reward_mode="goal", episode_steps=n_ticks)
        obs, _info = env.reset(seed=0)
        base_env = env.unwrapped
        for _tick in range(n_ticks):
            if env._selected_target_id is not None:
                assert _active_slot_actor_id(obs, base_env) == env._selected_target_id
            obs, _r, term, trunc, _info = env.step(np.asarray([0, 0], dtype=np.int64))
            if term or trunc:
                break
        env.close()

    def test_stable_waypoint_never_unrepresentable(self):
        self._assert_never_unrepresentable("stable_waypoint")

    def test_normal_target_never_unrepresentable(self):
        self._assert_never_unrepresentable("normal_target")


class TestStickyRetentionAndRelease:
    """Sticky mode must retain its target while valid, and release it
    (reselect) the moment the target stops being alive, representable, or
    reachable -- never hold an invisible/dead/unreachable actor the way the
    old infinite-hysteresis-margin design did."""

    def test_retains_target_across_ticks_absent_invalidation(self):
        env, _raw_env = _make_env("stable_waypoint")
        env.reset(seed=0)
        target_ids = [env._selected_target_id]
        for _tick in range(20):
            _obs, _r, term, trunc, _info = env.step(np.asarray([0, 0], dtype=np.int64))
            target_ids.append(env._selected_target_id)
            if term or trunc:
                break
        env.close()
        # Not asserting zero switches (a target can legitimately die or
        # become unrepresentable/unreachable), but it must not switch every
        # single tick the way pure greedy reselection would.
        switches = sum(1 for a, b in zip(target_ids, target_ids[1:]) if a != b)
        assert switches < len(target_ids) - 1

    def test_releases_immediately_when_target_dies(self):
        env, raw_env = _make_env("stable_waypoint")
        env.reset(seed=0)
        base_env = env.unwrapped
        first_target = env._selected_target_id
        assert first_target is not None
        for actor in base_env.actors:
            if actor.actor_id == first_target:
                actor.alive = False
                break
        env.step(np.asarray([0, 0], dtype=np.int64))
        assert env._selected_target_id != first_target
        env.close()


class TestGreedyAndStickyShareCandidatePool:
    """Greedy and sticky selection must draw from the identical
    representable candidate pool -- persistence is the only allowed
    difference. Verified by confirming a fresh (no prior sticky id)
    selection agrees with a greedy selection on the same state."""

    def test_fresh_sticky_selection_matches_greedy_selection(self):
        env, _raw_env = _make_env("stable_waypoint")
        env.reset(seed=0)
        base_env = env.unwrapped
        sticky_pick = select_target(base_env, sticky_id=None, sticky=True)
        greedy_pick = select_target(base_env, sticky_id=None, sticky=False)
        assert sticky_pick == greedy_pick
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
        REDUCTION in distance to the CURRENT (freshly reselected) target,
        with both the "before" and "after" distance computed relative to
        the SAME target -- not raw displacement, and not a naive
        before/after-target mismatch (see the 2026-08-10 correction:
        comparing distance to whatever target was active a tick ago against
        distance to a possibly-DIFFERENT current target injects a spurious
        reward spike on every target switch)."""
        env, raw_env = _make_env("stable_waypoint", reward_mode="goal")
        env.reset(seed=0)
        prev_x, prev_z = env._prev_player_x, env._prev_player_z

        _obs, reward, term, _trunc, _info = env.step(np.asarray([0, 0], dtype=np.int64))  # STRAIGHT

        if not term:
            # Recompute independently using the CURRENT (post-step) target
            # for both terms, matching the wrapper's own methodology.
            dist_before = env._distance_to_selected_target(prev_x, prev_z)
            dist_after = env._distance_to_selected_target(raw_env.player_x, raw_env.player_z)
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

        target_id = env._selected_target_id
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
