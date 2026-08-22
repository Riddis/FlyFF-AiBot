"""2026-08-14: direct contract test for RewardAblationWrapper
(scratchpad_generalized_waypoint_train_reward_ablation.py), per explicit
user instruction -- run WHILE the A/B ablation training jobs are still in
progress, as a safety check on whether the two ablations are genuinely
isolated (only the intended variable differs from baseline). If this
fails, the ablations are confounded and the running jobs must be stopped
immediately.

Verifies, from an identical world/spec/start-state, feeding the EXACT
SAME fixed action sequence to baseline (StaticWaypointWrapper,
unmodified), timeout_failure_only, and living_cost_only:
  - observations, rewards, and contacts are numerically identical between
    baseline and timeout_failure_only on every NON-timeout tick
  - at horizon expiry, baseline returns terminated=False, truncated=True
  - timeout_failure_only returns the SAME final observation and the SAME
    final reward as baseline, but terminated=True, truncated=False
    (per the user's terminology correction: this is precisely
    "timeout_terminal_no_bootstrap" -- no negative reward is applied, only
    the done-flag semantics change, removing SB3's value bootstrap)
  - living_cost_only preserves baseline's terminated/truncated semantics
    EXACTLY, including at horizon expiry
  - every living_cost_only reward equals baseline's reward minus EXACTLY
    LIVING_COST (0.0441), on EVERY tick including the terminal one -- the
    implementation's intentional, verified convention (the cost applies
    uniformly to ordinary progress ticks AND to the terminal success/
    collision payout, not just to ordinary ticks)
  - player trajectory (position/heading) and raw observations are
    identical across all three conditions for the same fixed action
    sequence (none of these wrappers touch movement physics)

Uses a fixed action sequence and a waypoint placed directly BEHIND the
player (bearing=180deg) so a plain STRAIGHT sequence for the whole
episode is guaranteed to time out (distance only increases) without any
risk of success or collision -- fully deterministic, no policy/model
involved at all, fast.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from simulator.scratchpad.scratchpad_generalized_waypoint_train_reward_ablation import LIVING_COST, RewardAblationWrapper
from simulator.navigation_history import NavigationHistoryWrapper
from simulator.static_waypoint_env import StaticWaypointWrapper, WaypointSpec, make_static_waypoint_env

TEST_EPISODE_STEPS = 10
TEST_SEED = 1234567
# Directly behind the player (bearing=180deg relative to heading=0) and
# far enough that TEST_EPISODE_STEPS of pure STRAIGHT can never reach it --
# guarantees a timeout with zero risk of success. TEST_EPISODE_STEPS is
# deliberately kept well below MAP_HALF_SIZE_CELLS=40 / PATH_LENGTH_
# CELLS_PER_TICK~=2.74 =~ 14.6 ticks -- confirmed directly (not assumed)
# that a longer horizon runs the player off the open map's edge, itself a
# genuine COLLISION with the boundary (reward=-500, terminated=True) at
# tick 14 -- which would test the wrong branch entirely (this was caught
# by test_setup_is_valid_guaranteed_timeout failing on first attempt at
# TEST_EPISODE_STEPS=20; the fixture, not the wrapper, was wrong).
BEHIND_SPEC = WaypointSpec(heading=0.0, bearing=math.pi, distance=25.0, position_offset=(0.0, 0.0))
STRAIGHT_ACTION = np.asarray([0, 0], dtype=np.int64)


def _make_env(reward_mode: str | None):
    raw_env = make_static_waypoint_env(episode_steps=TEST_EPISODE_STEPS, seed=TEST_SEED)
    if reward_mode is None:
        wrapped = StaticWaypointWrapper(NavigationHistoryWrapper(raw_env), spec_source=lambda: BEHIND_SPEC)
    else:
        wrapped = RewardAblationWrapper(
            NavigationHistoryWrapper(raw_env), reward_mode=reward_mode, spec_source=lambda: BEHIND_SPEC,
        )
    return wrapped


def _run_fixed_sequence(reward_mode: str | None) -> dict:
    env = _make_env(reward_mode)
    obs, _info = env.reset(seed=TEST_SEED)
    base_env = env.unwrapped
    distance_before = env.initial_distance_cells
    ticks = []
    for _tick in range(TEST_EPISODE_STEPS):
        obs, reward, terminated, truncated, info = env.step(STRAIGHT_ACTION)
        distance_after = env._prev_distance  # updated in-place by StaticWaypointWrapper.step()
        ticks.append({
            "obs": np.asarray(obs, dtype=np.float32).copy(),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "contacts": int(info.get("contacts", 0)),
            "player_x": base_env.player_x,
            "player_z": base_env.player_z,
            "heading": base_env.heading,
            "distance_before": distance_before,
            "distance_after": distance_after,
        })
        distance_before = distance_after
        if terminated or truncated:
            break
    env.close()
    return {"ticks": ticks}


class TestRewardAblationWrapperContract:
    def test_setup_is_valid_guaranteed_timeout(self):
        """Sanity check on the test fixture itself: confirm STRAIGHT for
        the full horizon toward a behind-the-player target really does
        time out, not succeed or collide -- otherwise the rest of this
        test class would be checking the wrong branch."""
        baseline = _run_fixed_sequence(None)
        last = baseline["ticks"][-1]
        assert len(baseline["ticks"]) == TEST_EPISODE_STEPS
        assert last["terminated"] is False
        assert last["truncated"] is True
        assert last["contacts"] == 0
        assert last["reward"] not in (20.0, -500.0)  # not a success/collision terminal payout

    def test_timeout_failure_only_matches_baseline_on_every_non_timeout_tick(self):
        baseline = _run_fixed_sequence(None)
        variant = _run_fixed_sequence("timeout_failure_only")
        assert len(baseline["ticks"]) == len(variant["ticks"])
        for i, (b, v) in enumerate(zip(baseline["ticks"], variant["ticks"])):
            if i == len(baseline["ticks"]) - 1:
                continue  # final (timeout) tick checked separately below
            np.testing.assert_array_equal(b["obs"], v["obs"], err_msg=f"tick {i}: observation mismatch")
            assert b["reward"] == v["reward"], f"tick {i}: reward mismatch"
            assert b["contacts"] == v["contacts"], f"tick {i}: contacts mismatch"
            assert b["terminated"] == v["terminated"] == False, f"tick {i}: unexpected terminal"
            assert b["truncated"] == v["truncated"] == False, f"tick {i}: unexpected truncation"
            assert b["player_x"] == v["player_x"] and b["player_z"] == v["player_z"] and b["heading"] == v["heading"], \
                f"tick {i}: trajectory diverged"

    def test_timeout_failure_only_final_tick_same_obs_and_reward_different_done_flags(self):
        baseline = _run_fixed_sequence(None)
        variant = _run_fixed_sequence("timeout_failure_only")
        b_final, v_final = baseline["ticks"][-1], variant["ticks"][-1]

        np.testing.assert_array_equal(b_final["obs"], v_final["obs"], err_msg="final observation must be identical")
        assert b_final["reward"] == v_final["reward"], "final reward must be identical (no penalty added)"

        assert b_final["terminated"] is False and b_final["truncated"] is True, "baseline must be a plain truncation"
        assert v_final["terminated"] is True and v_final["truncated"] is False, \
            "timeout_failure_only must report a genuine terminal, not a truncation"

    def test_living_cost_only_preserves_baseline_done_flags_exactly(self):
        baseline = _run_fixed_sequence(None)
        variant = _run_fixed_sequence("living_cost_only")
        assert len(baseline["ticks"]) == len(variant["ticks"])
        for i, (b, v) in enumerate(zip(baseline["ticks"], variant["ticks"])):
            assert b["terminated"] == v["terminated"], f"tick {i}: terminated flag diverged"
            assert b["truncated"] == v["truncated"], f"tick {i}: truncated flag diverged"

    def test_living_cost_only_reward_differs_by_exactly_living_cost_every_tick(self):
        baseline = _run_fixed_sequence(None)
        variant = _run_fixed_sequence("living_cost_only")
        assert len(baseline["ticks"]) == len(variant["ticks"])
        for i, (b, v) in enumerate(zip(baseline["ticks"], variant["ticks"])):
            assert v["reward"] == pytest.approx(b["reward"] - LIVING_COST, abs=1e-9), (
                f"tick {i}: living_cost_only reward {v['reward']} != baseline {b['reward']} - {LIVING_COST}"
            )
            # Applies uniformly, including whatever the final (terminal-ish)
            # tick's convention is -- explicitly confirmed here, not assumed.
            if i == len(baseline["ticks"]) - 1:
                assert v["reward"] == b["reward"] - LIVING_COST, "living cost must also apply on the final tick"

    def test_living_cost_only_trajectory_and_observations_identical_to_baseline(self):
        """The living cost only touches the scalar reward -- physics,
        observations, and contacts must be untouched."""
        baseline = _run_fixed_sequence(None)
        variant = _run_fixed_sequence("living_cost_only")
        for i, (b, v) in enumerate(zip(baseline["ticks"], variant["ticks"])):
            np.testing.assert_array_equal(b["obs"], v["obs"], err_msg=f"tick {i}: observation mismatch")
            assert b["contacts"] == v["contacts"], f"tick {i}: contacts mismatch"
            assert b["player_x"] == v["player_x"] and b["player_z"] == v["player_z"] and b["heading"] == v["heading"], \
                f"tick {i}: trajectory diverged"

    def test_timeout_failure_only_trajectory_identical_to_baseline(self):
        baseline = _run_fixed_sequence(None)
        variant = _run_fixed_sequence("timeout_failure_only")
        for i, (b, v) in enumerate(zip(baseline["ticks"], variant["ticks"])):
            assert b["player_x"] == v["player_x"] and b["player_z"] == v["player_z"] and b["heading"] == v["heading"], \
                f"tick {i}: trajectory diverged"

    # -- combined ("both") mode, added 2026-08-14 before launching the
    # combined A+B training run: "both" is a straight composition of the
    # two already-verified modes (RewardAblationWrapper.step() applies the
    # timeout_failure_only done-flag rewrite AND the living_cost_only
    # reward subtraction independently), so this re-verifies the same
    # invariants rather than assuming composition is safe.

    def test_combined_trajectory_and_observations_identical_to_baseline(self):
        baseline = _run_fixed_sequence(None)
        variant = _run_fixed_sequence("both")
        assert len(baseline["ticks"]) == len(variant["ticks"])
        for i, (b, v) in enumerate(zip(baseline["ticks"], variant["ticks"])):
            np.testing.assert_array_equal(b["obs"], v["obs"], err_msg=f"tick {i}: observation mismatch")
            assert b["contacts"] == v["contacts"], f"tick {i}: contacts mismatch"
            assert b["player_x"] == v["player_x"] and b["player_z"] == v["player_z"] and b["heading"] == v["heading"], \
                f"tick {i}: trajectory diverged"

    def test_combined_reward_differs_by_exactly_living_cost_every_tick(self):
        baseline = _run_fixed_sequence(None)
        variant = _run_fixed_sequence("both")
        for i, (b, v) in enumerate(zip(baseline["ticks"], variant["ticks"])):
            assert v["reward"] == pytest.approx(b["reward"] - LIVING_COST, abs=1e-9), (
                f"tick {i}: combined reward {v['reward']} != baseline {b['reward']} - {LIVING_COST}"
            )

    def test_combined_ordinary_ticks_have_baseline_done_flags(self):
        baseline = _run_fixed_sequence(None)
        variant = _run_fixed_sequence("both")
        for i, (b, v) in enumerate(zip(baseline["ticks"], variant["ticks"])):
            if i == len(baseline["ticks"]) - 1:
                continue  # final (timeout) tick checked separately below
            assert b["terminated"] == v["terminated"] == False, f"tick {i}: unexpected terminal"
            assert b["truncated"] == v["truncated"] == False, f"tick {i}: unexpected truncation"

    def test_combined_final_tick_true_terminal_no_bootstrap(self):
        baseline = _run_fixed_sequence(None)
        variant = _run_fixed_sequence("both")
        b_final, v_final = baseline["ticks"][-1], variant["ticks"][-1]

        assert b_final["terminated"] is False and b_final["truncated"] is True, "baseline must be a plain truncation"
        assert v_final["terminated"] is True and v_final["truncated"] is False, (
            "combined must report a genuine terminal at horizon expiry, not a truncation -- "
            "so SB3 receives no TimeLimit.truncated bootstrap"
        )
        assert v_final["reward"] == pytest.approx(b_final["reward"] - LIVING_COST, abs=1e-9), (
            "final reward must be baseline's final reward minus exactly the living cost -- "
            "true-terminal semantics must not also zero out or otherwise alter the reward"
        )

    # -- combined_discount_consistent_progress mode, added 2026-08-14 after
    # the potential-shaping audit (scratchpad_potential_shaping_audit.py)
    # confirmed the telescoping identity holds to floating-point precision.
    # This mode = "both" (true-terminal + living cost) PLUS replacing the
    # plain progress delta with the potential-shaped form on ordinary ticks.

    def test_discount_consistent_trajectory_and_observations_identical_to_baseline(self):
        baseline = _run_fixed_sequence(None)
        variant = _run_fixed_sequence("combined_discount_consistent_progress")
        assert len(baseline["ticks"]) == len(variant["ticks"])
        for i, (b, v) in enumerate(zip(baseline["ticks"], variant["ticks"])):
            np.testing.assert_array_equal(b["obs"], v["obs"], err_msg=f"tick {i}: observation mismatch")
            assert b["contacts"] == v["contacts"], f"tick {i}: contacts mismatch"
            assert b["player_x"] == v["player_x"] and b["player_z"] == v["player_z"] and b["heading"] == v["heading"], \
                f"tick {i}: trajectory diverged"
            assert b["distance_before"] == v["distance_before"] and b["distance_after"] == v["distance_after"], \
                f"tick {i}: distance sequence diverged -- reward math below assumes these match exactly"

    def test_discount_consistent_ordinary_ticks_have_baseline_done_flags(self):
        baseline = _run_fixed_sequence(None)
        variant = _run_fixed_sequence("combined_discount_consistent_progress")
        for i, (b, v) in enumerate(zip(baseline["ticks"], variant["ticks"])):
            if i == len(baseline["ticks"]) - 1:
                continue  # final (timeout) tick checked separately below
            assert b["terminated"] == v["terminated"] == False, f"tick {i}: unexpected terminal"
            assert b["truncated"] == v["truncated"] == False, f"tick {i}: unexpected truncation"

    def test_discount_consistent_final_tick_true_terminal_no_bootstrap(self):
        baseline = _run_fixed_sequence(None)
        variant = _run_fixed_sequence("combined_discount_consistent_progress")
        b_final, v_final = baseline["ticks"][-1], variant["ticks"][-1]
        assert b_final["terminated"] is False and b_final["truncated"] is True, "baseline must be a plain truncation"
        assert v_final["terminated"] is True and v_final["truncated"] is False, (
            "combined_discount_consistent_progress must report a genuine terminal at horizon expiry"
        )

    def test_discount_consistent_reward_matches_potential_shaped_formula_every_tick(self):
        """Every tick in this fixture is an ordinary progress tick (the
        setup guarantees timeout, never success/collision -- verified by
        test_setup_is_valid_guaranteed_timeout), so EVERY tick's reward
        must equal exactly (distance_before - GAMMA*distance_after) -
        LIVING_COST, including the final (now true-terminal) tick --
        matching living_cost_only's own established convention that the
        cost applies uniformly, and confirming the progress reformulation
        applies on the terminal tick too, not just ordinary ones."""
        from simulator.scratchpad.scratchpad_generalized_waypoint_train_reward_ablation import GAMMA

        variant = _run_fixed_sequence("combined_discount_consistent_progress")
        for i, v in enumerate(variant["ticks"]):
            expected = v["distance_before"] - GAMMA * v["distance_after"] - LIVING_COST
            assert v["reward"] == pytest.approx(expected, abs=1e-9), (
                f"tick {i}: reward {v['reward']} != expected potential-shaped value {expected}"
            )

    def test_discount_consistent_differs_from_both_mode_by_exactly_the_shaping_term(self):
        """Isolates the ONE thing this mode changes relative to the
        already-validated "both" mode: reward_new - reward_both must equal
        exactly distance_after*(1-GAMMA) on every tick -- the same
        mechanistic identity verified in scratchpad_potential_shaping_
        audit.py's Part B, now checked against the actual wrapper output
        rather than an offline recomputation."""
        from simulator.scratchpad.scratchpad_generalized_waypoint_train_reward_ablation import GAMMA

        both = _run_fixed_sequence("both")
        variant = _run_fixed_sequence("combined_discount_consistent_progress")
        for i, (b, v) in enumerate(zip(both["ticks"], variant["ticks"])):
            expected_delta = v["distance_after"] * (1.0 - GAMMA)
            actual_delta = v["reward"] - b["reward"]
            assert actual_delta == pytest.approx(expected_delta, abs=1e-9), (
                f"tick {i}: reward delta vs 'both' {actual_delta} != expected {expected_delta}"
            )
