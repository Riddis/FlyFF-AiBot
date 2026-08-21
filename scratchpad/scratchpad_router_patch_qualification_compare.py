"""2026-08-14: paired A/B/C comparison on the fresh 830M qualification
pool, per explicit user instruction, to decide the router-patch question
out-of-sample (812M is saturated by both candidates and can no longer
discriminate).

  A = frozen 0051200 + OLD selector (pre-patch `select_persistent_waypoint`)
  B = frozen 0051200 + PATCHED selector (current production code)
  C = seed100@56320 (continuation-trained) + PATCHED selector

2026-08-15 update: after this script's own run found the patch does NOT
qualify (see `evaluations/router_patch_qualification_compare_result.json`
-- B repairs A's 4 collisions but introduces a new one, `single_wall_
right[1]`), `simulator/kinodynamic_route_planner.py`'s `select_persistent_
waypoint` (the name every production caller imports) was reverted to the
qualified three-tier algorithm, and the experimental fourth tier now
lives ONLY in the explicitly-named `select_persistent_waypoint_
experimental_collision_free_fallback`. This script's own `select_
persistent_waypoint_old` below (a frozen, standalone reimplementation
of the pre-patch algorithm, predating that production revert) is now
REDUNDANT with production `select_persistent_waypoint` itself -- kept
only because `_spot_check_old_selector_fidelity` already cross-validates
it independently and other diagnostic scripts import it by this name.
`eval_obstacle_with_selector` below explicitly monkeypatches in one of
the two NAMED functions (never relies on "whatever production currently
resolves to") so this script's A/B/C meaning stays correct regardless of
which algorithm is the current production default.

Predeclared decision rules (fixed before this pool is evaluated):

  B qualifies over A iff:
    - open: no regression (checked via the established 777M held_out
      pool -- the router patch cannot affect open-waypoint episodes at
      all, since select_persistent_waypoint is never invoked for them,
      but this is checked explicitly rather than assumed)
    - obstacle: B's collision-episode set introduces NO episode absent
      from A's collision-episode set (B_collisions subset-of A_collisions)
    - obstacle: B's success count >= A's success count
    - IF A has any collisions at all: B's collision set must be a STRICT
      subset of A's (B must actually remove at least one, not merely tie)

  C is chosen over B only if C "materially beats" B on failures:
    - C's collision-episode set is a STRICT subset of B's (removes at
      least one of B's collisions, introduces none new)
    - C's success count >= B's success count
  Otherwise (including any tie) B wins -- the frozen baseline is already
  the more established policy; a continuation-trained checkpoint is not
  carried forward for a marginal efficiency/tick improvement alone.

No training, no further router changes. This script only evaluates.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from stable_baselines3 import PPO

import scratchpad_general_router_episode as gre
from scratchpad_beginner_navigation_mix_pools import eval_obstacle_manifest, load_manifest
from scratchpad_generalized_waypoint_train_reward_ablation import eval_held_out
from scratchpad_router_patch_qualification_pool import QUALIFICATION_SPEC_SEED
from simulator.kinodynamic_route_planner import (
    DESIRED_CLEARANCE_CELLS, _direct_hop_min_clearance, annotate_route_edges,
    select_persistent_waypoint_experimental_collision_free_fallback,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE_CHECKPOINT = ROOT / "models" / "generalized_waypoint_both_seed2_0051200.zip"
TRAINED_CHECKPOINT = ROOT / "models" / "generalized_waypoint_router_mix_seed100_0056320.zip"
DEV_POOL_EPISODE_SEED_BASE = 830_600_000


def select_persistent_waypoint_old(
    map_model, route, *, player_x, player_z, heading,
    max_heading_change_radians=math.radians(75.0), min_progress_cells=2.0, min_robust_clearance_cells=2.0,
):
    """Frozen pre-patch reimplementation -- see module docstring. Matches
    the production function as it existed before 2026-08-14's
    COLLISION_FREE_LOW_MARGIN_FALLBACK tier was added: best -> safe_
    fallback -> any_fallback only."""
    if len(route) < 2:
        return None
    cell_size = map_model.native_units_per_cell
    start_index = min(range(len(route)), key=lambda i: math.hypot(route[i].x - player_x, route[i].z - player_z))
    sub_route = route[start_index:]
    if len(sub_route) < 2:
        return None
    edge_infos = annotate_route_edges(map_model, sub_route)

    best = None
    safe_fallback = None
    any_fallback = None

    cumulative_heading_change = 0.0
    min_clearance_so_far = math.inf
    for i, info in enumerate(edge_infos):
        cumulative_heading_change += info.heading_change_radians
        min_clearance_so_far = min(min_clearance_so_far, info.robust_clearance_cells)
        state = sub_route[i + 1]
        any_fallback = (state.x, state.z)

        real_distance_cells = math.hypot(state.x - player_x, state.z - player_z) / cell_size
        within_budget = (
            cumulative_heading_change <= max_heading_change_radians
            and min_clearance_so_far >= min_robust_clearance_cells
        )
        if real_distance_cells >= min_progress_cells:
            direct_clearance = _direct_hop_min_clearance(map_model, player_x, player_z, state.x, state.z)
            if direct_clearance >= DESIRED_CLEARANCE_CELLS:
                if within_budget:
                    best = (state.x, state.z)
                elif safe_fallback is None:
                    safe_fallback = (state.x, state.z)
        if not within_budget:
            break
    if best is not None:
        return best
    if safe_fallback is not None:
        return safe_fallback
    return any_fallback


def _spot_check_old_selector_fidelity() -> None:
    """Cross-validate select_persistent_waypoint_old against the preserved
    pre-patch RTL8 audit trace before trusting it for the full pool."""
    from scratchpad_beginner_navigation_mix_pools import DEV_POOL_SPEC_SEED, _reconstruct_two_wall_world, load_manifest as _load

    preserved = json.loads((ROOT / "evaluations" / "audit_selector_fallback_pre_selector_patch.json").read_text(encoding="utf-8"))
    rtl8_tick2 = next(t for t in preserved["RTL8"]["results"]["trained"]["ticks"] if t["tick"] == 2)
    expected_output = tuple(rtl8_tick2["selector_output"])
    expected_tier = rtl8_tick2["selector_fired_tier"]

    manifest = _load(ROOT / "evaluations" / f"router_mix_dev_pool_{DEV_POOL_SPEC_SEED}_manifest.json")
    episode = manifest["strata"]["two_wall_right_then_left"]["accepted"][8]
    map_model, world, final_native, initial_heading = _reconstruct_two_wall_world(episode)
    from simulator.kinodynamic_route_planner import plan_route
    route = plan_route(map_model, start_x=0.0, start_z=0.0, start_heading=initial_heading,
                        destination_x=final_native[0], destination_z=final_native[1])
    pre_pose = tuple(rtl8_tick2["pre_pose_xzh_deg"])
    got = select_persistent_waypoint_old(
        map_model, route, player_x=pre_pose[0], player_z=pre_pose[1], heading=math.radians(pre_pose[2]),
    )
    got_rounded = (round(got[0], 3), round(got[1], 3))
    assert got_rounded == expected_output, (
        f"select_persistent_waypoint_old fidelity check FAILED: expected {expected_output} "
        f"(tier={expected_tier}), got {got_rounded} -- do not trust condition A results"
    )
    print(f"select_persistent_waypoint_old fidelity check PASSED (RTL8 tick 2: {got_rounded} == {expected_output})")


def eval_obstacle_with_selector(model, manifest: dict, *, use_old_selector: bool, episode_seed_base: int) -> dict:
    """Always explicitly monkeypatches one of the two NAMED functions
    (never relies on whatever `select_persistent_waypoint` currently
    resolves to in production) so this script's A/B/C meaning is stable
    regardless of which algorithm is the current production default."""
    original = gre.select_persistent_waypoint
    gre.select_persistent_waypoint = select_persistent_waypoint_old if use_old_selector else select_persistent_waypoint_experimental_collision_free_fallback
    try:
        return eval_obstacle_manifest(model, manifest, episode_seed_base=episode_seed_base)
    finally:
        gre.select_persistent_waypoint = original


def main() -> None:
    _spot_check_old_selector_fidelity()

    manifest = load_manifest(ROOT / "evaluations" / f"router_mix_qualification_pool_{QUALIFICATION_SPEC_SEED}_manifest.json")
    baseline_model = PPO.load(str(BASELINE_CHECKPOINT), device="cpu")
    trained_model = PPO.load(str(TRAINED_CHECKPOINT), device="cpu")

    systems = {
        "A_baseline_old_selector": (baseline_model, True),
        "B_baseline_patched_selector": (baseline_model, False),
        "C_trained_patched_selector": (trained_model, False),
    }

    results = {}
    for name, (model, use_old) in systems.items():
        print(f"\n{'=' * 90}\nEvaluating {name} on 830M ({'OLD' if use_old else 'PATCHED'} selector)\n{'=' * 90}")
        obstacle = eval_obstacle_with_selector(model, manifest, use_old_selector=use_old, episode_seed_base=DEV_POOL_EPISODE_SEED_BASE)
        open_regression = eval_held_out(model, name, deterministic=True)
        results[name] = {"obstacle": obstacle, "open_regression_777M": open_regression}
        s = obstacle["combined_summary"]
        print(f"  obstacle: n={obstacle['n_total']} success={s['success_rate']:.4f} collision={s['collision_rate']:.4f} "
              f"timeout={s['timeout_rate']} planner_fail={s['planner_failure_rate']} path_eff={s['mean_path_efficiency']}")
        print(f"  collisions ({len(obstacle['collision_episode_keys'])}): {obstacle['collision_episode_keys']}")
        print(f"  open_regression(777M): success={open_regression['success_rate']:.4f} collision={open_regression['collision_rate']:.4f}")

    # -- predeclared decision rules, applied mechanically --
    a_coll = set(results["A_baseline_old_selector"]["obstacle"]["collision_episode_keys"])
    b_coll = set(results["B_baseline_patched_selector"]["obstacle"]["collision_episode_keys"])
    c_coll = set(results["C_trained_patched_selector"]["obstacle"]["collision_episode_keys"])
    a_success = round(results["A_baseline_old_selector"]["obstacle"]["combined_summary"]["success_rate"] * results["A_baseline_old_selector"]["obstacle"]["n_total"])
    b_success = round(results["B_baseline_patched_selector"]["obstacle"]["combined_summary"]["success_rate"] * results["B_baseline_patched_selector"]["obstacle"]["n_total"])
    c_success = round(results["C_trained_patched_selector"]["obstacle"]["combined_summary"]["success_rate"] * results["C_trained_patched_selector"]["obstacle"]["n_total"])
    a_open_ok = results["A_baseline_old_selector"]["open_regression_777M"]["success_rate"] == 1.0 and results["A_baseline_old_selector"]["open_regression_777M"]["collision_rate"] == 0.0
    b_open_ok = results["B_baseline_patched_selector"]["open_regression_777M"]["success_rate"] == 1.0 and results["B_baseline_patched_selector"]["open_regression_777M"]["collision_rate"] == 0.0
    c_open_ok = results["C_trained_patched_selector"]["open_regression_777M"]["success_rate"] == 1.0 and results["C_trained_patched_selector"]["open_regression_777M"]["collision_rate"] == 0.0

    b_no_new_collisions = b_coll.issubset(a_coll)
    b_success_ge_a = b_success >= a_success
    b_strict_improvement = (len(a_coll) == 0) or (b_coll < a_coll)
    b_qualifies_over_a = b_open_ok and b_no_new_collisions and b_success_ge_a and b_strict_improvement

    c_strict_subset_of_b = c_coll < b_coll
    c_success_ge_b = c_success >= b_success
    c_materially_beats_b = c_open_ok and c_strict_subset_of_b and c_success_ge_b
    selected = "C_trained_patched_selector" if c_materially_beats_b else "B_baseline_patched_selector"

    print(f"\n{'=' * 90}\nPREDECLARED DECISION RULES, APPLIED\n{'=' * 90}")
    print(f"A: collisions={sorted(a_coll)} success={a_success} open_ok={a_open_ok}")
    print(f"B: collisions={sorted(b_coll)} success={b_success} open_ok={b_open_ok}")
    print(f"C: collisions={sorted(c_coll)} success={c_success} open_ok={c_open_ok}")
    print(f"\nB qualifies over A: {b_qualifies_over_a} "
          f"(no_new_collisions={b_no_new_collisions}, success_ge_a={b_success_ge_a}, strict_improvement={b_strict_improvement}, open_ok={b_open_ok})")
    if not b_qualifies_over_a:
        print("  *** B DOES NOT QUALIFY -- the router patch is NOT validated on this fresh pool. STOP. Do not proceed to select C or touch 820M. ***")
    print(f"\nC materially beats B: {c_materially_beats_b} "
          f"(strict_subset_of_b={c_strict_subset_of_b}, success_ge_b={c_success_ge_b}, open_ok={c_open_ok})")
    print(f"\nSELECTED SYSTEM: {selected if b_qualifies_over_a else 'NONE -- B failed to qualify over A'}")

    output = {
        "qualification_spec_seed": QUALIFICATION_SPEC_SEED,
        "results": results,
        "rule_evaluation": {
            "a_collisions": sorted(a_coll), "b_collisions": sorted(b_coll), "c_collisions": sorted(c_coll),
            "a_success": a_success, "b_success": b_success, "c_success": c_success,
            "b_qualifies_over_a": b_qualifies_over_a, "c_materially_beats_b": c_materially_beats_b,
            "selected": selected if b_qualifies_over_a else None,
        },
    }
    out_path = ROOT / "evaluations" / "router_patch_qualification_compare_result.json"
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
