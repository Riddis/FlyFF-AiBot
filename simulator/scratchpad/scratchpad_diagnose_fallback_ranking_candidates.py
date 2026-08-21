"""2026-08-15: focused follow-up diagnostic, per explicit user instruction
after the 830M A/B/C comparison found the router patch is NOT qualified
out-of-sample (repairs A's 4 collisions but introduces a new one,
single_wall_right[1]). AUDIT ONLY -- select_persistent_waypoint() is not
touched, nothing is trained, 820M is not evaluated.

Two questions, for the single decisive tick in each of 5 episodes (the
first tick where the OLD selector fires ANY_FALLBACK -- the only tier
where old and patched CAN diverge, since BEST/SAFE_FALLBACK are
unchanged by the patch):

  1. Is the OLD selector's actually-returned any_fallback target itself
     _segment_clear=True at that tick? (Established directly, not
     inferred -- this is the fact the user needs before deciding whether
     restoring "prefer distance" behavior would just re-introduce the
     RTL8/RTL9-style invalid-hop defect.)
  2. Among the SAME candidate set, does "farthest _segment_clear=True
     candidate" (the proposed alternative ranking) coincide with "max
     direct_hop_min_clearance _segment_clear=True candidate" (the current
     patch's ranking)? If they coincide for all 4 B-repaired episodes and
     differ (favoring the farther one) at single_wall_right[1], that is
     direct evidence the max-clearance-first rule is what shortened the
     waypoint at the one episode where it hurt.

Trajectory basis: for each episode, ONE trace is walked, driven by the
OLD selector's real output (matching what condition A actually executed
and what is recorded as A's outcome). At every tick, the PATCHED tier is
ALSO evaluated from the identical pre-tick pose (hypothetically -- it
does not drive movement in this trace) so old vs. patched are compared
from the exact same state, not confounded by trajectory drift. This is
valid only up to and including the first divergence tick; ticks after
that are not used for the candidate-table comparison (the two systems'
real trajectories separate at that point, which is exactly why 830M
had to actually run both to get real outcomes -- already done in
scratchpad_router_patch_qualification_compare.py).

Both instrumented replicas are cross-validated against their real
counterparts (select_persistent_waypoint_old / select_persistent_
waypoint) every tick, same discipline as every other audit script in
this investigation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from .scratchpad_beginner_navigation_mix_pools import _reconstruct_single_wall_world, _reconstruct_two_wall_world, load_manifest
from .scratchpad_diagnose_single_wall_right_1_regression import _instrumented_select_old, _instrumented_select_patched, _seed_counter_for
from .scratchpad_router_patch_qualification_compare import DEV_POOL_EPISODE_SEED_BASE, select_persistent_waypoint_old
from .scratchpad_router_patch_qualification_pool import QUALIFICATION_SPEC_SEED
from simulator.environment import RecordedFarmingEnv
from simulator.kinodynamic_route_planner import (
    TargetPersistenceController, plan_route, select_persistent_waypoint_experimental_collision_free_fallback,
)
from simulator.navigation_history import NavigationHistoryWrapper
from simulator.static_waypoint_env import SUCCESS_RADIUS_CELLS

ROOT = Path(__file__).resolve().parents[2]
BASELINE_CHECKPOINT = ROOT / "models" / "generalized_waypoint_both_seed2_0051200.zip"
EPISODE_STEPS = 200
STEERING_NAMES = {0: "STRAIGHT", 1: "LEFT", 2: "RIGHT"}

# (label, stratum, index, real_A_outcome, real_B_outcome) -- real outcomes
# taken directly from evaluations/router_patch_qualification_compare_result.json,
# not re-derived here.
CASES = [
    ("single_wall_left_14", "single_wall_left", 14, "collision", "success"),
    ("single_wall_left_21", "single_wall_left", 21, "collision", "success"),
    ("two_wall_right_then_left_16", "two_wall_right_then_left", 16, "collision", "success"),
    ("two_wall_right_then_left_5", "two_wall_right_then_left", 5, "collision", "success"),
    ("single_wall_right_1", "single_wall_right", 1, "success", "collision"),
]


def _reconstruct(stratum: str, episode: dict):
    if stratum.startswith("single_wall_"):
        return _reconstruct_single_wall_world(episode)
    return _reconstruct_two_wall_world(episode)


def _farthest_segment_clear(candidates: list[dict]) -> dict | None:
    clear = [c for c in candidates if c["segment_clear"]]
    return clear[-1] if clear else None  # candidates are in route order (nearest -> farthest)


def _max_clearance_segment_clear(candidates: list[dict]) -> dict | None:
    clear = [c for c in candidates if c["segment_clear"]]
    if not clear:
        return None
    return max(clear, key=lambda c: (c["direct_hop_min_clearance"], c["real_distance_cells"]))


def run_case(model, label: str, stratum: str, episode: dict, episode_seed: int, real_a: str, real_b: str) -> dict:
    map_model, world, final_native, initial_heading = _reconstruct(stratum, episode)
    raw_env = RecordedFarmingEnv(world, map_model=map_model, episode_steps=EPISODE_STEPS)
    env = NavigationHistoryWrapper(raw_env)
    obs, _info = env.reset(seed=episode_seed)
    base_env = env.unwrapped
    for actor in base_env.actors[1:]:
        actor.alive = False
    base_env.heading = initial_heading
    cell_size = base_env.map.native_units_per_cell
    base_env.actors[0].x, base_env.actors[0].z = final_native
    base_env.actors[0].alive = True

    route = plan_route(
        map_model, start_x=base_env.player_x, start_z=base_env.player_z, start_heading=base_env.heading,
        destination_x=final_native[0], destination_z=final_native[1],
    )
    assert len(route) >= 2, f"{label}: no route found"
    controller = TargetPersistenceController(map_model, final_native[0], final_native[1])

    critical_tick_record = None
    prev_contacts = 0

    for tick in range(EPISODE_STEPS):
        pre_x, pre_z, pre_heading = base_env.player_x, base_env.player_z, base_env.heading

        old_real = select_persistent_waypoint_old(map_model, route, player_x=pre_x, player_z=pre_z, heading=pre_heading)
        old_replica, old_candidates, old_tier = _instrumented_select_old(map_model, route, player_x=pre_x, player_z=pre_z, heading=pre_heading)
        assert (old_real is None and old_tier.startswith("NONE")) or old_replica == old_real, (
            f"{label} tick {tick}: OLD replica diverged: real={old_real} replica={old_replica}"
        )

        patched_real = select_persistent_waypoint_experimental_collision_free_fallback(map_model, route, player_x=pre_x, player_z=pre_z, heading=pre_heading)
        patched_replica, patched_candidates, patched_tier = _instrumented_select_patched(map_model, route, player_x=pre_x, player_z=pre_z, heading=pre_heading)
        assert (patched_real is None and patched_tier.startswith("NONE")) or patched_replica == patched_real, (
            f"{label} tick {tick}: PATCHED replica diverged: real={patched_real} replica={patched_replica}"
        )

        if critical_tick_record is None and old_tier == "ANY_FALLBACK":
            old_target = old_real
            patched_target = patched_real
            farthest_clear = _farthest_segment_clear(old_candidates)
            max_clear = _max_clearance_segment_clear(old_candidates)
            # any_fallback is unconditionally set to the LAST candidate examined
            # every iteration (see select_persistent_waypoint's docstring), so
            # when old_tier == "ANY_FALLBACK" the old target IS old_candidates[-1]
            # -- no fuzzy distance-matching needed.
            old_target_candidate = old_candidates[-1]
            assert math.isclose(old_target_candidate["real_distance_cells"], math.hypot(old_target[0] - pre_x, old_target[1] - pre_z) / cell_size, abs_tol=0.05), (
                f"{label} tick {tick}: old_candidates[-1] does not match old any_fallback target -- indexing assumption wrong"
            )
            critical_tick_record = {
                "tick": tick,
                "pre_pose": (round(pre_x, 3), round(pre_z, 3), round(math.degrees(pre_heading), 2)),
                "old_tier": old_tier, "old_target": (round(old_target[0], 3), round(old_target[1], 3)),
                "old_target_segment_clear": old_target_candidate["segment_clear"] if old_target_candidate else None,
                "old_target_direct_hop_clearance": old_target_candidate["direct_hop_min_clearance"] if old_target_candidate else None,
                "patched_tier": patched_tier, "patched_target": (round(patched_target[0], 3), round(patched_target[1], 3)) if patched_target else None,
                "farthest_segment_clear_true_candidate": farthest_clear,
                "max_clearance_segment_clear_true_candidate": max_clear,
                "farthest_equals_max_clearance": farthest_clear == max_clear,
                "farthest_equals_old_any_fallback_target": (
                    farthest_clear is not None and old_target_candidate is not None and farthest_clear["route_index"] == old_target_candidate["route_index"]
                ),
                "all_candidates_old_view": old_candidates,
            }

        candidate_for_use = old_real if old_real is not None else final_native
        target = controller.update(candidate_for_use, player_x=pre_x, player_z=pre_z, route=route)

        base_env.actors[0].x, base_env.actors[0].z = target
        obs = env._augment(base_env._observation(), base_env.previous_steering)
        action, _state = model.predict(obs, deterministic=True)
        action_arr = np.asarray(action, dtype=np.int64).copy()
        action_arr[1] = 0
        obs, _reward, term, trunc, info = env.step(action_arr)

        contacts = int(info.get("contacts", 0))
        if contacts > prev_contacts:
            break
        prev_contacts = contacts
        final_distance = math.hypot(final_native[0] - base_env.player_x, final_native[1] - base_env.player_z) / cell_size
        if final_distance <= SUCCESS_RADIUS_CELLS:
            break

    env.close()
    return {
        "label": label, "stratum": stratum, "seed": episode_seed,
        "real_outcome_A_old_selector": real_a, "real_outcome_B_patched_selector": real_b,
        "critical_tick": critical_tick_record,
    }


def main() -> None:
    baseline_model = PPO.load(str(BASELINE_CHECKPOINT), device="cpu")
    manifest = load_manifest(ROOT / "simulator" / "evaluations" / f"router_mix_qualification_pool_{QUALIFICATION_SPEC_SEED}_manifest.json")

    all_results = {}
    for label, stratum, index, real_a, real_b in CASES:
        counter = _seed_counter_for(manifest, stratum, index)
        episode_seed = DEV_POOL_EPISODE_SEED_BASE + counter
        episode = manifest["strata"][stratum]["accepted"][index]
        result = run_case(baseline_model, label, stratum, episode, episode_seed, real_a, real_b)
        all_results[label] = result

        c = result["critical_tick"]
        print(f"\n{'=' * 100}\n{label} (A={real_a}, B={real_b}) -- critical tick", "NONE (old never hit ANY_FALLBACK)" if c is None else c["tick"])
        if c is not None:
            print(f"  old_tier={c['old_tier']} old_target={c['old_target']} "
                  f"segment_clear={c['old_target_segment_clear']} direct_hop_clearance={c['old_target_direct_hop_clearance']}")
            print(f"  patched_tier={c['patched_tier']} patched_target={c['patched_target']}")
            print(f"  farthest_segment_clear_true: {c['farthest_segment_clear_true_candidate']}")
            print(f"  max_clearance_segment_clear_true: {c['max_clearance_segment_clear_true_candidate']}")
            print(f"  farthest == max_clearance candidate: {c['farthest_equals_max_clearance']}")
            print(f"  farthest == old any_fallback target: {c['farthest_equals_old_any_fallback_target']}")

    out_path = ROOT / "simulator" / "evaluations" / "diagnose_fallback_ranking_candidates.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
