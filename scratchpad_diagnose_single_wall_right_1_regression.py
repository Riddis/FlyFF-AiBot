"""2026-08-14: diagnose the ONE new collision the router patch introduced
on the fresh 830M qualification pool (scratchpad_router_patch_
qualification_compare.py's condition-A/B comparison): `single_wall_right`
index 1, collision under B (frozen 0051200 + PATCHED selector) that did
NOT occur under A (frozen 0051200 + OLD selector) -- the predeclared
B-qualifies-over-A rule's "no new collision relative to A" gate correctly
FAILED on this pool (B repaired A's 4 collisions but introduced this one),
so the router patch does not qualify as-is. This script is diagnosis
only -- no router/controller code is touched, per explicit user
instruction not to make any further router changes right now.

AUDIT ONLY, same discipline as scratchpad_audit_selector_fallback.py:
both the OLD-tier and PATCHED-tier instrumented replicas below are
cross-validated every tick against their real counterparts (select_
persistent_waypoint_old / select_persistent_waypoint) before being
trusted for logging.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from scratchpad_beginner_navigation_mix_pools import load_manifest
from scratchpad_router_patch_qualification_pool import QUALIFICATION_SPEC_SEED
from scratchpad_router_patch_qualification_compare import DEV_POOL_EPISODE_SEED_BASE, select_persistent_waypoint_old
from simulator.environment import RecordedFarmingEnv
from simulator.kinodynamic_route_planner import (
    DESIRED_CLEARANCE_CELLS, TargetPersistenceController, _direct_hop_min_clearance, _segment_clear,
    annotate_route_edges, plan_route, select_persistent_waypoint_experimental_collision_free_fallback,
)
from simulator.navigation_history import NavigationHistoryWrapper
from simulator.static_waypoint_env import SUCCESS_RADIUS_CELLS

ROOT = Path(__file__).resolve().parent
BASELINE_CHECKPOINT = ROOT / "models" / "generalized_waypoint_both_seed2_0051200.zip"
EPISODE_STEPS = 200
STEERING_NAMES = {0: "STRAIGHT", 1: "LEFT", 2: "RIGHT"}

STRATUM = "single_wall_right"
INDEX = 1


def _seed_counter_for(manifest: dict, stratum_name: str, index: int) -> int:
    counter = 0
    for name, stratum in manifest["strata"].items():
        if name == "open":
            continue
        if name == stratum_name:
            return counter + index
        counter += len(stratum["accepted"])
    raise KeyError(stratum_name)


def _instrumented_select_old(
    map_model, route, *, player_x, player_z, heading,
    max_heading_change_radians=math.radians(75.0), min_progress_cells=2.0, min_robust_clearance_cells=2.0,
):
    """Instrumented replica of the PRE-PATCH 3-tier algorithm (best/
    safe_fallback/any_fallback only), mirroring select_persistent_
    waypoint_old exactly. Cross-validated against it every tick by the
    caller."""
    if len(route) < 2:
        return None, [], "NONE_route_too_short"
    cell_size = map_model.native_units_per_cell
    start_index = min(range(len(route)), key=lambda i: math.hypot(route[i].x - player_x, route[i].z - player_z))
    sub_route = route[start_index:]
    if len(sub_route) < 2:
        return None, [], "NONE_sub_route_too_short"
    edge_infos = annotate_route_edges(map_model, sub_route)

    best = None
    safe_fallback = None
    any_fallback = None
    candidate_log = []

    cumulative_heading_change = 0.0
    min_clearance_so_far = math.inf
    for i, info in enumerate(edge_infos):
        cumulative_heading_change += info.heading_change_radians
        min_clearance_so_far = min(min_clearance_so_far, info.robust_clearance_cells)
        state = sub_route[i + 1]
        route_index = start_index + i + 1
        any_fallback = (state.x, state.z)

        real_distance_cells = math.hypot(state.x - player_x, state.z - player_z) / cell_size
        within_budget = (
            cumulative_heading_change <= max_heading_change_radians
            and min_clearance_so_far >= min_robust_clearance_cells
        )
        meets_min_progress = real_distance_cells >= min_progress_cells
        direct_clearance = _direct_hop_min_clearance(map_model, player_x, player_z, state.x, state.z)
        segment_clear = _segment_clear(map_model, player_x, player_z, state.x, state.z)

        eligible_best = False
        eligible_safe_fallback = False
        if meets_min_progress and direct_clearance >= DESIRED_CLEARANCE_CELLS:
            if within_budget:
                eligible_best = True
                best = (state.x, state.z)
            elif safe_fallback is None:
                eligible_safe_fallback = True
                safe_fallback = (state.x, state.z)

        candidate_log.append({
            "route_index": route_index, "real_distance_cells": round(real_distance_cells, 3),
            "direct_hop_min_clearance": round(direct_clearance, 3), "segment_clear": segment_clear,
            "within_budget": within_budget, "meets_min_progress": meets_min_progress,
            "eligible_best": eligible_best, "eligible_safe_fallback": eligible_safe_fallback,
        })
        if not within_budget:
            break

    if best is not None:
        return best, candidate_log, "BEST"
    if safe_fallback is not None:
        return safe_fallback, candidate_log, "SAFE_FALLBACK"
    return any_fallback, candidate_log, "ANY_FALLBACK"


def _instrumented_select_patched(
    map_model, route, *, player_x, player_z, heading,
    max_heading_change_radians=math.radians(75.0), min_progress_cells=2.0, min_robust_clearance_cells=2.0,
):
    """Instrumented replica of the CURRENT (patched) 4-tier algorithm.
    Cross-validated against select_persistent_waypoint every tick."""
    if len(route) < 2:
        return None, [], "NONE_route_too_short"
    cell_size = map_model.native_units_per_cell
    start_index = min(range(len(route)), key=lambda i: math.hypot(route[i].x - player_x, route[i].z - player_z))
    sub_route = route[start_index:]
    if len(sub_route) < 2:
        return None, [], "NONE_sub_route_too_short"
    edge_infos = annotate_route_edges(map_model, sub_route)

    best = None
    safe_fallback = None
    collision_free_fallback = None
    collision_free_fallback_key = None
    any_fallback = None
    candidate_log = []

    cumulative_heading_change = 0.0
    min_clearance_so_far = math.inf
    for i, info in enumerate(edge_infos):
        cumulative_heading_change += info.heading_change_radians
        min_clearance_so_far = min(min_clearance_so_far, info.robust_clearance_cells)
        state = sub_route[i + 1]
        route_index = start_index + i + 1
        any_fallback = (state.x, state.z)

        real_distance_cells = math.hypot(state.x - player_x, state.z - player_z) / cell_size
        within_budget = (
            cumulative_heading_change <= max_heading_change_radians
            and min_clearance_so_far >= min_robust_clearance_cells
        )
        meets_min_progress = real_distance_cells >= min_progress_cells
        direct_clearance = _direct_hop_min_clearance(map_model, player_x, player_z, state.x, state.z)
        segment_clear = _segment_clear(map_model, player_x, player_z, state.x, state.z)

        eligible_best = False
        eligible_safe_fallback = False
        eligible_collision_free_fallback = False
        if meets_min_progress and direct_clearance >= DESIRED_CLEARANCE_CELLS:
            if within_budget:
                eligible_best = True
                best = (state.x, state.z)
            elif safe_fallback is None:
                eligible_safe_fallback = True
                safe_fallback = (state.x, state.z)
        elif meets_min_progress and within_budget and segment_clear:
            eligible_collision_free_fallback = True
            key = (direct_clearance, real_distance_cells)
            if collision_free_fallback_key is None or key > collision_free_fallback_key:
                collision_free_fallback = (state.x, state.z)
                collision_free_fallback_key = key

        candidate_log.append({
            "route_index": route_index, "real_distance_cells": round(real_distance_cells, 3),
            "direct_hop_min_clearance": round(direct_clearance, 3), "segment_clear": segment_clear,
            "within_budget": within_budget, "meets_min_progress": meets_min_progress,
            "eligible_best": eligible_best, "eligible_safe_fallback": eligible_safe_fallback,
            "eligible_collision_free_fallback": eligible_collision_free_fallback,
        })
        if not within_budget:
            break

    if best is not None:
        return best, candidate_log, "BEST"
    if safe_fallback is not None:
        return safe_fallback, candidate_log, "SAFE_FALLBACK"
    if collision_free_fallback is not None:
        return collision_free_fallback, candidate_log, "COLLISION_FREE_LOW_MARGIN_FALLBACK"
    return any_fallback, candidate_log, "ANY_FALLBACK"


def run_traced(model, map_model, world, *, initial_heading, final_native, episode_seed, use_old_selector: bool, label: str):
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

    ticks = []
    prev_contacts = 0
    outcome = "timeout"

    for tick in range(EPISODE_STEPS):
        pre_x, pre_z, pre_heading = base_env.player_x, base_env.player_z, base_env.heading

        if use_old_selector:
            real_candidate = select_persistent_waypoint_old(map_model, route, player_x=pre_x, player_z=pre_z, heading=pre_heading)
            replica_candidate, candidate_log, fired_tier = _instrumented_select_old(map_model, route, player_x=pre_x, player_z=pre_z, heading=pre_heading)
        else:
            real_candidate = select_persistent_waypoint_experimental_collision_free_fallback(map_model, route, player_x=pre_x, player_z=pre_z, heading=pre_heading)
            replica_candidate, candidate_log, fired_tier = _instrumented_select_patched(map_model, route, player_x=pre_x, player_z=pre_z, heading=pre_heading)

        if real_candidate is None:
            assert replica_candidate is None or fired_tier.startswith("NONE"), (
                f"{label} tick {tick}: replica returned {replica_candidate} but real function returned None"
            )
        else:
            assert replica_candidate == real_candidate, (
                f"{label} tick {tick}: REPLICA DIVERGED -- real={real_candidate} replica={replica_candidate}"
            )

        candidate_for_use = real_candidate if real_candidate is not None else final_native
        target = controller.update(candidate_for_use, player_x=pre_x, player_z=pre_z, route=route)
        reason = controller.last_switch_reason.value if controller.last_switch_reason else None

        base_env.actors[0].x, base_env.actors[0].z = target
        obs = env._augment(base_env._observation(), base_env.previous_steering)

        action, _state = model.predict(obs, deterministic=True)
        a = int(action[0])
        action_arr = np.asarray(action, dtype=np.int64).copy()
        action_arr[1] = 0
        obs, _reward, term, trunc, info = env.step(action_arr)

        post_x, post_z, post_heading = base_env.player_x, base_env.player_z, base_env.heading
        contacts = int(info.get("contacts", 0))
        contact_this_tick = contacts > prev_contacts
        prev_contacts = contacts
        final_distance = math.hypot(final_native[0] - post_x, final_native[1] - post_z) / cell_size

        ticks.append({
            "tick": tick, "pre_pose": (round(pre_x, 3), round(pre_z, 3), round(math.degrees(pre_heading), 2)),
            "selector_output": (round(real_candidate[0], 3), round(real_candidate[1], 3)) if real_candidate else "NONE",
            "fired_tier": fired_tier, "n_candidates": len(candidate_log), "candidates": candidate_log,
            "controller_target": (round(target[0], 3), round(target[1], 3)), "controller_reason": reason,
            "steering": STEERING_NAMES[a], "post_pose": (round(post_x, 3), round(post_z, 3), round(math.degrees(post_heading), 2)),
            "final_distance_cells": round(final_distance, 3), "contact": contact_this_tick,
        })

        if contact_this_tick:
            outcome = "collision"
            break
        if final_distance <= SUCCESS_RADIUS_CELLS:
            outcome = "success"
            break

    env.close()
    return outcome, ticks


def main() -> None:
    baseline_model = PPO.load(str(BASELINE_CHECKPOINT), device="cpu")
    manifest = load_manifest(ROOT / "evaluations" / f"router_mix_qualification_pool_{QUALIFICATION_SPEC_SEED}_manifest.json")
    counter = _seed_counter_for(manifest, STRATUM, INDEX)
    episode_seed = DEV_POOL_EPISODE_SEED_BASE + counter
    episode = manifest["strata"][STRATUM]["accepted"][INDEX]
    print(f"{STRATUM}[{INDEX}] seed={episode_seed} spec={episode}")

    from scratchpad_beginner_navigation_mix_pools import _reconstruct_single_wall_world
    map_model, world, final_native, initial_heading = _reconstruct_single_wall_world(episode)

    all_results: dict = {"stratum": STRATUM, "index": INDEX, "seed": episode_seed, "spec": episode, "traces": {}}
    for use_old, label in [(True, "A_old_selector"), (False, "B_patched_selector")]:
        outcome, ticks = run_traced(
            baseline_model, map_model, world, initial_heading=initial_heading, final_native=final_native,
            episode_seed=episode_seed, use_old_selector=use_old, label=label,
        )
        all_results["traces"][label] = {"outcome": outcome, "ticks": ticks}
        print(f"\n{'=' * 100}\n{label}: outcome={outcome} ticks_run={len(ticks)}\n{'=' * 100}")
        for row in ticks:
            print(f"  tick={row['tick']:3d} steer={row['steering']:8s} tier={row['fired_tier']:32s} "
                  f"n_cand={row['n_candidates']:2d} selector_out={row['selector_output']} "
                  f"controller_target={row['controller_target']} reason={row['controller_reason']:24s} "
                  f"dist={row['final_distance_cells']:.2f} contact={row['contact']}")
        if outcome == "collision":
            last = ticks[-1]
            print(f"\n  Contact tick candidates (fired_tier={last['fired_tier']}):")
            for c in last["candidates"]:
                print(f"    {c}")

    out_path = ROOT / "evaluations" / "diagnose_single_wall_right_1_regression.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved full old-vs-patched trace to {out_path}")


if __name__ == "__main__":
    main()
