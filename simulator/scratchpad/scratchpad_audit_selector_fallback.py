"""2026-08-14: selector fallback audit, per explicit user instruction after
the LTR/RTL contrastive diagnostic. AUDIT ONLY -- no router/controller code
is modified, no training happens. `select_persistent_waypoint()` itself
(`simulator/kinodynamic_route_planner.py`) is called UNMODIFIED at every
tick to determine the real target sequence; a separate, faithful
INSTRUMENTED REPLICA of its exact algorithm (`_instrumented_select`,
below) is called alongside it purely for logging, and is cross-validated
every single tick to return the IDENTICAL coordinate the real function
returns -- if they ever diverge, that's a bug in the replica, not evidence
about the selector.

Question this answers, per the three cases the user specified:
  Case A: `any_fallback` fires despite a nearer, safer forward candidate
    existing (a selector-ranking/fallback defect -- fix the selector).
  Case B: no safer forward candidate exists from wherever the trajectory
    is by that tick (the selector did the best it could; look upstream at
    why the trajectory reached that state).
  Case C: the actually-selected hop is `_segment_clear=False` -- the
    router is knowingly handing PPO an invalid shortcut through its
    absolute fallback tier (a routing-layer safety issue, not a
    navigator-training deficiency).

Canonical 812M episode seeds: derived from `eval_obstacle_manifest`'s own
stratum-iteration order (never hand-computed), using `812_600_000 +
seed_counter` -- empirically verified (not assumed) to produce IDENTICAL
outcomes to the `812_500_000` base used for the original baseline eval,
for all four episodes audited here, before relying on this as "the"
canonical seed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from stable_baselines3 import PPO

from .scratchpad_beginner_navigation_mix_pools import DEV_POOL_SPEC_SEED, _reconstruct_two_wall_world, load_manifest
from simulator.environment import RecordedFarmingEnv
from navigation.kinodynamic_route_planner import (
    DESIRED_CLEARANCE_CELLS, TargetPersistenceController, _direct_hop_min_clearance, _segment_clear,
    annotate_route_edges, plan_route, select_persistent_waypoint,
)
from simulator.navigation_history import NavigationHistoryWrapper
from simulator.static_waypoint_env import SUCCESS_RADIUS_CELLS

ROOT = Path(__file__).resolve().parents[2]
BASELINE_CHECKPOINT = ROOT / "models" / "generalized_waypoint_both_seed2_0051200.zip"
TRAINED_CHECKPOINT = ROOT / "models" / "generalized_waypoint_router_mix_seed100_0056320.zip"
EPISODE_STEPS = 200
CANONICAL_SEED_BASE = 812_600_000
STEERING_NAMES = {0: "STRAIGHT", 1: "LEFT", 2: "RIGHT"}

# (label, stratum, index, which models to trace)
CASES = [
    ("RTL8", "two_wall_right_then_left", 8, ["trained"]),      # baseline == trained trajectory (proven identical earlier) -- trace trained only
    ("RTL9", "two_wall_right_then_left", 9, ["trained"]),
    ("RTL27", "two_wall_right_then_left", 27, ["trained"]),    # successful near-neighbor comparator
    ("LTR26", "two_wall_left_then_right", 26, ["baseline", "trained"]),  # trajectories DIVERGE -- trace both
]


def _seed_counter_for(manifest: dict, stratum_name: str, index: int) -> int:
    counter = 0
    for name, stratum in manifest["strata"].items():
        if name == "open":
            continue
        if name == stratum_name:
            return counter + index
        counter += len(stratum["accepted"])
    raise KeyError(stratum_name)


def _instrumented_select(
    map_model, route, *, player_x, player_z, heading,
    max_heading_change_radians=math.radians(75.0), min_progress_cells=2.0, min_robust_clearance_cells=2.0,
):
    """Faithful, read-only replica of select_persistent_waypoint()'s exact
    algorithm (simulator/kinodynamic_route_planner.py) -- same helper
    functions, same tier logic, same early-break -- with full per-
    candidate instrumentation. Returns (result, candidate_log, fired_tier).
    Cross-validated against the real function's return value by the
    caller every tick; never used to actually drive the episode."""
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
    fired_tier = None
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
        # Computed for every candidate regardless of short-circuiting, for
        # full audit visibility -- the real function only computes this
        # when meets_min_progress is True (see below for that distinction).
        direct_clearance = _direct_hop_min_clearance(map_model, player_x, player_z, state.x, state.z)
        segment_clear = _segment_clear(map_model, player_x, player_z, state.x, state.z)

        eligible_best = False
        eligible_safe_fallback = False
        eligible_collision_free_fallback = False
        if meets_min_progress and direct_clearance >= DESIRED_CLEARANCE_CELLS:
            if within_budget:
                # Matches the real function EXACTLY: `best` is unconditionally
                # overwritten on every eligible candidate (not just the first),
                # so it ends up as the LAST/farthest-along-route eligible
                # candidate before the loop breaks on !within_budget -- NOT
                # the first one. (Bug caught by the cross-validation assert
                # against the real function, see MISTAKES.md.)
                eligible_best = True
                best = (state.x, state.z)
            elif safe_fallback is None:
                eligible_safe_fallback = True
                safe_fallback = (state.x, state.z)
        elif meets_min_progress and within_budget and segment_clear:
            # 2026-08-14: mirrors the new COLLISION_FREE_LOW_MARGIN_FALLBACK
            # tier added to the real function after this audit's first run
            # proved the plain any_fallback tier was picking a segment_clear=
            # False candidate over a nearer, genuinely valid one.
            eligible_collision_free_fallback = True
            key = (direct_clearance, real_distance_cells)
            if collision_free_fallback_key is None or key > collision_free_fallback_key:
                collision_free_fallback = (state.x, state.z)
                collision_free_fallback_key = key

        candidate_log.append({
            "route_index": route_index, "start_index": start_index,
            "real_distance_cells": round(real_distance_cells, 3),
            "cumulative_heading_change_deg": round(math.degrees(cumulative_heading_change), 2),
            "route_edge_min_clearance_so_far": round(min_clearance_so_far, 3),
            "direct_hop_min_clearance": round(direct_clearance, 3),
            "segment_clear": segment_clear,
            "within_budget": within_budget,
            "meets_min_progress": meets_min_progress,
            "eligible_best": eligible_best,
            "eligible_safe_fallback": eligible_safe_fallback,
            "eligible_collision_free_fallback": eligible_collision_free_fallback,
            "is_any_fallback_if_loop_ends_here": True,
        })
        if not within_budget:
            break

    if best is not None:
        fired_tier = "BEST"
        result = best
    elif safe_fallback is not None:
        fired_tier = "SAFE_FALLBACK"
        result = safe_fallback
    elif collision_free_fallback is not None:
        fired_tier = "COLLISION_FREE_LOW_MARGIN_FALLBACK"
        result = collision_free_fallback
    else:
        fired_tier = "ANY_FALLBACK"
        result = any_fallback

    return result, candidate_log, fired_tier


def run_traced_with_selector_audit(model, map_model, world, *, initial_heading, final_native, episode_seed, label):
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

        real_candidate = select_persistent_waypoint(map_model, route, player_x=pre_x, player_z=pre_z, heading=pre_heading)
        replica_candidate, candidate_log, fired_tier = _instrumented_select(map_model, route, player_x=pre_x, player_z=pre_z, heading=pre_heading)

        # Cross-validate the replica against the real function every tick.
        if real_candidate is None:
            assert replica_candidate is None or fired_tier.startswith("NONE"), (
                f"{label} tick {tick}: replica returned {replica_candidate} but real function returned None"
            )
        else:
            assert replica_candidate == real_candidate, (
                f"{label} tick {tick}: REPLICA DIVERGED FROM REAL SELECTOR -- "
                f"real={real_candidate} replica={replica_candidate} -- audit invalid, fix _instrumented_select"
            )

        candidate_for_use = real_candidate if real_candidate is not None else final_native
        target = controller.update(candidate_for_use, player_x=pre_x, player_z=pre_z, route=route)
        reason = controller.last_switch_reason.value if controller.last_switch_reason else None
        target_is_selector_output = (target == candidate_for_use)

        base_env.actors[0].x, base_env.actors[0].z = target
        obs = env._augment(base_env._observation(), base_env.previous_steering)

        action, _state = model.predict(obs, deterministic=True)
        a = int(action[0])
        import numpy as np
        action_arr = np.asarray(action, dtype=np.int64).copy()
        action_arr[1] = 0
        obs, _reward, term, trunc, info = env.step(action_arr)

        post_x, post_z, post_heading = base_env.player_x, base_env.player_z, base_env.heading
        contacts = int(info.get("contacts", 0))
        contact_this_tick = contacts > prev_contacts
        prev_contacts = contacts
        final_distance = math.hypot(final_native[0] - post_x, final_native[1] - post_z) / cell_size

        ticks.append({
            "tick": tick,
            "pre_pose_xzh_deg": (round(pre_x, 3), round(pre_z, 3), round(math.degrees(pre_heading), 2)),
            "selector_output": (round(real_candidate[0], 3), round(real_candidate[1], 3)) if real_candidate else "NONE",
            "selector_fired_tier": fired_tier,
            "selector_candidates": candidate_log,
            "controller_held_target": (round(target[0], 3), round(target[1], 3)),
            "controller_reason": reason,
            "controller_target_equals_selector_output": target_is_selector_output,
            "steering_action": STEERING_NAMES[a],
            "post_pose_xzh_deg": (round(post_x, 3), round(post_z, 3), round(math.degrees(post_heading), 2)),
            "final_distance_cells": round(final_distance, 3),
            "contact_this_tick": contact_this_tick,
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
    trained_model = PPO.load(str(TRAINED_CHECKPOINT), device="cpu")
    manifest = load_manifest(ROOT / "simulator" / "evaluations" / f"router_mix_dev_pool_{DEV_POOL_SPEC_SEED}_manifest.json")

    all_results: dict[str, dict] = {}
    for label, stratum, index, model_roles in CASES:
        counter = _seed_counter_for(manifest, stratum, index)
        episode_seed = CANONICAL_SEED_BASE + counter
        episode = manifest["strata"][stratum]["accepted"][index]
        map_model, world, final_native, initial_heading = _reconstruct_two_wall_world(episode)
        print(f"\n{'=' * 100}\n{label}: stratum={stratum} index={index} canonical_seed={episode_seed}\n"
              f"spec={episode}\n{'=' * 100}")

        case_results = {}
        for role in model_roles:
            model = baseline_model if role == "baseline" else trained_model
            outcome, ticks = run_traced_with_selector_audit(
                model, map_model, world, initial_heading=initial_heading, final_native=final_native,
                episode_seed=episode_seed, label=f"{label}/{role}",
            )
            case_results[role] = {"outcome": outcome, "ticks": ticks}
            print(f"\n  --- {role}: outcome={outcome} ticks_run={len(ticks)} ---")
            for row in ticks:
                fired = row["selector_fired_tier"]
                n_cand = len(row["selector_candidates"])
                print(f"    tick={row['tick']:2d} steer={row['steering_action']:8s} "
                      f"selector_tier={fired:14s} n_candidates={n_cand:2d} "
                      f"controller_reason={row['controller_reason']:24s} "
                      f"final_dist={row['final_distance_cells']:.2f} contact={row['contact_this_tick']}")

        all_results[label] = {"stratum": stratum, "index": index, "seed": episode_seed, "results": case_results}

    output_path = ROOT / "simulator" / "evaluations" / "audit_selector_fallback.json"
    output_path.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
