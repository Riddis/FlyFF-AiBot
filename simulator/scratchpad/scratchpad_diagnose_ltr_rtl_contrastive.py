"""2026-08-14: contrastive diagnostic, per explicit user instruction after
Phase B's clean rerun closed with no eligible checkpoint. Compares the
frozen `generalized_waypoint_both_seed2_0051200.zip` baseline against the
earliest clean, stably-improved continuation checkpoint
(`generalized_waypoint_router_mix_seed100_0056320.zip`) on the exact four
812M-dev-pool episodes that anchor Phase B's whole result:

  REPAIRED family  -- baseline collides, seed100@56320 succeeds:
    two_wall_left_then_right[21]  ("LTR21")
    two_wall_left_then_right[26]  ("LTR26")
  PERSISTENT-FAILURE family -- baseline collides, seed100@56320 ALSO collides:
    two_wall_right_then_left[8]   ("RTL8")
    two_wall_right_then_left[9]   ("RTL9")

Both checkpoints use the SAME default router configuration
(TargetPersistenceController, use_persistence_controller=True) -- this is
NOT an A/B selector comparison like the episode-67 diagnostic; the only
variable is which POLICY checkpoint is executing. Router/planner/
controller are frozen and untouched, exactly as in every other diagnostic
in this investigation.

DIAGNOSTIC ONLY: no tuning, no training on these episodes, no router/
controller changes. Per-tick trace fields match the episode-67 diagnostic
exactly (player pose, route candidate + its direct-hop clearance,
persisted/held target + its clearance, controller decision reason,
distance to each, steering action, realized post-step pose, post-step
point clearance, distance to true final target, contact) so the two
diagnostics are directly comparable.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from .scratchpad_beginner_navigation_mix_pools import DEV_POOL_SPEC_SEED, _reconstruct_two_wall_world, load_manifest
from simulator.environment import RecordedFarmingEnv
from simulator.kinodynamic_route_planner import (
    DESIRED_CLEARANCE_CELLS, TargetPersistenceController, _clearance_cells_native, _direct_hop_min_clearance,
    plan_route, select_persistent_waypoint,
)
from simulator.navigation_history import NavigationHistoryWrapper
from simulator.static_waypoint_env import SUCCESS_RADIUS_CELLS

ROOT = Path(__file__).resolve().parents[2]
BASELINE_CHECKPOINT = ROOT / "models" / "generalized_waypoint_both_seed2_0051200.zip"
SEED100_56320_CHECKPOINT = ROOT / "models" / "generalized_waypoint_router_mix_seed100_0056320.zip"
EPISODE_STEPS = 200
DIAGNOSTIC_SEED_BASE = 990_000_000
STEERING_NAMES = {0: "STRAIGHT", 1: "LEFT", 2: "RIGHT"}

EPISODES = [
    ("LTR21", "two_wall_left_then_right", 21, "repaired"),
    ("LTR26", "two_wall_left_then_right", 26, "repaired"),
    ("RTL8", "two_wall_right_then_left", 8, "persistent_failure"),
    ("RTL9", "two_wall_right_then_left", 9, "persistent_failure"),
]


def run_traced(model, map_model, world, *, initial_heading, final_native, episode_seed: int, label: str):
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
    assert len(route) >= 2, f"{label}: no route found -- cannot diagnose"

    controller = TargetPersistenceController(map_model, final_native[0], final_native[1])

    trace: list[dict] = []
    prev_contacts = 0
    outcome = "timeout"

    for tick in range(EPISODE_STEPS):
        pre_x, pre_z, pre_heading = base_env.player_x, base_env.player_z, base_env.heading

        candidate = select_persistent_waypoint(
            map_model, route, player_x=pre_x, player_z=pre_z, heading=pre_heading,
        )
        candidate_for_use = candidate if candidate is not None else final_native
        target = controller.update(candidate_for_use, player_x=pre_x, player_z=pre_z, route=route)
        reason = controller.last_switch_reason.value if controller.last_switch_reason else None

        candidate_clearance = _direct_hop_min_clearance(map_model, pre_x, pre_z, candidate_for_use[0], candidate_for_use[1])
        target_clearance = _direct_hop_min_clearance(map_model, pre_x, pre_z, target[0], target[1])
        dist_to_candidate = math.hypot(candidate_for_use[0] - pre_x, candidate_for_use[1] - pre_z) / cell_size
        dist_to_target = math.hypot(target[0] - pre_x, target[1] - pre_z) / cell_size

        base_env.actors[0].x, base_env.actors[0].z = target
        obs = env._augment(base_env._observation(), base_env.previous_steering)

        action, _state = model.predict(obs, deterministic=True)
        a = int(action[0])
        action_arr = np.asarray(action, dtype=np.int64).copy()
        action_arr[1] = 0
        obs, _reward, term, trunc, info = env.step(action_arr)

        post_x, post_z, post_heading = base_env.player_x, base_env.player_z, base_env.heading
        clearance_after = _clearance_cells_native(base_env.map, post_x, post_z)

        contacts = int(info.get("contacts", 0))
        contact_this_tick = contacts > prev_contacts
        prev_contacts = contacts

        fdx = final_native[0] - post_x
        fdz = final_native[1] - post_z
        final_distance = math.hypot(fdx, fdz) / cell_size

        row = {
            "tick": tick,
            "pre_pose_xzh_deg": (round(pre_x, 3), round(pre_z, 3), round(math.degrees(pre_heading), 2)),
            "route_candidate": (round(candidate_for_use[0], 3), round(candidate_for_use[1], 3)) if candidate is not None else "NONE_used_final_native",
            "candidate_clearance": round(candidate_clearance, 3),
            "dist_to_candidate_cells": round(dist_to_candidate, 3),
            "held_target": (round(target[0], 3), round(target[1], 3)),
            "target_clearance": round(target_clearance, 3),
            "dist_to_target_cells": round(dist_to_target, 3),
            "controller_reason": reason,
            "steering_action": STEERING_NAMES[a],
            "post_pose_xzh_deg": (round(post_x, 3), round(post_z, 3), round(math.degrees(post_heading), 2)),
            "clearance_after_step": round(clearance_after, 3),
            "final_distance_cells": round(final_distance, 3),
            "contact_this_tick": contact_this_tick,
        }
        trace.append(row)

        if contact_this_tick:
            outcome = "collision"
            break
        if final_distance <= SUCCESS_RADIUS_CELLS:
            outcome = "success"
            break

    env.close()
    return outcome, trace


def main() -> None:
    assert BASELINE_CHECKPOINT.exists(), f"missing: {BASELINE_CHECKPOINT}"
    assert SEED100_56320_CHECKPOINT.exists(), f"missing: {SEED100_56320_CHECKPOINT}"
    baseline_model = PPO.load(str(BASELINE_CHECKPOINT), device="cpu")
    trained_model = PPO.load(str(SEED100_56320_CHECKPOINT), device="cpu")

    manifest = load_manifest(ROOT / "simulator" / "evaluations" / f"router_mix_dev_pool_{DEV_POOL_SPEC_SEED}_manifest.json")

    results: dict[str, dict] = {}
    for label, stratum_name, index, family in EPISODES:
        episode = manifest["strata"][stratum_name]["accepted"][index]
        map_model, world, final_native, initial_heading = _reconstruct_two_wall_world(episode)
        print(f"\n{'=' * 100}\n{label} ({family}): stratum={stratum_name} index={index} "
              f"final_native={final_native} initial_heading_deg={math.degrees(initial_heading):.2f}\n"
              f"spec: {episode}\n{'=' * 100}")

        episode_results = {}
        for model_label, model in [("baseline_0051200", baseline_model), ("trained_seed100_0056320", trained_model)]:
            outcome, trace = run_traced(
                model, map_model, world, initial_heading=initial_heading, final_native=final_native,
                episode_seed=DIAGNOSTIC_SEED_BASE + index, label=f"{label}/{model_label}",
            )
            episode_results[model_label] = {"outcome": outcome, "trace": trace}
            print(f"\n  --- {model_label}: outcome={outcome} ticks={len(trace)} ---")
            for row in trace:
                print(f"    {json.dumps(row, default=str)}")

        results[label] = {"family": family, "stratum": stratum_name, "index": index, "results": episode_results}

    output_path = ROOT / "simulator" / "evaluations" / "diagnose_ltr_rtl_contrastive.json"
    output_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
