"""2026-08-14: focused diagnostic replay of corrected two_wall_right_then_left[67]
(spec_seed=706_000_000, index=67), A vs B, per explicit user instruction after
the corrected-baseline paired A/B rerun introduced this as a NEW regressed
episode (A: success, B: collision) that did not exist under the pre-fix
(previous_steering-bugged) observations. Purpose: distinguish two competing
hypotheses --

  (1) known PPO-execution mismatch: B's held target remains planner-safe
      throughout, but PPO's own realized arc drifts into the wall despite a
      safe target (the already-characterized "planner-safe, PPO-executed-
      unsafe clearance-decay" pathology), vs
  (2) controller-specific regression: TargetPersistenceController's
      persistence itself retains/selects a target whose HOLDING (as opposed
      to A's fresh-every-tick reselection) materially creates the hazardous
      trajectory -- i.e. a controller-induced difference, not merely a PPO
      execution-quality question.

NO TUNING. NO NEW POOL. NO CONTROLLER CHANGES. This reproduces the exact
already-observed episode (same seed, same sampler calls, same map/spec
construction as scratchpad_paired_ab_selector_test.py) under both conditions
with full per-tick instrumentation logged, reusing plan_route/select_
persistent_waypoint/TargetPersistenceController completely unmodified.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from scratchpad_beginner_routing_two_wall_s_route import held_out_two_wall_specs_for_direction
from scratchpad_general_router_episode import build_multi_wall_world
from simulator.environment import RecordedFarmingEnv
from simulator.kinodynamic_route_planner import (
    DESIRED_CLEARANCE_CELLS, TargetPersistenceController, _clearance_cells_native,
    _direct_hop_min_clearance, plan_route, select_persistent_waypoint,
)
from simulator.navigation_history import NavigationHistoryWrapper
from simulator.single_obstacle_env import MAP_HALF_SIZE_CELLS
from simulator.static_waypoint_env import FIXED_HEADING, SUCCESS_RADIUS_CELLS

ROOT = Path(__file__).resolve().parent
QUALIFIED_CHECKPOINT = ROOT / "models" / "generalized_waypoint_both_seed2_0051200.zip"

TWO_WALL_AB_SEED = 706_000_000
N_TWO_WALL_PER_DIRECTION = 100
DIRECTION = "right_then_left"
INDEX = 67
EPISODE_SEED = 950_000_000 + INDEX  # exact reproduction of scratchpad_paired_ab_selector_test.py's seed formula
EPISODE_STEPS = 200  # matches run_episode_general_router's EPISODE_STEPS
STEERING_NAMES = {0: "STRAIGHT", 1: "LEFT", 2: "RIGHT"}


def _final_native(map_model, distance_cells, heading_offset):
    cell_size = map_model.native_units_per_cell
    center = MAP_HALF_SIZE_CELLS
    start_native = map_model.layout_to_native(center, center)
    final_native = (
        start_native[0] + math.cos(FIXED_HEADING) * distance_cells * cell_size,
        start_native[1] + math.sin(FIXED_HEADING) * distance_cells * cell_size,
    )
    return final_native, FIXED_HEADING + heading_offset


def run_traced(model, map_model, world, *, initial_heading, final_native, use_persistence_controller: bool, label: str):
    raw_env = RecordedFarmingEnv(world, map_model=map_model, episode_steps=EPISODE_STEPS)
    env = NavigationHistoryWrapper(raw_env)
    obs, _info = env.reset(seed=EPISODE_SEED)
    base_env = env.unwrapped
    for actor in base_env.actors[1:]:
        actor.alive = False
    base_env.heading = initial_heading
    cell_size = base_env.map.native_units_per_cell

    base_env.actors[0].x, base_env.actors[0].z = final_native
    base_env.actors[0].alive = True

    stats: dict = {}
    route = plan_route(
        map_model, start_x=base_env.player_x, start_z=base_env.player_z, start_heading=base_env.heading,
        destination_x=final_native[0], destination_z=final_native[1], stats=stats,
    )
    assert len(route) >= 2, f"{label}: no route found -- cannot diagnose"

    controller = TargetPersistenceController(map_model, final_native[0], final_native[1]) if use_persistence_controller else None

    trace: list[dict] = []
    prev_contacts = 0
    outcome = "timeout"

    for tick in range(EPISODE_STEPS):
        pre_x, pre_z, pre_heading = base_env.player_x, base_env.player_z, base_env.heading

        candidate = select_persistent_waypoint(
            map_model, route, player_x=pre_x, player_z=pre_z, heading=pre_heading,
        )
        candidate_for_use = candidate if candidate is not None else final_native

        if controller is not None:
            target = controller.update(candidate_for_use, player_x=pre_x, player_z=pre_z, route=route)
            reason = controller.last_switch_reason.value if controller.last_switch_reason else None
        else:
            target = candidate_for_use
            reason = None

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
    assert QUALIFIED_CHECKPOINT.exists(), f"qualified checkpoint missing: {QUALIFIED_CHECKPOINT}"
    model = PPO.load(str(QUALIFIED_CHECKPOINT), device="cpu")

    specs = held_out_two_wall_specs_for_direction(N_TWO_WALL_PER_DIRECTION, direction=DIRECTION, seed=TWO_WALL_AB_SEED)
    spec = specs[INDEX]
    wall1, wall2 = spec.wall1_obstacle_spec(), spec.wall2_obstacle_spec()
    map_model, world = build_multi_wall_world([wall1, wall2])
    final_native, initial_heading = _final_native(map_model, spec.distance_cells, spec.approach_heading_offset_radians)

    print(f"{'=' * 100}")
    print(f"DIAGNOSTIC REPLAY: two_wall_{DIRECTION}[{INDEX}]  spec_seed={TWO_WALL_AB_SEED}  episode_seed={EPISODE_SEED}")
    print(f"final_native={final_native}  initial_heading_deg={math.degrees(initial_heading):.2f}  "
          f"DESIRED_CLEARANCE_CELLS={DESIRED_CLEARANCE_CELLS}  SUCCESS_RADIUS_CELLS={SUCCESS_RADIUS_CELLS}")
    print(f"spec: first_gap_side={spec.first_gap_side} wall1_offset={spec.wall1_offset_cells} "
          f"wall_separation={spec.wall_separation_cells} wall2_offset~={spec.wall1_offset_cells + spec.wall1_depth_cells + spec.wall_separation_cells} "
          f"distance_cells={spec.distance_cells:.2f} approach_heading_offset_deg={math.degrees(spec.approach_heading_offset_radians):.2f}")
    print(f"{'=' * 100}")

    results = {}
    for label, use_pc in [("A_stateless", False), ("B_persistence_controller", True)]:
        outcome, trace = run_traced(
            model, map_model, world, initial_heading=initial_heading, final_native=final_native,
            use_persistence_controller=use_pc, label=label,
        )
        results[label] = {"outcome": outcome, "trace": trace}
        print(f"\n{'=' * 100}\n{label}: outcome={outcome}  ticks_run={len(trace)}\n{'=' * 100}")
        for row in trace:
            print(json.dumps(row, default=str))

    output_path = ROOT / "evaluations" / "diagnose_two_wall_rtl_67_regression.json"
    output_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
