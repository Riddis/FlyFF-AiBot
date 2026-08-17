"""2026-08-14: single-obstacle ZERO-obstacle-specific-training transfer
evaluation of the three frozen calibrated-arc generalized-waypoint
checkpoints (generalized_waypoint_calibrated_arc_seed{0,2,8}_0040960.zip),
per explicit user instruction.

Direct successor to scratchpad_single_obstacle_route_subgoal_control.py
(same architecture, same held-out spec pool: held_out_obstacle_specs_for_
side(15, seed=779_000_000) per gap_side -- reused UNCHANGED so the legacy-
physics numbers from that script remain a genuine same-geometry historical
comparison, not merely same-template). Two conditions, both with the SAME
frozen checkpoint and ZERO PPO training:

  1. final-waypoint-only: the policy is given only the true final waypoint
     every tick -- tests how much local detour topology the corrected-
     physics navigator can solve entirely on its own.
  2. WITH route subgoal: the existing trivial deterministic router
     (compute_subgoal_cells -- a point just past the wall, offset to the
     gap's open side) feeds an intermediate subgoal first, then the true
     waypoint once past it -- tests whether route assistance helps (or is
     even necessary) under the corrected physics.

Per explicit instruction: do NOT tune LATERAL_MARGIN_CELLS/FORWARD_OFFSET_
CELLS against these new results -- they are copied UNCHANGED from the
legacy-physics-tuned control script. If the new, far stronger steering
model needs different margins, that is itself a finding to report, not
something to silently fix here.

Reports, per seed x condition x gap_side: success, any physical contact,
timeout/non-success, path efficiency (geodesic-based, matching single_
obstacle_env's own methodology), ticks to success, oscillation rate,
direct LEFT<->RIGHT reversal rate, STRAIGHT fraction, minimum obstacle
clearance (simulator.kinodynamic_route_planner._clearance_cells_native,
the same genuine local-search clearance measure the kinodynamic planner
itself uses), and the contact tick if applicable.

The `movement` field required by RecordedWorldModel/build_single_obstacle_
world is inert (provenance-only, verified in tests/test_physics_version_
tag_provenance_only.py) -- uses static_waypoint_env.SYMMETRIC_MOVEMENT as
placeholder content, not a legacy-curriculum "reference movement" fetch
(that dependency no longer means anything for physics, same reasoning as
scratchpad_generalized_waypoint_train_calibrated_arc.py).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from simulator.environment import RecordedFarmingEnv
from navigation.kinodynamic_route_planner import _clearance_cells_native
from simulator.navigation_history import NavigationHistoryWrapper
from simulator.single_obstacle_env import (
    GAP_SIDES, MAP_HALF_SIZE_CELLS, ObstacleSpec, SUCCESS_RADIUS_CELLS, _wall_cell_bounds,
    build_single_obstacle_world, held_out_obstacle_specs_for_side,
)
from simulator.static_waypoint_env import FIXED_HEADING, SYMMETRIC_MOVEMENT

ROOT = Path(__file__).resolve().parent
EPISODE_STEPS = 150
N_EVAL_PER_SIDE = 15
EVAL_SEED = 779_000_000  # SAME as the legacy scratchpad_single_obstacle_route_subgoal_control.py

SUBGOAL_RADIUS_CELLS = 3.0
LATERAL_MARGIN_CELLS = 6  # unchanged from the legacy-physics-tuned control script -- see module docstring
FORWARD_OFFSET_CELLS = 2

STEERING_NAMES = {0: "STRAIGHT", 1: "LEFT", 2: "RIGHT"}


def compute_subgoal_cells(spec: ObstacleSpec, center: int) -> tuple[int, int] | None:
    x_start, x_end, y_start, y_end = _wall_cell_bounds(spec, center)
    if spec.gap_side == "left":
        subgoal_y = y_start - LATERAL_MARGIN_CELLS
    elif spec.gap_side == "right":
        subgoal_y = y_end + LATERAL_MARGIN_CELLS
    else:
        return None
    subgoal_x = x_end + FORWARD_OFFSET_CELLS
    return subgoal_x, subgoal_y


def run_episode(model: PPO, spec: ObstacleSpec, *, use_subgoal: bool, seed: int) -> dict:
    map_model, world = build_single_obstacle_world(spec, movement=SYMMETRIC_MOVEMENT)
    raw_env = RecordedFarmingEnv(world, map_model=map_model, episode_steps=EPISODE_STEPS)
    env = NavigationHistoryWrapper(raw_env)
    obs, _info = env.reset(seed=seed)
    base_env = env.unwrapped
    for actor in base_env.actors[1:]:
        actor.alive = False
    base_env.heading = FIXED_HEADING

    center = MAP_HALF_SIZE_CELLS
    cell_size = base_env.map.native_units_per_cell
    final_native = (
        base_env.player_x + math.cos(FIXED_HEADING) * spec.distance_cells * cell_size,
        base_env.player_z + math.sin(FIXED_HEADING) * spec.distance_cells * cell_size,
    )
    base_env.actors[0].x, base_env.actors[0].z = final_native
    base_env.actors[0].alive = True

    start_cell = base_env.map.native_to_layout_cell(base_env.player_x, base_env.player_z)
    final_cell = base_env.map.native_to_layout_cell(*final_native)
    initial_geodesic = math.inf
    if start_cell is not None and final_cell is not None:
        field = base_env._geodesic_field(start_cell)
        initial_geodesic = float(field.get(final_cell, math.inf))

    subgoal_cells = compute_subgoal_cells(spec, center) if use_subgoal else None
    if subgoal_cells is not None:
        current_target_native = base_env.map.layout_to_native(*subgoal_cells)
        heading_toward_subgoal = True
    else:
        current_target_native = final_native
        heading_toward_subgoal = False

    obs = env._augment(base_env._observation())
    prev_contacts = 0
    switched_to_final = not heading_toward_subgoal
    steering_sequence: list[int] = []
    clearances: list[float] = []
    contact_tick: int | None = None
    outcome = "timeout"
    final_ticks = EPISODE_STEPS

    for tick in range(EPISODE_STEPS):
        target_x, target_z = current_target_native
        base_env.actors[0].x, base_env.actors[0].z = target_x, target_z
        obs = env._augment(base_env._observation())

        action, _state = model.predict(obs, deterministic=True)
        a = int(action[0])
        steering_sequence.append(a)
        action_arr = np.asarray(action, dtype=np.int64).copy()
        action_arr[1] = 0
        obs, _reward, term, trunc, info = env.step(action_arr)

        clearances.append(_clearance_cells_native(base_env.map, base_env.player_x, base_env.player_z))

        contacts = int(info.get("contacts", 0))
        if contacts > prev_contacts:
            contact_tick = tick + 1
            outcome = "collision"
            final_ticks = tick + 1
            break
        prev_contacts = contacts

        dx = target_x - base_env.player_x
        dz = target_z - base_env.player_z
        dist_to_current_target = math.hypot(dx, dz) / cell_size
        if not switched_to_final and dist_to_current_target <= SUBGOAL_RADIUS_CELLS:
            switched_to_final = True
            current_target_native = final_native
            continue

        fdx = final_native[0] - base_env.player_x
        fdz = final_native[1] - base_env.player_z
        dist_to_final = math.hypot(fdx, fdz) / cell_size
        if switched_to_final and dist_to_final <= SUCCESS_RADIUS_CELLS:
            outcome = "success"
            final_ticks = tick + 1
            break

    env.close()

    path_efficiency = None
    if outcome == "success" and math.isfinite(initial_geodesic):
        traveled = float(info.get("total_distance_cells", 0.0))
        required_progress = max(0.0, initial_geodesic - SUCCESS_RADIUS_CELLS)
        if traveled > 0 and required_progress > 0:
            path_efficiency = required_progress / traveled

    oscillation_rate = None
    reversal_rate = None
    if len(steering_sequence) > 1:
        switches = sum(1 for x, y in zip(steering_sequence, steering_sequence[1:]) if x != y)
        oscillation_rate = switches / (len(steering_sequence) - 1)
        reversals = sum(1 for x, y in zip(steering_sequence, steering_sequence[1:]) if (x == 1 and y == 2) or (x == 2 and y == 1))
        reversal_rate = reversals / (len(steering_sequence) - 1)
    straight_fraction = (steering_sequence.count(0) / len(steering_sequence)) if steering_sequence else None

    return {
        "outcome": outcome,
        "ticks": final_ticks,
        "contact_tick": contact_tick,
        "path_efficiency": path_efficiency,
        "oscillation_rate": oscillation_rate,
        "reversal_rate": reversal_rate,
        "straight_fraction": straight_fraction,
        "min_clearance_cells": min(clearances) if clearances else None,
        "initial_geodesic": initial_geodesic if math.isfinite(initial_geodesic) else None,
    }


def summarize(results: list[dict]) -> dict:
    n = len(results)
    successes = [r for r in results if r["outcome"] == "success"]
    collisions = [r for r in results if r["outcome"] == "collision"]
    timeouts = [r for r in results if r["outcome"] == "timeout"]
    return {
        "n": n,
        "success_rate": len(successes) / n if n else None,
        "collision_rate": len(collisions) / n if n else None,
        "timeout_rate": len(timeouts) / n if n else None,
        "mean_ticks_to_success": float(np.mean([r["ticks"] for r in successes])) if successes else None,
        "mean_path_efficiency": float(np.mean([r["path_efficiency"] for r in successes if r["path_efficiency"] is not None])) if successes else None,
        "mean_oscillation_rate": float(np.mean([r["oscillation_rate"] for r in results if r["oscillation_rate"] is not None])),
        "mean_reversal_rate": float(np.mean([r["reversal_rate"] for r in results if r["reversal_rate"] is not None])),
        "mean_straight_fraction": float(np.mean([r["straight_fraction"] for r in results if r["straight_fraction"] is not None])),
        "mean_min_clearance_cells": float(np.mean([r["min_clearance_cells"] for r in results if r["min_clearance_cells"] is not None])),
        "min_min_clearance_cells": min((r["min_clearance_cells"] for r in results if r["min_clearance_cells"] is not None), default=None),
        "contact_ticks": [r["contact_tick"] for r in collisions],
    }


def main() -> None:
    all_output = {}
    for seed in (0, 2, 8):
        model_path = ROOT / "models" / f"generalized_waypoint_calibrated_arc_seed{seed}_0040960.zip"
        if not model_path.exists():
            print(f"seed={seed}: checkpoint not found, skipping", flush=True)
            continue
        model = PPO.load(str(model_path), device="cpu")
        print(f"\n{'=' * 90}")
        print(f"seed={seed} (calibrated-arc generalized-waypoint checkpoint, ZERO obstacle-specific training)")
        print(f"{'=' * 90}")
        seed_output = {}

        for use_subgoal in (False, True):
            label = "WITH route subgoal" if use_subgoal else "final-waypoint-only"
            print(f"\n  --- {label} ---")
            condition_output = {}
            for side in GAP_SIDES:
                specs = held_out_obstacle_specs_for_side(N_EVAL_PER_SIDE, gap_side=side, seed=EVAL_SEED)
                results = [run_episode(model, spec, use_subgoal=use_subgoal, seed=920_000_000 + i) for i, spec in enumerate(specs)]
                summary = summarize(results)
                condition_output[side] = {"summary": summary, "episodes": results}
                print(f"    {side:6s}: success={summary['success_rate']:.2f} collision={summary['collision_rate']:.2f} "
                      f"timeout={summary['timeout_rate']:.2f} ticks={summary['mean_ticks_to_success']} "
                      f"path_eff={summary['mean_path_efficiency']} osc={summary['mean_oscillation_rate']:.3f} "
                      f"reversal={summary['mean_reversal_rate']:.3f} straight_frac={summary['mean_straight_fraction']:.3f} "
                      f"min_clearance(mean/min)={summary['mean_min_clearance_cells']:.2f}/{summary['min_min_clearance_cells']} "
                      f"contact_ticks={summary['contact_ticks']}")
            seed_output[("with_subgoal" if use_subgoal else "final_waypoint_only")] = condition_output
        all_output[str(seed)] = seed_output

    output_path = ROOT / "evaluations" / "single_obstacle_calibrated_arc_transfer_eval.json"
    output_path.write_text(json.dumps(all_output, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
