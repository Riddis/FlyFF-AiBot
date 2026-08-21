"""2026-08-14: Beginner routing-generalization suite, step 1 of the
user's explicit progression ("randomized single walls"). Tests whether
the FROZEN router+navigator pair (which just passed 45/45 on the
existing single-wall pool) generalizes to wall geometry it hasn't been
tested against, with ZERO PPO training and the router UNCHANGED.

Per explicit instruction, the 779M pool / margin=6/forward=2 router
combo is "no longer untouched evidence" (reused repeatedly during
development) -- this suite uses a genuinely NEW spec-pool seed
(640_000_000) and evaluates strictly for generalization, not as a
qualification-grade confirmation run.

What's already randomized by the existing `sample_obstacle_spec`
(simulator/single_obstacle_env.py, reused UNCHANGED here): waypoint
distance, wall offset (how far ahead), wall depth (thickness),
half-span (lateral extent/width), straight-offset (for "none" side).
NEW in this suite: approach-heading offset -- the player now starts
facing up to +/-30 degrees away from the corridor axis, so the router/
navigator must actually correct heading rather than always starting
perfectly aimed down the corridor. This directly matches the user's
"different approach headings" request. Player spawn position and 2D
final-target offset are NOT varied yet (still center-spawn, straight-
ahead target along the corridor axis) -- left for a later suite step if
this one passes, per "don't make it complicated yet."

The router stays EXACTLY compute_subgoal_cells (LATERAL_MARGIN_CELLS=6,
FORWARD_OFFSET_CELLS=2, imported from the frozen transfer-eval script,
not reimplemented) -- a single subgoal, offset past the wall's far edge
toward the gap side. No changes to simulator/ production modules.

Per-episode diagnostics (the user's requested failure-layer isolation,
adapted to this single-subgoal architecture -- see module note on
"during handoff" below):
  - route_found: whether a subgoal was produced when one was needed
    (always true for this deterministic formula; field kept for
    forward-compatibility with a general planner that COULD fail)
  - num_subgoals_issued: 0 ("none" side, direct route) or 1 (left/right)
  - subgoal_reached: whether the episode ever got within SUBGOAL_RADIUS_
    CELLS of the subgoal and switched to the final target
  - phase_at_end: "no_subgoal_needed" | "before_subgoal" | "after_subgoal"
  - failure_stage: derived from outcome + phase_at_end -- distinguishes
    "never got past the subgoal leg" from "handled the subgoal fine but
    failed on final approach" (NOTE: this single-subgoal architecture's
    handoff is a single instantaneous radius-triggered tick, not a
    multi-tick maneuver -- a literal separate "failure DURING handoff"
    category isn't meaningfully distinct here; a collision on the exact
    handoff tick is already captured under "before_subgoal" since the
    collision check runs before the switch check each tick. A true
    multi-tick handoff-hazard category becomes meaningful in step 2
    (multi-subgoal S-routes), not before.)
  - contact / timeout / min_clearance -- as in the original transfer eval.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from scratchpad_single_obstacle_transfer_eval_calibrated_arc import (
    FORWARD_OFFSET_CELLS, LATERAL_MARGIN_CELLS, SUBGOAL_RADIUS_CELLS, compute_subgoal_cells,
)
from simulator.environment import RecordedFarmingEnv
from navigation.kinodynamic_route_planner import _clearance_cells_native
from simulator.navigation_history import NavigationHistoryWrapper
from simulator.single_obstacle_env import (
    GAP_SIDE_SEED_OFFSET, GAP_SIDES, MAP_HALF_SIZE_CELLS, ObstacleSpec, SUCCESS_RADIUS_CELLS,
    build_single_obstacle_world, sample_obstacle_spec,
)
from simulator.static_waypoint_env import FIXED_HEADING, SYMMETRIC_MOVEMENT

ROOT = Path(__file__).resolve().parents[1]
QUALIFIED_CHECKPOINT = ROOT / "models" / "generalized_waypoint_both_seed2_0051200.zip"
EPISODE_STEPS = 150
N_EVAL_PER_SIDE = 30  # larger than the original 15/side: wider parameter space now (heading added)
SUITE_SPEC_SEED = 640_000_000  # NEW -- distinct from 779_000_000 (reused/no-longer-untouched)
APPROACH_HEADING_OFFSET_RANGE_RADIANS = (-math.radians(30.0), math.radians(30.0))

STEERING_NAMES = {0: "STRAIGHT", 1: "LEFT", 2: "RIGHT"}


@dataclass(frozen=True)
class RandomizedWallSpec:
    obstacle: ObstacleSpec
    approach_heading_offset_radians: float


def sample_randomized_wall_spec(rng: np.random.Generator, *, gap_side: str) -> RandomizedWallSpec:
    obstacle = sample_obstacle_spec(rng, gap_side=gap_side)  # UNCHANGED default ranges
    heading_offset = float(rng.uniform(*APPROACH_HEADING_OFFSET_RANGE_RADIANS))
    return RandomizedWallSpec(obstacle=obstacle, approach_heading_offset_radians=heading_offset)


def held_out_randomized_wall_specs_for_side(n: int, *, gap_side: str, seed: int) -> list[RandomizedWallSpec]:
    rng = np.random.default_rng(seed + GAP_SIDE_SEED_OFFSET[gap_side])
    return [sample_randomized_wall_spec(rng, gap_side=gap_side) for _ in range(n)]


def run_episode(model: PPO, spec: RandomizedWallSpec, *, use_subgoal: bool, seed: int) -> dict:
    obstacle = spec.obstacle
    map_model, world = build_single_obstacle_world(obstacle, movement=SYMMETRIC_MOVEMENT)
    raw_env = RecordedFarmingEnv(world, map_model=map_model, episode_steps=EPISODE_STEPS)
    env = NavigationHistoryWrapper(raw_env)
    obs, _info = env.reset(seed=seed)
    base_env = env.unwrapped
    for actor in base_env.actors[1:]:
        actor.alive = False
    # The NEW variation: player starts off the corridor axis by up to +/-30deg.
    # The final target and wall stay on the ORIGINAL corridor axis -- the
    # router/navigator must correct heading, not just proceed straight.
    base_env.heading = FIXED_HEADING + spec.approach_heading_offset_radians

    center = MAP_HALF_SIZE_CELLS
    cell_size = base_env.map.native_units_per_cell
    final_native = (
        base_env.player_x + math.cos(FIXED_HEADING) * obstacle.distance_cells * cell_size,
        base_env.player_z + math.sin(FIXED_HEADING) * obstacle.distance_cells * cell_size,
    )
    base_env.actors[0].x, base_env.actors[0].z = final_native
    base_env.actors[0].alive = True

    subgoal_cells = compute_subgoal_cells(obstacle, center) if use_subgoal else None
    route_found = (obstacle.gap_side == "none") or (subgoal_cells is not None)
    num_subgoals_issued = 0 if subgoal_cells is None else 1
    if subgoal_cells is not None:
        current_target_native = base_env.map.layout_to_native(*subgoal_cells)
        heading_toward_subgoal = True
    else:
        current_target_native = final_native
        heading_toward_subgoal = False

    obs = env._augment(base_env._observation())
    prev_contacts = 0
    switched_to_final = not heading_toward_subgoal
    subgoal_switches = 0
    steering_sequence: list[int] = []
    clearances: list[float] = []
    contact_tick: int | None = None
    outcome = "timeout"
    final_ticks = EPISODE_STEPS
    phase_at_end = "no_subgoal_needed" if subgoal_cells is None else "before_subgoal"

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
            subgoal_switches += 1
            phase_at_end = "after_subgoal"
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

    if outcome == "success":
        failure_stage = "success"
    elif phase_at_end == "before_subgoal":
        failure_stage = "failure_before_subgoal_reached"
    else:
        failure_stage = "failure_approaching_final_target"

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
        "oscillation_rate": oscillation_rate,
        "reversal_rate": reversal_rate,
        "straight_fraction": straight_fraction,
        "min_clearance_cells": min(clearances) if clearances else None,
        "route_found": route_found,
        "num_subgoals_issued": num_subgoals_issued,
        "subgoal_reached": switched_to_final if num_subgoals_issued > 0 else None,
        "subgoal_switches": subgoal_switches,
        "phase_at_end": phase_at_end,
        "failure_stage": failure_stage,
        "approach_heading_offset_degrees": math.degrees(spec.approach_heading_offset_radians),
        "wall_depth_cells": obstacle.wall_depth_cells,
        "half_span_cells": obstacle.half_span_cells,
        "wall_offset_cells": obstacle.wall_offset_cells,
        "distance_cells": obstacle.distance_cells,
    }


def summarize(results: list[dict]) -> dict:
    n = len(results)
    successes = [r for r in results if r["outcome"] == "success"]
    collisions = [r for r in results if r["outcome"] == "collision"]
    timeouts = [r for r in results if r["outcome"] == "timeout"]
    stage_counts: dict[str, int] = {}
    for r in results:
        stage_counts[r["failure_stage"]] = stage_counts.get(r["failure_stage"], 0) + 1
    return {
        "n": n,
        "success_rate": len(successes) / n if n else None,
        "collision_rate": len(collisions) / n if n else None,
        "timeout_rate": len(timeouts) / n if n else None,
        "mean_ticks_to_success": float(np.mean([r["ticks"] for r in successes])) if successes else None,
        "mean_oscillation_rate": float(np.mean([r["oscillation_rate"] for r in results if r["oscillation_rate"] is not None])),
        "mean_reversal_rate": float(np.mean([r["reversal_rate"] for r in results if r["reversal_rate"] is not None])),
        "mean_min_clearance_cells": float(np.mean([r["min_clearance_cells"] for r in results if r["min_clearance_cells"] is not None])) if results else None,
        "min_min_clearance_cells": min((r["min_clearance_cells"] for r in results if r["min_clearance_cells"] is not None), default=None),
        "failure_stage_counts": stage_counts,
        "contact_ticks": [r["contact_tick"] for r in collisions],
    }


def main() -> None:
    assert QUALIFIED_CHECKPOINT.exists(), f"qualified checkpoint missing: {QUALIFIED_CHECKPOINT}"
    model = PPO.load(str(QUALIFIED_CHECKPOINT), device="cpu")
    print(f"{'=' * 90}")
    print(f"Beginner routing-generalization suite, step 1: randomized single walls")
    print(f"Qualified checkpoint: {QUALIFIED_CHECKPOINT.name} -- ZERO obstacle-specific training")
    print(f"Suite spec pool: seed={SUITE_SPEC_SEED} n={N_EVAL_PER_SIDE}/side, NEW/untouched for this suite")
    print(f"{'=' * 90}")

    output = {}
    for use_subgoal in (False, True):
        label = "WITH route subgoal (frozen, margin=6/forward=2)" if use_subgoal else "final-waypoint-only"
        print(f"\n  --- {label} ---")
        condition_output = {}
        for side in GAP_SIDES:
            specs = held_out_randomized_wall_specs_for_side(N_EVAL_PER_SIDE, gap_side=side, seed=SUITE_SPEC_SEED)
            results = [run_episode(model, spec, use_subgoal=use_subgoal, seed=925_000_000 + i) for i, spec in enumerate(specs)]
            summary = summarize(results)
            condition_output[side] = {"summary": summary, "episodes": results}
            print(f"    {side:6s}: success={summary['success_rate']:.2f} collision={summary['collision_rate']:.2f} "
                  f"timeout={summary['timeout_rate']:.2f} ticks={summary['mean_ticks_to_success']} "
                  f"osc={summary['mean_oscillation_rate']:.3f} reversal={summary['mean_reversal_rate']:.3f} "
                  f"min_clearance(mean/min)={summary['mean_min_clearance_cells']:.2f}/{summary['min_min_clearance_cells']} "
                  f"stages={summary['failure_stage_counts']} contact_ticks={summary['contact_ticks']}")
        output["with_subgoal" if use_subgoal else "final_waypoint_only"] = condition_output

    routed = output["with_subgoal"]
    total_n = sum(routed[side]["summary"]["n"] for side in GAP_SIDES)
    total_collisions = sum(round(routed[side]["summary"]["collision_rate"] * routed[side]["summary"]["n"]) for side in GAP_SIDES)
    total_timeouts = sum(round(routed[side]["summary"]["timeout_rate"] * routed[side]["summary"]["n"]) for side in GAP_SIDES)
    strict_pass = total_collisions == 0 and total_timeouts == 0
    print(f"\n{'=' * 90}")
    print(f"ROUTED CONDITION: zero collisions AND zero timeouts across full pool (n={total_n})")
    print(f"  collisions={total_collisions}  timeouts={total_timeouts}  -> {'PASS' if strict_pass else 'FAIL -- classify before any training decision'}")
    print(f"{'=' * 90}")

    output_path = ROOT / "evaluations" / "beginner_routing_randomized_walls_eval.json"
    output_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
