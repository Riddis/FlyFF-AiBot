"""2026-08-14: Beginner routing-generalization suite, step 2 --
staggered two-wall S-shaped routes, per explicit user instruction.
General `kinodynamic_route_planner` (plan_route + select_persistent_
waypoint), default constants, UNCHANGED/no tuning. Zero PPO training.
Fresh spec pool, never inspected before this run.

Deliberately generous geometry, per explicit instruction ("test the
planner, not PPO"): two offset walls forcing an S-route, comfortable
gaps/separation, randomized mirror direction, randomized wall offsets/
spans/depths within safe ranges, randomized start heading (+/-30deg,
same range as step 1), simple straight-ahead final target, enough total
distance that a valid route always exists geometrically.

Wall 2's gap side is always the MIRROR of wall 1's (a true S-shape: go
around wall 1 to one side, then wall 2 forces the opposite side).
`wall_separation_cells` is the generous gap between wall 1's far edge
and wall 2's near edge -- this is the room the route/navigator has to
transition from avoiding one side to avoiding the other.

`_wall_cell_bounds` (single_obstacle_env) is reused unchanged per wall
(confirmed independent of `distance_cells`) via `build_multi_wall_world`
(scratchpad_general_router_episode.py, new, generalizes `build_single_
obstacle_world` to N walls).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from stable_baselines3 import PPO

from scratchpad_general_router_episode import build_multi_wall_world, run_episode_general_router, summarize_general_router
from simulator.single_obstacle_env import GAP_SIDE_SEED_OFFSET, MAP_HALF_SIZE_CELLS, ObstacleSpec, sample_obstacle_spec
from simulator.static_waypoint_env import FIXED_HEADING

ROOT = Path(__file__).resolve().parent
QUALIFIED_CHECKPOINT = ROOT / "models" / "generalized_waypoint_both_seed2_0051200.zip"
N_EVAL_PER_DIRECTION = 30
SUITE_SPEC_SEED = 663_000_000  # fresh, never used anywhere in this investigation before this script
APPROACH_HEADING_OFFSET_RANGE_RADIANS = (-math.radians(30.0), math.radians(30.0))

WALL1_OFFSET_RANGE = (6, 10)
WALL1_DEPTH_RANGE = (2, 4)
WALL1_HALF_SPAN_RANGE = (4, 7)
WALL_SEPARATION_RANGE = (10, 16)  # generous gap between wall1's far edge and wall2's near edge
WALL2_DEPTH_RANGE = (2, 4)
WALL2_HALF_SPAN_RANGE = (4, 7)
DISTANCE_PAST_WALL2_RANGE = (10.0, 16.0)  # final target this far past wall2's far edge

S_ROUTE_DIRECTIONS: tuple[Literal["left_then_right"], Literal["right_then_left"]] = ("left_then_right", "right_then_left")


@dataclass(frozen=True)
class TwoWallSpec:
    first_gap_side: Literal["left", "right"]
    wall1_offset_cells: int
    wall1_depth_cells: int
    wall1_half_span_cells: int
    wall_separation_cells: int
    wall2_depth_cells: int
    wall2_half_span_cells: int
    distance_cells: float  # total straight-line distance to the final target
    approach_heading_offset_radians: float

    @property
    def second_gap_side(self) -> Literal["left", "right"]:
        return "right" if self.first_gap_side == "left" else "left"

    def wall1_obstacle_spec(self) -> ObstacleSpec:
        return ObstacleSpec(
            gap_side=self.first_gap_side, distance_cells=self.distance_cells,
            wall_offset_cells=self.wall1_offset_cells, wall_depth_cells=self.wall1_depth_cells,
            half_span_cells=self.wall1_half_span_cells,
        )

    def wall2_obstacle_spec(self) -> ObstacleSpec:
        wall2_offset = self.wall1_offset_cells + self.wall1_depth_cells + self.wall_separation_cells
        return ObstacleSpec(
            gap_side=self.second_gap_side, distance_cells=self.distance_cells,
            wall_offset_cells=wall2_offset, wall_depth_cells=self.wall2_depth_cells,
            half_span_cells=self.wall2_half_span_cells,
        )


def sample_two_wall_spec(rng: np.random.Generator, *, direction: str) -> TwoWallSpec:
    first_gap_side = "left" if direction == "left_then_right" else "right"
    wall1_offset = int(rng.integers(WALL1_OFFSET_RANGE[0], WALL1_OFFSET_RANGE[1] + 1))
    wall1_depth = int(rng.integers(WALL1_DEPTH_RANGE[0], WALL1_DEPTH_RANGE[1] + 1))
    wall1_half_span = int(rng.integers(WALL1_HALF_SPAN_RANGE[0], WALL1_HALF_SPAN_RANGE[1] + 1))
    separation = int(rng.integers(WALL_SEPARATION_RANGE[0], WALL_SEPARATION_RANGE[1] + 1))
    wall2_depth = int(rng.integers(WALL2_DEPTH_RANGE[0], WALL2_DEPTH_RANGE[1] + 1))
    wall2_half_span = int(rng.integers(WALL2_HALF_SPAN_RANGE[0], WALL2_HALF_SPAN_RANGE[1] + 1))
    wall2_far_edge = wall1_offset + wall1_depth + separation + wall2_depth
    distance = wall2_far_edge + float(rng.uniform(*DISTANCE_PAST_WALL2_RANGE))
    heading_offset = float(rng.uniform(*APPROACH_HEADING_OFFSET_RANGE_RADIANS))
    return TwoWallSpec(
        first_gap_side=first_gap_side, wall1_offset_cells=wall1_offset, wall1_depth_cells=wall1_depth,
        wall1_half_span_cells=wall1_half_span, wall_separation_cells=separation,
        wall2_depth_cells=wall2_depth, wall2_half_span_cells=wall2_half_span,
        distance_cells=distance, approach_heading_offset_radians=heading_offset,
    )


def held_out_two_wall_specs_for_direction(n: int, *, direction: str, seed: int) -> list[TwoWallSpec]:
    offset = 0 if direction == "left_then_right" else 1000
    rng = np.random.default_rng(seed + offset)
    return [sample_two_wall_spec(rng, direction=direction) for _ in range(n)]


def main() -> None:
    assert QUALIFIED_CHECKPOINT.exists(), f"qualified checkpoint missing: {QUALIFIED_CHECKPOINT}"
    model = PPO.load(str(QUALIFIED_CHECKPOINT), device="cpu")
    print(f"{'=' * 90}")
    print("Beginner routing-generalization suite, step 2: staggered two-wall S-routes")
    print(f"Qualified checkpoint: {QUALIFIED_CHECKPOINT.name} -- ZERO obstacle-specific training")
    print("Router: general kinodynamic_route_planner, DEFAULT constants, no tuning")
    print(f"Fresh suite spec pool: seed={SUITE_SPEC_SEED} n={N_EVAL_PER_DIRECTION}/direction, never used before")
    print(f"{'=' * 90}")

    output = {}
    for direction in S_ROUTE_DIRECTIONS:
        specs = held_out_two_wall_specs_for_direction(N_EVAL_PER_DIRECTION, direction=direction, seed=SUITE_SPEC_SEED)
        results = []
        for i, spec in enumerate(specs):
            wall1, wall2 = spec.wall1_obstacle_spec(), spec.wall2_obstacle_spec()
            map_model, world = build_multi_wall_world([wall1, wall2])
            cell_size = map_model.native_units_per_cell
            center = MAP_HALF_SIZE_CELLS
            start_native = map_model.layout_to_native(center, center)
            final_native = (
                start_native[0] + math.cos(FIXED_HEADING) * spec.distance_cells * cell_size,
                start_native[1] + math.sin(FIXED_HEADING) * spec.distance_cells * cell_size,
            )
            initial_heading = FIXED_HEADING + spec.approach_heading_offset_radians
            result = run_episode_general_router(
                model, map_model, world, initial_heading=initial_heading, final_native=final_native,
                seed=935_000_000 + i,
                # pinned False: this is a historical run (recorded 55/60) predating
                # TargetPersistenceController's adoption as the default -- pin explicitly
                # so a rerun reproduces the same recorded numbers, not the new default.
                use_persistence_controller=False,
            )
            results.append(result)
        summary = summarize_general_router(results)
        output[direction] = {"summary": summary, "episodes": [r.__dict__ for r in results]}
        print(f"\n  --- {direction} ---")
        print(f"    success={summary['success_rate']:.2f} collision={summary['collision_rate']:.2f} "
              f"timeout={summary['timeout_rate']:.2f} planner_fail={summary['planner_failure_rate']:.2f} "
              f"ticks={summary['mean_ticks_to_success']} switches={summary['mean_target_switches']} "
              f"route_nodes={summary['mean_route_length_nodes']} "
              f"min_clearance(mean/min)={summary['mean_min_clearance_cells']:.2f}/{summary['min_min_clearance_cells']} "
              f"stages={summary['failure_stage_counts']} contact_ticks={summary['contact_ticks']}")

    total_n = sum(output[d]["summary"]["n"] for d in S_ROUTE_DIRECTIONS)
    total_success = sum(round(output[d]["summary"]["success_rate"] * output[d]["summary"]["n"]) for d in S_ROUTE_DIRECTIONS)
    total_collisions = sum(round(output[d]["summary"]["collision_rate"] * output[d]["summary"]["n"]) for d in S_ROUTE_DIRECTIONS)
    total_timeouts = sum(round(output[d]["summary"]["timeout_rate"] * output[d]["summary"]["n"]) for d in S_ROUTE_DIRECTIONS)
    total_planner_fail = sum(round(output[d]["summary"]["planner_failure_rate"] * output[d]["summary"]["n"]) for d in S_ROUTE_DIRECTIONS)
    print(f"\n{'=' * 90}")
    print(f"TOTAL (two-wall S-route, general planner): {total_success}/{total_n} success, "
          f"{total_collisions} collisions, {total_timeouts} timeouts, {total_planner_fail} planner failures")
    print(f"{'=' * 90}")

    output_path = ROOT / "evaluations" / "beginner_routing_two_wall_s_route_eval.json"
    output_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
