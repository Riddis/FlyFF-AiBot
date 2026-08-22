"""2026-08-14: durable, mechanically-replayable regression fixtures for
the general-router selector layer, per explicit user instruction --
"save these specific failing geometries/specs as regression fixtures...
months from now we can ask: does the new model/router fix these old
cases? did it break anything that used to work?"

Every fixture is specified as (pool_type, side_or_direction, spec_seed,
index) -- NOT hand-copied coordinates -- so it regenerates the exact
spec deterministically via the same sampler functions used throughout
this investigation (`held_out_randomized_wall_specs_for_side` /
`held_out_two_wall_specs_for_direction`), immune to transcription error
and trivially reusable against a different checkpoint or router config
later.

Records the EXPECTED outcome under BOTH conditions as observed on
2026-08-14, against the checkpoint frozen at that time
(models/generalized_waypoint_both_seed2_0051200.zip):
  - condition "A" = plain select_persistent_waypoint() (no wrapper)
  - condition "B" = select_persistent_waypoint() + TargetPersistenceController
    (ADOPTED, the active configuration as of this writing)

Sources:
  - 6 cases first diagnosed in scratchpad_route_follower_selector_audit.py
    (2 orbit-timeouts + 4 large-index-jump collisions on the 640M/663M
    development pools)
  - 1 planner search-budget-exhaustion case (663M pool, right_then_left[20],
    outcome identical regardless of selector choice -- plan_route's own
    search, untouched by either A or B)
  - 19 cases B still fails on the fresh, untouched 705M/706M paired-test
    pool (15 collisions + 1 timeout + 3 planner-search-budget failures --
    3 of those are the SAME planner-budget issue as above, on different
    geometry)

Usage: `python scratchpad_routing_regression_fixtures.py [checkpoint_path]`
-- defaults to the currently-qualified checkpoint. Reports PASS (matches
the recorded expected outcome) / DIFFERENT (worth investigating -- could
be an improvement OR a new regression, both worth a human look) per
fixture, for both A and B, plus a summary. Does NOT tune anything; a
"DIFFERENT" result is information, not automatically good or bad.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from stable_baselines3 import PPO

from .scratchpad_beginner_routing_randomized_walls import held_out_randomized_wall_specs_for_side
from .scratchpad_beginner_routing_two_wall_s_route import held_out_two_wall_specs_for_direction
from simulator.single_obstacle_env import MAP_HALF_SIZE_CELLS
from simulator.static_waypoint_env import FIXED_HEADING
from tests.helpers.router_qualification_harness import build_multi_wall_world, run_episode_general_router

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "models" / "generalized_waypoint_both_seed2_0051200.zip"

SINGLE_WALL_640M = 640_000_000
TWO_WALL_663M = 663_000_000
SINGLE_WALL_705M = 705_000_000
TWO_WALL_706M = 706_000_000


@dataclass(frozen=True)
class RoutingFixture:
    name: str
    pool_type: str  # "single_wall" | "two_wall"
    side_or_direction: str
    spec_seed: int
    n_pool: int  # pool size the fixture's index was drawn from (needed to regenerate identically)
    index: int
    expected_a: str
    expected_b: str
    source: str


FIXTURES: list[RoutingFixture] = [
    # -- 6 originally-diagnosed selector-instability cases --
    RoutingFixture("bridge_left_11", "single_wall", "left", SINGLE_WALL_640M, 30, 11, "collision", "collision",
                    "original selector-instability audit; large-index-jump collision, still fails under B"),
    RoutingFixture("bridge_left_15", "single_wall", "left", SINGLE_WALL_640M, 30, 15, "timeout", "success",
                    "original selector-instability audit; destination-flicker orbit, fixed under B"),
    RoutingFixture("step2_ltr_17", "two_wall", "left_then_right", TWO_WALL_663M, 30, 17, "collision", "success",
                    "original selector-instability audit; also the 'stable-target-still-collides' case under the rejected follower -- fixed under B"),
    RoutingFixture("step2_rtl_9", "two_wall", "right_then_left", TWO_WALL_663M, 30, 9, "timeout", "success",
                    "original selector-instability audit; destination-flicker orbit, fixed under B"),
    RoutingFixture("step2_rtl_17", "two_wall", "right_then_left", TWO_WALL_663M, 30, 17, "collision", "collision",
                    "original selector-instability audit; large-index-jump collision, still fails under B"),
    RoutingFixture("step2_rtl_19", "two_wall", "right_then_left", TWO_WALL_663M, 30, 19, "collision", "collision",
                    "original selector-instability audit; large-index-jump collision, still fails under B"),
    # -- planner search-budget exhaustion (identical regardless of selector) --
    RoutingFixture("step2_rtl_20_planner_budget", "two_wall", "right_then_left", TWO_WALL_663M, 30, 20,
                    "planner_failure_no_route_found", "planner_failure_no_route_found",
                    "40,000-expansion search budget exhausted; separate from selector choice, plan_route's own search"),
    # -- 19 remaining B failures on the fresh, untouched 705M/706M paired-test pool --
    RoutingFixture("fresh_swl_5", "single_wall", "left", SINGLE_WALL_705M, 40, 5, "collision", "collision", "fresh paired A/B test, 2026-08-14"),
    RoutingFixture("fresh_swl_13", "single_wall", "left", SINGLE_WALL_705M, 40, 13, "collision", "collision", "fresh paired A/B test, 2026-08-14"),
    RoutingFixture("fresh_swl_21", "single_wall", "left", SINGLE_WALL_705M, 40, 21, "collision", "collision", "fresh paired A/B test, 2026-08-14"),
    RoutingFixture("fresh_swr_11", "single_wall", "right", SINGLE_WALL_705M, 40, 11, "timeout", "timeout", "fresh paired A/B test, 2026-08-14 -- B's one remaining timeout (B had 1 total timeout in this pool, not 2)"),
    RoutingFixture("fresh_ltr_9", "two_wall", "left_then_right", TWO_WALL_706M, 100, 9, "collision", "collision", "fresh paired A/B test, 2026-08-14"),
    RoutingFixture("fresh_ltr_19", "two_wall", "left_then_right", TWO_WALL_706M, 100, 19, "collision", "collision", "fresh paired A/B test, 2026-08-14"),
    RoutingFixture("fresh_ltr_32", "two_wall", "left_then_right", TWO_WALL_706M, 100, 32, "collision", "collision", "fresh paired A/B test, 2026-08-14"),
    RoutingFixture("fresh_ltr_39_planner_budget", "two_wall", "left_then_right", TWO_WALL_706M, 100, 39,
                    "planner_failure_no_route_found", "planner_failure_no_route_found", "fresh paired A/B test, 2026-08-14"),
    RoutingFixture("fresh_ltr_75_planner_budget", "two_wall", "left_then_right", TWO_WALL_706M, 100, 75,
                    "planner_failure_no_route_found", "planner_failure_no_route_found", "fresh paired A/B test, 2026-08-14"),
    RoutingFixture("fresh_rtl_8", "two_wall", "right_then_left", TWO_WALL_706M, 100, 8, "collision", "collision", "fresh paired A/B test, 2026-08-14"),
    RoutingFixture("fresh_rtl_12", "two_wall", "right_then_left", TWO_WALL_706M, 100, 12, "collision", "collision", "fresh paired A/B test, 2026-08-14"),
    RoutingFixture("fresh_rtl_19", "two_wall", "right_then_left", TWO_WALL_706M, 100, 19, "collision", "collision", "fresh paired A/B test, 2026-08-14"),
    RoutingFixture("fresh_rtl_27", "two_wall", "right_then_left", TWO_WALL_706M, 100, 27, "collision", "collision", "fresh paired A/B test, 2026-08-14"),
    RoutingFixture("fresh_rtl_28", "two_wall", "right_then_left", TWO_WALL_706M, 100, 28, "collision", "collision", "fresh paired A/B test, 2026-08-14"),
    RoutingFixture("fresh_rtl_38", "two_wall", "right_then_left", TWO_WALL_706M, 100, 38, "collision", "collision", "fresh paired A/B test, 2026-08-14"),
    RoutingFixture("fresh_rtl_42", "two_wall", "right_then_left", TWO_WALL_706M, 100, 42, "collision", "collision", "fresh paired A/B test, 2026-08-14"),
    RoutingFixture("fresh_rtl_62", "two_wall", "right_then_left", TWO_WALL_706M, 100, 62, "collision", "collision", "fresh paired A/B test, 2026-08-14"),
    RoutingFixture("fresh_rtl_63_planner_budget", "two_wall", "right_then_left", TWO_WALL_706M, 100, 63,
                    "planner_failure_no_route_found", "planner_failure_no_route_found", "fresh paired A/B test, 2026-08-14"),
    RoutingFixture("fresh_rtl_67", "two_wall", "right_then_left", TWO_WALL_706M, 100, 67, "collision", "collision", "fresh paired A/B test, 2026-08-14"),
]


def _spec_and_world(fixture: RoutingFixture):
    if fixture.pool_type == "single_wall":
        specs = held_out_randomized_wall_specs_for_side(fixture.n_pool, gap_side=fixture.side_or_direction, seed=fixture.spec_seed)
        spec = specs[fixture.index]
        obstacle = spec.obstacle
        map_model, world = build_multi_wall_world([obstacle])
        distance_cells = obstacle.distance_cells
        heading_offset = spec.approach_heading_offset_radians
    else:
        specs = held_out_two_wall_specs_for_direction(fixture.n_pool, direction=fixture.side_or_direction, seed=fixture.spec_seed)
        spec = specs[fixture.index]
        wall1, wall2 = spec.wall1_obstacle_spec(), spec.wall2_obstacle_spec()
        map_model, world = build_multi_wall_world([wall1, wall2])
        distance_cells = spec.distance_cells
        heading_offset = spec.approach_heading_offset_radians

    cell_size = map_model.native_units_per_cell
    center = MAP_HALF_SIZE_CELLS
    start_native = map_model.layout_to_native(center, center)
    final_native = (
        start_native[0] + math.cos(FIXED_HEADING) * distance_cells * cell_size,
        start_native[1] + math.sin(FIXED_HEADING) * distance_cells * cell_size,
    )
    initial_heading = FIXED_HEADING + heading_offset
    return map_model, world, initial_heading, final_native


def main() -> None:
    checkpoint_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CHECKPOINT
    assert checkpoint_path.exists(), f"checkpoint missing: {checkpoint_path}"
    model = PPO.load(str(checkpoint_path), device="cpu")
    print(f"Replaying {len(FIXTURES)} routing regression fixtures against {checkpoint_path.name}\n")

    # Episode-seed base matches exactly what each source pool used when the
    # expected outcomes were recorded -- keyed by spec_seed, not guessed.
    episode_seed_base_by_spec_seed = {
        SINGLE_WALL_640M: 925_000_000,
        TWO_WALL_663M: 935_000_000,
        SINGLE_WALL_705M: 940_000_000,
        TWO_WALL_706M: 950_000_000,
    }

    results = []
    n_match_a = n_match_b = 0
    for fx in FIXTURES:
        map_model, world, initial_heading, final_native = _spec_and_world(fx)
        episode_seed = episode_seed_base_by_spec_seed[fx.spec_seed] + fx.index

        result_a = run_episode_general_router(model, map_model, world, initial_heading=initial_heading, final_native=final_native, seed=episode_seed, use_persistence_controller=False)
        map_model2, world2, _, _ = _spec_and_world(fx)  # fresh world instance for B (env is mutated during A's run)
        result_b = run_episode_general_router(model, map_model2, world2, initial_heading=initial_heading, final_native=final_native, seed=episode_seed, use_persistence_controller=True)

        match_a = result_a.outcome == fx.expected_a
        match_b = result_b.outcome == fx.expected_b
        n_match_a += int(match_a)
        n_match_b += int(match_b)
        print(f"  {fx.name:30s} A: expected={fx.expected_a:10s} actual={result_a.outcome:10s} {'PASS' if match_a else 'DIFFERENT'}   "
              f"B: expected={fx.expected_b:10s} actual={result_b.outcome:10s} {'PASS' if match_b else 'DIFFERENT'}")
        results.append({
            "name": fx.name, "source": fx.source,
            "expected_a": fx.expected_a, "actual_a": result_a.outcome, "match_a": match_a,
            "expected_b": fx.expected_b, "actual_b": result_b.outcome, "match_b": match_b,
        })

    print(f"\nA matches expected: {n_match_a}/{len(FIXTURES)}")
    print(f"B matches expected: {n_match_b}/{len(FIXTURES)}")
    if n_match_a < len(FIXTURES) or n_match_b < len(FIXTURES):
        print("\nSome fixtures differ from their recorded baseline -- this is INFORMATION, not")
        print("automatically good or bad. A DIFFERENT result could mean an improvement (e.g. a")
        print("new checkpoint fixed an old failure) or a new regression -- inspect before acting.")

    output_path = ROOT / "simulator" / "evaluations" / "routing_regression_fixtures_result.json"
    output_path.write_text(json.dumps({
        "checkpoint": str(checkpoint_path), "n_fixtures": len(FIXTURES),
        "a_matches": n_match_a, "b_matches": n_match_b, "results": results,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
