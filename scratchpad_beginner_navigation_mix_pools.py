"""2026-08-14: Beginner Navigation Training Mix, Parts 3 & 5 -- prevalidated,
manifest-frozen dev and final-confirmation pools.

Every obstacle candidate is validated (final_native in-bounds+traversable,
plan_route finds a route) BEFORE being admitted -- a policy-independent
geometry filter, not cherry-picking based on PPO outcomes (per the
approved plan). On rejection, resample ANOTHER candidate of the SAME
stratum (never cross-contaminate strata) and log the rejection + reason;
if a stratum can't reach its target count within MAX_ATTEMPTS_PER_STRATUM,
abort loudly -- that is a real generator problem, not something to hide.

The manifest stores fully MATERIALIZED spec parameters (not a seed+index
formula) so it is self-contained and immune to any future sampler-
implementation drift -- every checkpoint during training, and the one-shot
final confirmation, evaluate against the exact same frozen episodes
forever, regardless of what scratchpad_beginner_routing_*.py's sampler
code looks like by then.

GAP_SIDES has THREE values (left/right/none, simulator/single_obstacle_
env.py) -- the plan's "60 single-wall (30/side)" arithmetic assumed 2;
corrected here to N_PER_SIDE x 3 sides, same total intent.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from scratchpad_beginner_routing_randomized_walls import sample_randomized_wall_spec
from scratchpad_beginner_routing_two_wall_s_route import S_ROUTE_DIRECTIONS, TwoWallSpec, sample_two_wall_spec
from scratchpad_general_router_episode import build_multi_wall_world, run_episode_general_router, summarize_general_router
from simulator.kinodynamic_route_planner import plan_route
from simulator.router_waypoint_env import ObstacleEpisodeSpec, _final_native_valid
from simulator.single_obstacle_env import GAP_SIDES, MAP_HALF_SIZE_CELLS, ObstacleSpec
from simulator.static_waypoint_env import FIXED_HEADING, WaypointSpec, sample_generalized_spec

ROOT = Path(__file__).resolve().parent

DEV_POOL_SPEC_SEED = 812_000_000
FINAL_CONFIRMATION_SPEC_SEED = 820_000_000

DEV_N_PER_SIDE = 20         # x3 sides = 60 single-wall
DEV_N_PER_DIRECTION = 30    # x2 directions = 60 two-wall
DEV_N_OPEN = 20

FINAL_N_PER_SIDE = 17       # x3 sides = 51 single-wall
FINAL_N_PER_DIRECTION = 25  # x2 directions = 50 two-wall
FINAL_N_OPEN = 20

MAX_ATTEMPTS_PER_STRATUM = 2000

DISTANCE_RANGE = (8.0, 25.0)
POSITION_OFFSET_RADIUS_CELLS = 10.0


def _final_native_for(map_model, distance_cells: float, heading_offset: float) -> tuple[tuple[float, float], float]:
    cell_size = map_model.native_units_per_cell
    center = map_model.layout_to_native(MAP_HALF_SIZE_CELLS, MAP_HALF_SIZE_CELLS)
    final_native = (
        center[0] + math.cos(FIXED_HEADING) * distance_cells * cell_size,
        center[1] + math.sin(FIXED_HEADING) * distance_cells * cell_size,
    )
    return final_native, FIXED_HEADING + heading_offset


def _validate_route(map_model, distance_cells: float, heading_offset: float) -> bool:
    final_native, initial_heading = _final_native_for(map_model, distance_cells, heading_offset)
    if not _final_native_valid(map_model, final_native):
        return False
    center = map_model.layout_to_native(MAP_HALF_SIZE_CELLS, MAP_HALF_SIZE_CELLS)
    route = plan_route(
        map_model, start_x=center[0], start_z=center[1], start_heading=initial_heading,
        destination_x=final_native[0], destination_z=final_native[1],
    )
    return len(route) >= 2


def _build_single_wall_stratum(rng: np.random.Generator, gap_side: str, n: int) -> dict:
    accepted: list[dict] = []
    rejected: list[dict] = []
    attempts = 0
    while len(accepted) < n:
        attempts += 1
        if attempts > MAX_ATTEMPTS_PER_STRATUM:
            raise RuntimeError(
                f"single_wall[{gap_side}]: exhausted {MAX_ATTEMPTS_PER_STRATUM} attempts, "
                f"only {len(accepted)}/{n} valid episodes found -- real generator problem"
            )
        wall_spec = sample_randomized_wall_spec(rng, gap_side=gap_side)
        obstacle = wall_spec.obstacle
        map_model, _world = build_multi_wall_world([obstacle])
        if _validate_route(map_model, obstacle.distance_cells, wall_spec.approach_heading_offset_radians):
            accepted.append({
                "gap_side": gap_side, "wall_offset_cells": obstacle.wall_offset_cells,
                "wall_depth_cells": obstacle.wall_depth_cells, "half_span_cells": obstacle.half_span_cells,
                "distance_cells": obstacle.distance_cells, "straight_offset_cells": obstacle.straight_offset_cells,
                "approach_heading_offset_radians": wall_spec.approach_heading_offset_radians,
            })
        else:
            rejected.append({"gap_side": gap_side, "distance_cells": obstacle.distance_cells, "reason": "out_of_bounds_or_no_route"})
    return {"target": n, "attempts": attempts, "accepted": accepted, "rejected": rejected}


def _build_two_wall_stratum(rng: np.random.Generator, direction: str, n: int) -> dict:
    accepted: list[dict] = []
    rejected: list[dict] = []
    attempts = 0
    while len(accepted) < n:
        attempts += 1
        if attempts > MAX_ATTEMPTS_PER_STRATUM:
            raise RuntimeError(
                f"two_wall[{direction}]: exhausted {MAX_ATTEMPTS_PER_STRATUM} attempts, "
                f"only {len(accepted)}/{n} valid episodes found -- real generator problem"
            )
        spec = sample_two_wall_spec(rng, direction=direction)
        wall1, wall2 = spec.wall1_obstacle_spec(), spec.wall2_obstacle_spec()
        map_model, _world = build_multi_wall_world([wall1, wall2])
        if _validate_route(map_model, spec.distance_cells, spec.approach_heading_offset_radians):
            accepted.append({
                "first_gap_side": spec.first_gap_side, "wall1_offset_cells": spec.wall1_offset_cells,
                "wall1_depth_cells": spec.wall1_depth_cells, "wall1_half_span_cells": spec.wall1_half_span_cells,
                "wall_separation_cells": spec.wall_separation_cells, "wall2_depth_cells": spec.wall2_depth_cells,
                "wall2_half_span_cells": spec.wall2_half_span_cells, "distance_cells": spec.distance_cells,
                "approach_heading_offset_radians": spec.approach_heading_offset_radians,
            })
        else:
            rejected.append({"direction": direction, "distance_cells": spec.distance_cells, "reason": "out_of_bounds_or_no_route"})
    return {"target": n, "attempts": attempts, "accepted": accepted, "rejected": rejected}


def _build_open_stratum(rng: np.random.Generator, n: int) -> dict:
    accepted = []
    for _ in range(n):
        spec = sample_generalized_spec(rng, distance_range=DISTANCE_RANGE, position_offset_radius_cells=POSITION_OFFSET_RADIUS_CELLS)
        accepted.append({
            "heading": spec.heading, "bearing": spec.bearing, "distance": spec.distance,
            "position_offset": list(spec.position_offset),
        })
    return {"target": n, "attempts": n, "accepted": accepted, "rejected": []}


# 2026-08-14 MISTAKES.md: this module previously derived per-stratum seed
# offsets via Python's built-in hash() on strings/tuples, which is
# randomized per-process by default (PEP 456) -- confirmed empirically
# (two `python -c` invocations of hash(("single_wall","left")) in the same
# session returned different values). That made manifest generation
# internally consistent within one process run but NOT reproducible across
# sessions, defeating the entire point of a "deterministically regenerate
# from a declared seed" pool. Fixed numeric stream IDs below are stable
# across any Python process, forever. The already-frozen 812M/820M
# manifests are NOT regenerated (their materialized JSON is the canonical
# truth) -- this fix only matters for any future manifest generation.
_STRATUM_STREAM_ID: dict[str, int] = {
    "single_wall_left": 1, "single_wall_right": 2, "single_wall_none": 3,
    "two_wall_left_then_right": 4, "two_wall_right_then_left": 5,
    "open": 6,
}


def build_manifest(spec_seed: int, *, n_per_side: int, n_per_direction: int, n_open: int) -> dict:
    manifest: dict[str, Any] = {"spec_seed": spec_seed, "strata": {}}
    for side in GAP_SIDES:
        rng = np.random.default_rng(spec_seed + _STRATUM_STREAM_ID[f"single_wall_{side}"])
        manifest["strata"][f"single_wall_{side}"] = _build_single_wall_stratum(rng, str(side), n_per_side)
    for direction in S_ROUTE_DIRECTIONS:
        rng = np.random.default_rng(spec_seed + _STRATUM_STREAM_ID[f"two_wall_{direction}"])
        manifest["strata"][f"two_wall_{direction}"] = _build_two_wall_stratum(rng, direction, n_per_direction)
    rng = np.random.default_rng(spec_seed + _STRATUM_STREAM_ID["open"])
    manifest["strata"]["open"] = _build_open_stratum(rng, n_open)
    return manifest


def save_manifest(manifest: dict, path: Path) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total_accepted = sum(len(s["accepted"]) for s in manifest["strata"].values())
    total_rejected = sum(len(s["rejected"]) for s in manifest["strata"].values())
    print(f"Saved manifest to {path} -- {total_accepted} accepted episodes, {total_rejected} rejected attempts across all strata")
    for name, stratum in manifest["strata"].items():
        if stratum["rejected"]:
            print(f"  {name}: {stratum['attempts']} attempts for {stratum['target']} target -- {len(stratum['rejected'])} rejected")


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# -- reconstruction + evaluation against a frozen manifest -----------------

def _reconstruct_single_wall_world(episode: dict):
    obstacle = ObstacleSpec(
        gap_side=episode["gap_side"], distance_cells=episode["distance_cells"],
        wall_offset_cells=episode["wall_offset_cells"], wall_depth_cells=episode["wall_depth_cells"],
        half_span_cells=episode["half_span_cells"], straight_offset_cells=episode["straight_offset_cells"],
    )
    map_model, world = build_multi_wall_world([obstacle])
    final_native, initial_heading = _final_native_for(map_model, episode["distance_cells"], episode["approach_heading_offset_radians"])
    return map_model, world, final_native, initial_heading


def _reconstruct_two_wall_world(episode: dict):
    spec = TwoWallSpec(
        first_gap_side=episode["first_gap_side"], wall1_offset_cells=episode["wall1_offset_cells"],
        wall1_depth_cells=episode["wall1_depth_cells"], wall1_half_span_cells=episode["wall1_half_span_cells"],
        wall_separation_cells=episode["wall_separation_cells"], wall2_depth_cells=episode["wall2_depth_cells"],
        wall2_half_span_cells=episode["wall2_half_span_cells"], distance_cells=episode["distance_cells"],
        approach_heading_offset_radians=episode["approach_heading_offset_radians"],
    )
    wall1, wall2 = spec.wall1_obstacle_spec(), spec.wall2_obstacle_spec()
    map_model, world = build_multi_wall_world([wall1, wall2])
    final_native, initial_heading = _final_native_for(map_model, spec.distance_cells, spec.approach_heading_offset_radians)
    return map_model, world, final_native, initial_heading


def eval_obstacle_manifest(model, manifest: dict, *, episode_seed_base: int, selector_fn=None) -> dict:
    """Evaluates `model` against every single_wall/two_wall episode in a
    frozen manifest (built by build_manifest/save_manifest). Returns
    {stratum_name: [GeneralRouterEpisodeResult, ...]} plus a combined
    summary. Never touches the manifest's "open" stratum (that's evaluated
    separately via the existing eval_held_out / StaticWaypointWrapper
    path, matching Part 4 gate 1's convention exactly).

    `selector_fn` (2026-08-15, per explicit user instruction for 840M
    qualification): passed through explicitly to run_episode_general_
    router's own `selector_fn` parameter when given, so a caller can A/B
    a specific selector variant (e.g. `select_persistent_waypoint_
    experimental_invalid_hop_guard`) without monkeypatching any module's
    `select_persistent_waypoint` name. `None` (default) omits the kwarg
    entirely, so run_episode_general_router's own default (the qualified
    production selector) applies -- unchanged behavior for every existing
    caller."""
    per_stratum_results: dict[str, list] = {}
    seed_counter = 0
    for name, stratum in manifest["strata"].items():
        if name == "open":
            continue
        results = []
        for episode in stratum["accepted"]:
            if name.startswith("single_wall_"):
                map_model, world, final_native, initial_heading = _reconstruct_single_wall_world(episode)
            else:
                map_model, world, final_native, initial_heading = _reconstruct_two_wall_world(episode)
            kwargs = {} if selector_fn is None else {"selector_fn": selector_fn}
            result = run_episode_general_router(
                model, map_model, world, initial_heading=initial_heading, final_native=final_native,
                seed=episode_seed_base + seed_counter, **kwargs,
            )
            seed_counter += 1
            results.append(result)
        per_stratum_results[name] = results

    all_results = [r for results in per_stratum_results.values() for r in results]
    combined_collision_indices = {
        (name, i) for name, results in per_stratum_results.items() for i, r in enumerate(results) if r.outcome == "collision"
    }
    combined_planner_failure_indices = {
        (name, i) for name, results in per_stratum_results.items() for i, r in enumerate(results)
        if r.outcome == "planner_failure_no_route_found"
    }
    combined_timeout_indices = {
        (name, i) for name, results in per_stratum_results.items() for i, r in enumerate(results) if r.outcome == "timeout"
    }
    return {
        "per_stratum": {name: summarize_general_router(results) for name, results in per_stratum_results.items()},
        "combined_summary": summarize_general_router(all_results),
        "collision_episode_keys": sorted(str(k) for k in combined_collision_indices),
        "planner_failure_episode_keys": sorted(str(k) for k in combined_planner_failure_indices),
        "timeout_episode_keys": sorted(str(k) for k in combined_timeout_indices),
        "n_total": len(all_results),
    }


def main() -> None:
    dev_path = ROOT / "evaluations" / f"router_mix_dev_pool_{DEV_POOL_SPEC_SEED}_manifest.json"
    final_path = ROOT / "evaluations" / f"router_mix_final_pool_{FINAL_CONFIRMATION_SPEC_SEED}_manifest.json"

    print(f"{'=' * 90}\nBuilding dev pool manifest (spec_seed={DEV_POOL_SPEC_SEED})\n{'=' * 90}")
    dev_manifest = build_manifest(DEV_POOL_SPEC_SEED, n_per_side=DEV_N_PER_SIDE, n_per_direction=DEV_N_PER_DIRECTION, n_open=DEV_N_OPEN)
    save_manifest(dev_manifest, dev_path)

    print(f"\n{'=' * 90}\nBuilding final-confirmation pool manifest (spec_seed={FINAL_CONFIRMATION_SPEC_SEED})\n{'=' * 90}")
    final_manifest = build_manifest(FINAL_CONFIRMATION_SPEC_SEED, n_per_side=FINAL_N_PER_SIDE, n_per_direction=FINAL_N_PER_DIRECTION, n_open=FINAL_N_OPEN)
    save_manifest(final_manifest, final_path)

    print(f"\n{'=' * 90}\nBoth manifests frozen. No PPO checkpoint has been run against either yet.\n{'=' * 90}")


if __name__ == "__main__":
    main()
