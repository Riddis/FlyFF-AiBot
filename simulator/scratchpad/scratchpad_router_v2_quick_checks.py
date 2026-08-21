"""2026-08-15: quick individual-episode sanity checks for `select_
persistent_waypoint_experimental_invalid_hop_guard` ("v2") before the
full development-pool validation, per explicit user instruction. Reuses
`run_episode_general_router` unmodified, monkeypatching only `scratchpad_
general_router_episode.select_persistent_waypoint` to v2 for the duration
of each check (never touches production). Checks:

  RTL8, RTL9 (812M dev pool, two_wall_right_then_left[8]/[9]): the
    original two episodes the whole investigation started from -- old
    any_fallback is _segment_clear=False, so v2 should behave like v1
    and repair both to success.
  episode 67 (706M paired A/B pool, two_wall_right_then_left[67]): the
    corrected-baseline regression episode -- reused here as a general
    v2 sanity check, outcome not predetermined by the invalid-hop
    mechanism specifically.
  SWR1 (830M, single_wall_right[1]): old any_fallback is _segment_clear=
    True, so v2's guard should be a no-op here and reproduce A's success.
"""
from __future__ import annotations

import math
from pathlib import Path

from stable_baselines3 import PPO

from . import scratchpad_general_router_episode as gre
from .scratchpad_beginner_navigation_mix_pools import DEV_POOL_SPEC_SEED, _reconstruct_single_wall_world, _reconstruct_two_wall_world, load_manifest
from .scratchpad_beginner_routing_two_wall_s_route import held_out_two_wall_specs_for_direction
from .scratchpad_general_router_episode import build_multi_wall_world, run_episode_general_router
from .scratchpad_router_patch_qualification_compare import DEV_POOL_EPISODE_SEED_BASE as SEED_830M
from .scratchpad_router_patch_qualification_pool import QUALIFICATION_SPEC_SEED
from navigation.kinodynamic_route_planner import select_persistent_waypoint_experimental_invalid_hop_guard
from simulator.single_obstacle_env import MAP_HALF_SIZE_CELLS
from simulator.static_waypoint_env import FIXED_HEADING

ROOT = Path(__file__).resolve().parents[2]
BASELINE_CHECKPOINT = ROOT / "models" / "generalized_waypoint_both_seed2_0051200.zip"

TWO_WALL_AB_SEED = 706_000_000
EPISODE67_SEED = 950_000_000 + 67


def _final_native_706(map_model, distance_cells, heading_offset):
    cell_size = map_model.native_units_per_cell
    start_native = map_model.layout_to_native(MAP_HALF_SIZE_CELLS, MAP_HALF_SIZE_CELLS)
    final_native = (
        start_native[0] + math.cos(FIXED_HEADING) * distance_cells * cell_size,
        start_native[1] + math.sin(FIXED_HEADING) * distance_cells * cell_size,
    )
    return final_native, FIXED_HEADING + heading_offset


def _seed_counter_for(manifest: dict, stratum_name: str, index: int) -> int:
    counter = 0
    for name, stratum in manifest["strata"].items():
        if name == "open":
            continue
        if name == stratum_name:
            return counter + index
        counter += len(stratum["accepted"])
    raise KeyError(stratum_name)


def main() -> None:
    model = PPO.load(str(BASELINE_CHECKPOINT), device="cpu")
    original = gre.select_persistent_waypoint
    gre.select_persistent_waypoint = select_persistent_waypoint_experimental_invalid_hop_guard
    try:
        # RTL8, RTL9 -- 812M dev pool
        dev_manifest = load_manifest(ROOT / "simulator" / "evaluations" / f"router_mix_dev_pool_{DEV_POOL_SPEC_SEED}_manifest.json")
        for index in (8, 9):
            episode = dev_manifest["strata"]["two_wall_right_then_left"]["accepted"][index]
            map_model, world, final_native, initial_heading = _reconstruct_two_wall_world(episode)
            result = run_episode_general_router(model, map_model, world, initial_heading=initial_heading, final_native=final_native, seed=812_600_000 + _seed_counter_for(dev_manifest, "two_wall_right_then_left", index))
            print(f"RTL{index} under v2: outcome={result.outcome} (expected: success)")

        # episode 67 -- 706M paired A/B pool
        specs = held_out_two_wall_specs_for_direction(100, direction="right_then_left", seed=TWO_WALL_AB_SEED)
        spec = specs[67]
        wall1, wall2 = spec.wall1_obstacle_spec(), spec.wall2_obstacle_spec()
        map_model, world = build_multi_wall_world([wall1, wall2])
        final_native, initial_heading = _final_native_706(map_model, spec.distance_cells, spec.approach_heading_offset_radians)
        result = run_episode_general_router(model, map_model, world, initial_heading=initial_heading, final_native=final_native, seed=EPISODE67_SEED)
        print(f"episode67 (two_wall_right_then_left[67], 706M) under v2: outcome={result.outcome}")

        # SWR1 -- 830M
        qual_manifest = load_manifest(ROOT / "simulator" / "evaluations" / f"router_mix_qualification_pool_{QUALIFICATION_SPEC_SEED}_manifest.json")
        episode = qual_manifest["strata"]["single_wall_right"]["accepted"][1]
        map_model, world, final_native, initial_heading = _reconstruct_single_wall_world(episode)
        result = run_episode_general_router(model, map_model, world, initial_heading=initial_heading, final_native=final_native, seed=SEED_830M + _seed_counter_for(qual_manifest, "single_wall_right", 1))
        print(f"SWR1 (single_wall_right[1], 830M) under v2: outcome={result.outcome} (expected: success, matching A -- old any_fallback here is already _segment_clear=True so v2 should be a no-op)")
    finally:
        gre.select_persistent_waypoint = original


if __name__ == "__main__":
    main()
