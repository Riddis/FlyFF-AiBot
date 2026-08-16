"""2026-08-15: policy-independent geometry audit of the frozen 850M
obstacle_approach stratum, per explicit user instruction. The pool
generator places a wall in the world but never explicitly required the
direct player->monster segment to be blocked by it -- some episodes may
be effectively open approaches with a wall nearby rather than genuine
detours. Checked directly from the already-frozen manifest, no PPO
involved.

2026-08-15 CORRECTED (per user review of the first pass, which had two
real bugs):
  1. Used `start_heading=0.0` for every episode's plan_route() call,
     when the ACTUAL episode used a heading randomized by env.reset()
     (obstacle_approach has no heading_override). Now reads each
     episode's ACTUAL `initial_heading_radians` from the already-run
     corrected baseline's persisted raw_episodes
     (evaluations/monster_approach_baseline_850000000_result_corrected.
     json) -- the exact heading that episode really used, not a
     canonical stand-in.
  2. Called `route_length_cells / direct_distance_cells` a "detour
     ratio" -- but plan_route()'s own goal check only requires landing
     within `GOAL_RADIUS_CELLS` (2.5 cells) of the destination, not
     exactly on it (see plan_route's own docstring), so the returned
     route's own length is NOT the full path cost to the destination.
     Proven wrong directly: several ratios came out below 1.0 (e.g.
     0.767 for episode 3), which is geometrically impossible for a true
     A->B path cost (can never beat the straight-line distance). Fixed
     by adding the remaining route-endpoint->destination distance to
     route_length_cells before dividing -- this "total_path_cost" can
     no longer be below the direct distance by construction, and any
     residual sub-1.0 case remaining after this fix would itself be a
     real bug report, not swept under a mislabeled metric.

For each of the 20 obstacle_approach episodes: is the STRAIGHT player->
monster segment blocked (`_segment_clear`, the planner's own hard-reject
predicate)? What is the TRUE total path cost (route + remaining
endpoint->destination distance) relative to the direct distance?
`obstacle_approach[3]` (the one collision) is reported in full detail.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from scratchpad_monster_approach_baseline_pool import FULL_POOL_SPEC_SEED, build_monster_approach_world, load_manifest, spec_from_episode_dict
from simulator.kinodynamic_route_planner import _segment_clear, plan_route
from simulator.single_obstacle_env import MAP_HALF_SIZE_CELLS

ROOT = Path(__file__).resolve().parent


def _load_actual_headings() -> dict[int, float]:
    """index -> the ACTUAL initial_heading_radians that episode used in
    the corrected baseline run (env.reset()'s own randomization for
    obstacle_approach, since it has no heading_override)."""
    result_path = ROOT / "evaluations" / f"monster_approach_baseline_{FULL_POOL_SPEC_SEED}_result_corrected.json"
    data = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        ep["index"]: ep["initial_heading_radians"]
        for ep in data["raw_episodes"] if ep["category"] == "obstacle_approach"
    }


def main() -> None:
    manifest = load_manifest(ROOT / "evaluations" / f"monster_approach_baseline_pool_{FULL_POOL_SPEC_SEED}_manifest.json")
    stratum = manifest["strata"]["obstacle_approach"]
    actual_headings = _load_actual_headings()

    print(f"{'=' * 100}\nGEOMETRY AUDIT (CORRECTED): obstacle_approach ({len(stratum['accepted'])} episodes)\n{'=' * 100}")

    rows = []
    for index, episode_dict in enumerate(stratum["accepted"]):
        spec = spec_from_episode_dict("obstacle_approach", episode_dict)
        map_model, world, monster_positions = build_monster_approach_world(spec)
        center = MAP_HALF_SIZE_CELLS
        player_native = map_model.layout_to_native(center, center)
        target = monster_positions[0]
        cell_size = map_model.native_units_per_cell
        actual_heading = actual_headings[index]

        direct_clear = _segment_clear(map_model, player_native[0], player_native[1], target[0], target[1])
        direct_distance_cells = math.hypot(target[0] - player_native[0], target[1] - player_native[1]) / cell_size

        route = plan_route(
            map_model, start_x=player_native[0], start_z=player_native[1], start_heading=actual_heading,
            destination_x=target[0], destination_z=target[1],
        )
        if len(route) >= 2:
            route_length_cells = sum(
                math.hypot(route[i + 1].x - route[i].x, route[i + 1].z - route[i].z) / cell_size
                for i in range(len(route) - 1)
            )
            # plan_route's own goal check only requires landing within
            # GOAL_RADIUS_CELLS of the destination -- add the remaining
            # endpoint->destination distance so this is the TRUE total
            # path cost, not just the returned route's own partial length.
            remainder_cells = math.hypot(target[0] - route[-1].x, target[1] - route[-1].z) / cell_size
            total_path_cost_cells = route_length_cells + remainder_cells
            path_cost_ratio = total_path_cost_cells / direct_distance_cells if direct_distance_cells > 0 else None
        else:
            route_length_cells = remainder_cells = total_path_cost_cells = path_cost_ratio = None

        wall = spec.wall_specs[0]
        row = {
            "index": index, "wall": wall, "direct_segment_clear": direct_clear,
            "direct_distance_cells": round(direct_distance_cells, 2),
            "actual_initial_heading_deg": round(math.degrees(actual_heading), 2),
            "route_nodes": len(route),
            "route_length_cells": round(route_length_cells, 2) if route_length_cells is not None else None,
            "remainder_to_destination_cells": round(remainder_cells, 2) if remainder_cells is not None else None,
            "total_path_cost_cells": round(total_path_cost_cells, 2) if total_path_cost_cells is not None else None,
            "path_cost_ratio": round(path_cost_ratio, 3) if path_cost_ratio is not None else None,
        }
        rows.append(row)
        marker = "  <-- obstacle_approach[3], the one collision" if index == 3 else ""
        print(f"  [{index:2d}] direct_segment_clear={str(direct_clear):5s} direct_dist={row['direct_distance_cells']:6.2f} "
              f"heading={row['actual_initial_heading_deg']:7.2f} route_nodes={row['route_nodes']:3d} "
              f"path_cost={row['total_path_cost_cells']} path_cost_ratio={row['path_cost_ratio']}{marker}")

    n_blocked = sum(1 for r in rows if not r["direct_segment_clear"])
    n_clear = sum(1 for r in rows if r["direct_segment_clear"])
    ratios = [r["path_cost_ratio"] for r in rows if r["path_cost_ratio"] is not None]
    n_below_one = sum(1 for r in ratios if r < 0.999)
    print(f"\n{'=' * 100}")
    print(f"Direct segment BLOCKED (genuine obstruction requiring a detour): {n_blocked}/20")
    print(f"Direct segment CLEAR (wall present but does not block the direct line -- effectively an open approach): {n_clear}/20")
    if ratios:
        print(f"Corrected path_cost_ratio (true total path cost / direct distance): min={min(ratios):.3f} max={max(ratios):.3f} "
              f"mean={sum(ratios)/len(ratios):.3f}")
    print(f"Ratios below 1.0 (should be geometrically impossible now): {n_below_one}")
    print(f"{'=' * 100}")

    print("\nobstacle_approach[3] full detail (CORRECTED, actual heading used):")
    print(json.dumps(rows[3], indent=2, default=str))

    out_path = ROOT / "evaluations" / "monster_approach_obstacle_geometry_audit.json"
    out_path.write_text(json.dumps({"rows": rows, "n_blocked": n_blocked, "n_clear": n_clear}, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
