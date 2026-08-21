"""2026-08-15: fresh, dedicated deterministic pool for the post-router-fix
"complete bot" baseline, per explicit user instruction. Decoupled from
the canonical curriculum_manifests system (those pools answer a
different question -- graduation gates for the canonical training
lineage -- and would muddy what THIS baseline means if reused).

Six strata (~20 each, 120 total), balanced across what the router-only
820M/840M qualification pools deliberately did NOT cover (pure static-
waypoint wall-routing): live monsters, target succession across multiple
kills, target-switch pressure, longer routes with real monster drift,
and turning/recovery from awkward starting orientations.

  open_control      -- 1 monster, 10-24 cells, +-35deg starting bearing.
                        Sanity/control stratum -- NOT intentionally hard.
  awkward_heading    -- 1 monster, 10-24 cells, player heading forced
                        120-180deg away from the true bearing to it.
                        Tests raw turning/recovery (RecoveryController
                        deliberately NOT used in the primary baseline).
  obstacle_approach  -- 1 monster behind ONE wall, geometry broader/
                        deeper than the smoke pool but still geometry-
                        prevalidated and clearly solvable -- not another
                        router torture test (the router already had its
                        exam: 840M/820M).
  long_route         -- 1 monster near the longer practical end of the
                        map (within MAP_HALF_SIZE_CELLS margin), giving
                        real wander time for meaningful drift. Monster
                        speed is the SAME value used for every stratum
                        (0.15 cells/sec, within simulator/synthetic.py's
                        own fitted-from-recordings 0.0-0.18 range) --
                        never artificially accelerated just for this
                        stratum.
  competing_targets  -- 3-5 monsters; the two best INITIAL candidates by
                        distance are deliberately placed within ~1.5
                        cells of each other (well under the environment's
                        own _TARGET_HYSTERESIS_MARGIN_CELLS=3.0), so a
                        switch is a real possibility the native
                        hysteresis has to resolve -- not forced, not
                        guaranteed, and pool acceptance never depends on
                        whether a switch actually occurs.
  multi_kill_farming -- 4-6 monsters at substantially different bearings/
                        ranges. Runs until KILL_COUNT_TARGET (3) kills or
                        the tick budget, replanning only when the native
                        target selection's actor ID changes. The key
                        end-to-end farming stratum.

Every candidate is geometry-prevalidated (plan_route finds a route from
player start to the FIRST target's spawn position) before acceptance --
policy-independent, same discipline as every router-investigation pool.
No PPO/outcome-based resampling, no requirement that a switch actually
occur. Same-stratum resampling on rejection. Manifest stores fully
MATERIALIZED spec parameters, not a seed+index formula.

FULL_POOL_SPEC_SEED (850_000_000) is the real baseline pool. SMOKE_POOL_
SPEC_SEED (850_100_000) is smoke-only and is NEVER mixed into it, per
explicit instruction.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from navigation.kinodynamic_route_planner import plan_route
from simulator.map_model import MapModel
from simulator.single_obstacle_env import GAP_SIDES, GapSide, MAP_HALF_SIZE_CELLS, ObstacleSpec, _wall_cell_bounds
from simulator.static_waypoint_env import SYMMETRIC_MOVEMENT
from simulator.world_model import RecordedWorldModel

ROOT = Path(__file__).resolve().parents[2]

SMOKE_POOL_SPEC_SEED = 850_100_000  # smoke-only, never mixed into the baseline pool
FULL_POOL_SPEC_SEED = 850_000_000   # the real baseline pool

CATEGORIES: tuple[str, ...] = (
    "open_control", "awkward_heading", "obstacle_approach", "long_route", "competing_targets", "multi_kill_farming",
)

# monster_speed_cells_per_second: NOT set per-stratum. simulator/synthetic.py
# (the real fitted-from-human-recordings curriculum generator) uses
# rng.uniform(0.0, 0.18) -- confirmed by direct source read. 0.15 is a
# single representative value near the top of that realistic range, used
# UNIFORMLY across every stratum here (including long_route) -- never
# artificially accelerated just to manufacture drift.
MONSTER_SPEED_CELLS_PER_SECOND = 0.15

MONSTER_DISTANCE_RANGE = (10.0, 24.0)
LONG_ROUTE_DISTANCE_RANGE = (32.0, 40.0)  # near MAP_HALF_SIZE_CELLS=45's practical margin, leaving room for wall geometry/plan_route search
BEARING_OFFSET_RANGE_RADIANS = (-math.radians(35.0), math.radians(35.0))
AWKWARD_HEADING_OFFSET_RANGE_RADIANS = (math.radians(120.0), math.radians(180.0))  # magnitude; sign randomized

COMPETING_N_RANGE = (3, 5)
COMPETING_PRIMARY_DISTANCE_RANGE = (10.0, 20.0)
COMPETING_SECONDARY_DELTA_RANGE = (0.3, 1.5)   # cells closer/farther than primary -- well under the 3.0-cell hysteresis margin
COMPETING_DECOY_DISTANCE_RANGE = (22.0, 32.0)  # clearly non-competitive
COMPETING_BEARING_SPREAD_RADIANS = math.radians(70.0)

MULTI_KILL_N_RANGE = (4, 6)
MULTI_KILL_DISTANCE_RANGE = (8.0, 30.0)
KILL_COUNT_TARGET = 3  # mirrors scratchpad_monster_approach_baseline_eval.KILL_COUNT_TARGET

# obstacle_approach: "somewhat broader/deeper than smoke, but still...
# clearly solvable" -- widened from the smoke pool's ranges, not pushed to
# an extreme (that would just be another router torture test).
WALL_OFFSET_RANGE = (5, 10)
WALL_DEPTH_RANGE = (3, 5)
WALL_HALF_SPAN_RANGE = (5, 9)

MAX_ATTEMPTS_PER_CATEGORY = 800

_STRATUM_STREAM_ID: dict[str, int] = {
    "open_control": 1, "awkward_heading": 2, "obstacle_approach": 3,
    "long_route": 4, "competing_targets": 5, "multi_kill_farming": 6,
}


@dataclass(frozen=True)
class MonsterApproachSpec:
    category: str
    monster_offsets: tuple[tuple[float, float], ...]   # (distance_cells, bearing_offset_radians) relative to native +x ("forward")
    wall_specs: tuple[dict, ...]                        # ObstacleSpec field-dicts, "forward"-relative (heading=0 convention)
    heading_override_radians: float | None              # None = let env.reset() randomize naturally; set = force this heading post-reset


def _sample_walls(rng: np.random.Generator, n: int) -> tuple[dict, ...]:
    walls = []
    for _ in range(n):
        gap_side: GapSide = GAP_SIDES[int(rng.integers(0, len(GAP_SIDES)))]
        walls.append({
            "gap_side": gap_side,
            "wall_offset_cells": int(rng.integers(*WALL_OFFSET_RANGE)),
            "wall_depth_cells": int(rng.integers(*WALL_DEPTH_RANGE)),
            "half_span_cells": int(rng.integers(*WALL_HALF_SPAN_RANGE)),
            "straight_offset_cells": int(rng.integers(1, 4)) if gap_side == "none" else 0,
        })
    return tuple(walls)


def sample_monster_approach_spec(rng: np.random.Generator, *, category: str) -> MonsterApproachSpec:
    if category == "open_control":
        distance = float(rng.uniform(*MONSTER_DISTANCE_RANGE))
        bearing = float(rng.uniform(*BEARING_OFFSET_RANGE_RADIANS))
        return MonsterApproachSpec(category, ((distance, bearing),), (), None)

    if category == "awkward_heading":
        distance = float(rng.uniform(*MONSTER_DISTANCE_RANGE))
        bearing = float(rng.uniform(*BEARING_OFFSET_RANGE_RADIANS))
        magnitude = float(rng.uniform(*AWKWARD_HEADING_OFFSET_RANGE_RADIANS))
        sign = 1.0 if rng.random() < 0.5 else -1.0
        heading_override = bearing + sign * magnitude
        return MonsterApproachSpec(category, ((distance, bearing),), (), heading_override)

    if category == "obstacle_approach":
        distance = float(rng.uniform(*MONSTER_DISTANCE_RANGE))
        bearing = float(rng.uniform(*BEARING_OFFSET_RANGE_RADIANS))
        walls = _sample_walls(rng, 1)
        return MonsterApproachSpec(category, ((distance, bearing),), walls, None)

    if category == "long_route":
        distance = float(rng.uniform(*LONG_ROUTE_DISTANCE_RANGE))
        bearing = float(rng.uniform(*BEARING_OFFSET_RANGE_RADIANS))
        return MonsterApproachSpec(category, ((distance, bearing),), (), None)

    if category == "competing_targets":
        n = int(rng.integers(*COMPETING_N_RANGE, endpoint=True))
        primary_distance = float(rng.uniform(*COMPETING_PRIMARY_DISTANCE_RANGE))
        primary_bearing = float(rng.uniform(-math.pi * 0.5, math.pi * 0.5))
        delta = float(rng.uniform(*COMPETING_SECONDARY_DELTA_RANGE))
        secondary_distance = primary_distance + (delta if rng.random() < 0.5 else -delta)
        # Distinct bearing (never stacked on the primary), still nearby.
        secondary_bearing = primary_bearing + float(rng.uniform(0.35, COMPETING_BEARING_SPREAD_RADIANS)) * (1.0 if rng.random() < 0.5 else -1.0)
        offsets = [(primary_distance, primary_bearing), (max(2.0, secondary_distance), secondary_bearing)]
        for _ in range(n - 2):
            decoy_distance = float(rng.uniform(*COMPETING_DECOY_DISTANCE_RANGE))
            decoy_bearing = float(rng.uniform(-math.pi, math.pi))
            offsets.append((decoy_distance, decoy_bearing))
        return MonsterApproachSpec(category, tuple(offsets), (), None)

    if category == "multi_kill_farming":
        n = int(rng.integers(*MULTI_KILL_N_RANGE, endpoint=True))
        offsets = []
        for _ in range(n):
            distance = float(rng.uniform(*MULTI_KILL_DISTANCE_RANGE))
            bearing = float(rng.uniform(-math.pi, math.pi))
            offsets.append((distance, bearing))
        return MonsterApproachSpec(category, tuple(offsets), (), None)

    raise ValueError(f"unknown category: {category}")


def build_monster_approach_world(spec: MonsterApproachSpec) -> tuple[MapModel, RecordedWorldModel, tuple[tuple[float, float], ...]]:
    """Returns (map_model, world, monster_native_positions). Player always
    starts at map center; "forward" (native +x, heading 0.0) is the
    reference direction for both wall placement and monster bearings, so
    reset()'s own heading randomization must be overridden by the caller
    to reproduce awkward_heading specs (see run_monster_approach_episode)."""
    size = MAP_HALF_SIZE_CELLS * 2 + 1
    arr = np.ones((size, size), dtype=bool)
    center = MAP_HALF_SIZE_CELLS
    for wall_dict in spec.wall_specs:
        obstacle = ObstacleSpec(
            gap_side=wall_dict["gap_side"], distance_cells=0.0,
            wall_offset_cells=wall_dict["wall_offset_cells"], wall_depth_cells=wall_dict["wall_depth_cells"],
            half_span_cells=wall_dict["half_span_cells"], straight_offset_cells=wall_dict["straight_offset_cells"],
        )
        x_start, x_end, y_start, y_end = _wall_cell_bounds(obstacle, center)
        x_start, x_end = max(0, x_start), min(size - 1, x_end)
        y_start, y_end = max(0, y_start), min(size - 1, y_end)
        arr[y_start : y_end + 1, x_start : x_end + 1] = False
    map_model = MapModel.from_arrays(arr)
    cell_size = map_model.native_units_per_cell
    player_native = map_model.layout_to_native(center, center)

    monster_positions = []
    for distance_cells, bearing in spec.monster_offsets:
        mx = player_native[0] + math.cos(bearing) * distance_cells * cell_size
        mz = player_native[1] + math.sin(bearing) * distance_cells * cell_size
        monster_positions.append((mx, mz))
    monster_positions = tuple(monster_positions)

    # section_count=0 divides by zero inside RecordedFarmingEnv._cast_eva's
    # death_section bookkeeping (map_model.section() divides by
    # section_count) -- confirmed directly (ZeroDivisionError) during smoke
    # testing. Mirror build_multi_wall_world's proven-working shape instead
    # (section_count=2, hub_section=2, 3 equal-probability sections) and
    # put the FULL N-point candidate list into all three sections
    # identically -- _sample_spawn_position (simulator/environment.py)
    # rejection-samples WITH REPLACEMENT among unoccupied candidates, so N
    # population draws against N available candidates reliably exhaust all
    # N points regardless of which section a given draw picks.
    n = len(monster_positions)
    sections = tuple(monster_positions for _ in range(3))
    world = RecordedWorldModel(
        schema_version=5, source_recordings=(f"monster_approach_{spec.category}",),
        section_count=2, hub_section=2,
        population_median=n, section_population_probabilities=(1 / 3, 1 / 3, 1 / 3),
        player_start_positions=(player_native,),
        spawn_positions_by_section=sections,
        transition_probabilities=tuple((1 / 3, 1 / 3, 1 / 3) for _ in range(3)),
        respawn_delay_seconds=(9999.0,),  # effectively no respawn within episode budget
        movement=SYMMETRIC_MOVEMENT,
        monster_speed_cells_per_second=MONSTER_SPEED_CELLS_PER_SECOND,
        frame_interval_seconds=0.2, native_units_per_cell=cell_size,
        recording_frame_interval_seconds=0.2, cast_step_seconds=0.8, cast_movement_seconds=0.2,
        respawn_model_mode="global_redistribution", respawn_delay_source="test",
    )
    return map_model, world, monster_positions


def _validate_candidate(spec: MonsterApproachSpec) -> bool:
    map_model, world, monster_positions = build_monster_approach_world(spec)
    center = MAP_HALF_SIZE_CELLS
    player_native = map_model.layout_to_native(center, center)
    start_heading = spec.heading_override_radians if spec.heading_override_radians is not None else 0.0
    # Geometry-only prevalidation against the FIRST (nearest-declared)
    # target -- policy-independent, matches every other pool in this
    # investigation's discipline. Does not require a switch, does not
    # inspect any PPO outcome.
    first_target = monster_positions[0]
    route = plan_route(
        map_model, start_x=player_native[0], start_z=player_native[1], start_heading=start_heading,
        destination_x=first_target[0], destination_z=first_target[1],
    )
    return len(route) >= 2


def _build_category(rng: np.random.Generator, category: str, n: int) -> dict:
    accepted: list[dict] = []
    rejected: list[dict] = []
    attempts = 0
    while len(accepted) < n:
        attempts += 1
        if attempts > MAX_ATTEMPTS_PER_CATEGORY:
            raise RuntimeError(f"{category}: exhausted {MAX_ATTEMPTS_PER_CATEGORY} attempts, only {len(accepted)}/{n} valid")
        spec = sample_monster_approach_spec(rng, category=category)
        if _validate_candidate(spec):
            accepted.append({
                "category": spec.category,
                "monster_offsets": [list(o) for o in spec.monster_offsets],
                "wall_specs": list(spec.wall_specs),
                "heading_override_radians": spec.heading_override_radians,
            })
        else:
            rejected.append({"category": category, "reason": "no_route_found"})
    return {"target": n, "attempts": attempts, "accepted": accepted, "rejected": rejected}


def build_manifest(spec_seed: int, *, n_per_category: dict[str, int]) -> dict:
    manifest: dict = {"spec_seed": spec_seed, "strata": {}}
    for category in CATEGORIES:
        rng = np.random.default_rng(spec_seed + _STRATUM_STREAM_ID[category])
        manifest["strata"][category] = _build_category(rng, category, n_per_category[category])
    return manifest


def save_manifest(manifest: dict, path: Path) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total = sum(len(s["accepted"]) for s in manifest["strata"].values())
    total_rejected = sum(len(s["rejected"]) for s in manifest["strata"].values())
    print(f"Saved manifest to {path} -- {total} accepted episodes, {total_rejected} rejected attempts")
    for name, stratum in manifest["strata"].items():
        if stratum["rejected"]:
            print(f"  {name}: {stratum['attempts']} attempts for {stratum['target']} target -- {len(stratum['rejected'])} rejected")


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def spec_from_episode_dict(category: str, episode: dict) -> MonsterApproachSpec:
    return MonsterApproachSpec(
        category=category,
        monster_offsets=tuple(tuple(o) for o in episode["monster_offsets"]),
        wall_specs=tuple(episode["wall_specs"]),
        heading_override_radians=episode["heading_override_radians"],
    )


def main() -> None:
    print(f"{'=' * 90}\nBuilding FULL monster-approach baseline pool (spec_seed={FULL_POOL_SPEC_SEED})\n{'=' * 90}")
    n_per_category = {c: 20 for c in CATEGORIES}
    manifest = build_manifest(FULL_POOL_SPEC_SEED, n_per_category=n_per_category)
    path = ROOT / "simulator" / "evaluations" / f"monster_approach_baseline_pool_{FULL_POOL_SPEC_SEED}_manifest.json"
    save_manifest(manifest, path)
    total = sum(len(s["accepted"]) for s in manifest["strata"].values())
    print(f"\nTotal baseline episodes: {total}.")


if __name__ == "__main__":
    main()
