"""2026-08-11: single-obstacle fixed-waypoint GoalNav -- first obstacle
stage of the redirected main PPO navigation path, run after the
generalized open-map stage passed (100% success, 0% collision on 40
held-out heading/bearing/distance/position combinations, seeds 0/2/8).

Architecture reminder (explicit, per user instruction): this environment
tests LOCAL detour capability only -- a single fixed waypoint that may be
occluded by ONE nearby obstacle, not arbitrary long-range topology. The
intended pipeline is farming/target selection -> stable destination/route
subgoal -> this PPO local movement policy -> steering. This module must
not become a general router; procedural/multi-obstacle-topology maps are
explicitly deferred pending this stage's result.

Three mirror-verifiable base configurations, player at a FIXED heading
(0.0) and fixed grid-center spawn (heading/position generalization for
obstacle maps is deferred to a follow-up stage, matching the same
fixed-then-generalized progression the open-map task itself used --
rotating obstacle geometry per-episode would require rebuilding the
MapModel every reset, a real performance/complexity cost not worth paying
before knowing whether local detour behavior works AT ALL):

  gap_side="left":  wall blocks the RIGHT side of the direct corridor,
    leaving the LEFT side open -- correct policy must detour LEFT.
  gap_side="right": exact mirror of "left" (reflected about the direct
    path's centerline) -- correct policy must detour RIGHT.
  gap_side="none":  wall sits entirely off to one side of the direct
    path, never blocking it -- correct policy should go essentially
    STRAIGHT (small course corrections aside), not react to the obstacle.

Wall depth/width/offset-from-player and waypoint distance are all
randomized within each gap_side category (the "varied obstacle width/
position, varied waypoint distance" the user asked for), subject to a
hard reachability/no-immediate-contact validation before every episode.

Reward is GEODESIC path-distance reduction (not euclidean -- euclidean
would reward "closer in a straight line through the wall", exactly the
confound flagged for the earlier monster-target reward and now directly
relevant since real obstacles exist). Success terminates within the
waypoint radius; physical contact terminates as failure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import gymnasium as gym
import numpy as np

from .environment import RecordedFarmingEnv
from .map_model import MapModel
from .static_waypoint_env import COLLISION_TERMINAL_REWARD, EVENT_NONE, SUCCESS_RADIUS_CELLS, SUCCESS_TERMINAL_REWARD
from .world_model import MovementModel, RecordedWorldModel

GapSide = Literal["left", "right", "none"]
GAP_SIDES: tuple[GapSide, ...] = ("left", "right", "none")

MAP_HALF_SIZE_CELLS = 45
FIXED_HEADING = 0.0


@dataclass(frozen=True)
class ObstacleSpec:
    gap_side: GapSide
    distance_cells: float          # player-to-waypoint, straight-line cell count
    wall_offset_cells: int         # how many cells ahead of the player the wall starts
    wall_depth_cells: int          # wall thickness along the travel direction
    half_span_cells: int           # how far the wall extends from the corridor centerline
    straight_offset_cells: int = 0  # gap_side="none" only: how far off-centerline the wall sits


def sample_obstacle_spec(
    rng: np.random.Generator,
    *,
    gap_side: GapSide,
    distance_range: tuple[float, float] = (14.0, 24.0),
    wall_offset_range: tuple[int, int] = (5, 9),
    wall_depth_range: tuple[int, int] = (2, 5),
    half_span_range: tuple[int, int] = (4, 8),
    straight_offset_range: tuple[int, int] = (10, 14),
) -> ObstacleSpec:
    distance = float(rng.uniform(*distance_range))
    wall_offset = int(rng.integers(wall_offset_range[0], wall_offset_range[1] + 1))
    wall_depth = int(rng.integers(wall_depth_range[0], wall_depth_range[1] + 1))
    half_span = int(rng.integers(half_span_range[0], half_span_range[1] + 1))
    straight_offset = int(rng.integers(straight_offset_range[0], straight_offset_range[1] + 1))
    return ObstacleSpec(
        gap_side=gap_side, distance_cells=distance, wall_offset_cells=wall_offset,
        wall_depth_cells=wall_depth, half_span_cells=half_span, straight_offset_cells=straight_offset,
    )


# 2026-08-11: fixed, explicit per-side seed offsets -- NOT hash(gap_side),
# which is process-dependent (PYTHONHASHSEED randomizes str hashing by
# default), so an eval pool built that way is not reproducible run-to-run.
GAP_SIDE_SEED_OFFSET: dict[GapSide, int] = {"left": 0, "right": 1000, "none": 2000}


def held_out_obstacle_specs_for_side(n: int, *, gap_side: GapSide, seed: int, **kwargs) -> list[ObstacleSpec]:
    """Deterministic, side-specific held-out pool -- samples directly for
    ONE gap_side (no wasted generation, no hash()-based seed offset)."""
    rng = np.random.default_rng(seed + GAP_SIDE_SEED_OFFSET[gap_side])
    return [sample_obstacle_spec(rng, gap_side=gap_side, **kwargs) for _ in range(n)]


def held_out_obstacle_specs(n_per_side: int, *, seed: int, **kwargs) -> list[ObstacleSpec]:
    rng = np.random.default_rng(seed)
    specs: list[ObstacleSpec] = []
    for side in GAP_SIDES:
        specs.extend(sample_obstacle_spec(rng, gap_side=side, **kwargs) for _ in range(n_per_side))
    return specs


def _wall_cell_bounds(spec: ObstacleSpec, center: int) -> tuple[int, int, int, int]:
    """(x_start, x_end, y_start, y_end) inclusive layout-cell bounds of the
    wall for a given spec, player fixed at (center, center), heading=0
    (+cell_x is straight ahead). cell_x increases with native_x (forward);
    cell_y DECREASES as native_z increases, and +native_z is LEFT (see
    scratchpad_static_waypoint_symmetry_check.py's verified convention) --
    so decreasing cell_y is LEFT, increasing cell_y is RIGHT."""
    x_start = center + spec.wall_offset_cells
    x_end = x_start + spec.wall_depth_cells - 1
    if spec.gap_side == "left":
        # Block the RIGHT side (increasing cell_y from centerline), leave LEFT open.
        y_start, y_end = center, center + spec.half_span_cells - 1
    elif spec.gap_side == "right":
        # Exact mirror: block the LEFT side (decreasing cell_y), leave RIGHT open.
        y_start, y_end = center - spec.half_span_cells + 1, center
    else:
        # "none": wall sits entirely off to one side, never touching the
        # centerline (straight_offset_cells > 0 guarantees a gap).
        y_start = center + spec.straight_offset_cells
        y_end = y_start + spec.half_span_cells - 1
    return x_start, x_end, y_start, y_end


def build_single_obstacle_world(
    spec: ObstacleSpec, *, movement: tuple[MovementModel, ...], population: int = 1,
) -> tuple[MapModel, RecordedWorldModel]:
    size = MAP_HALF_SIZE_CELLS * 2 + 1
    arr = np.ones((size, size), dtype=bool)
    center = MAP_HALF_SIZE_CELLS
    x_start, x_end, y_start, y_end = _wall_cell_bounds(spec, center)
    x_start = max(0, x_start)
    x_end = min(size - 1, x_end)
    y_start = max(0, y_start)
    y_end = min(size - 1, y_end)
    arr[y_start : y_end + 1, x_start : x_end + 1] = False
    map_model = MapModel.from_arrays(arr)

    positions = tuple(map_model.layout_to_native(center, center) for _ in range(4))
    sections = tuple(positions for _ in range(3))
    world = RecordedWorldModel(
        schema_version=5, source_recordings=("single_obstacle",), section_count=2, hub_section=2,
        population_median=population, section_population_probabilities=(1 / 3, 1 / 3, 1 / 3),
        player_start_positions=(map_model.layout_to_native(center, center),),
        spawn_positions_by_section=sections,
        transition_probabilities=tuple((1 / 3, 1 / 3, 1 / 3) for _ in range(3)),
        respawn_delay_seconds=(2.0,), movement=movement, monster_speed_cells_per_second=0.0,
        frame_interval_seconds=0.2, native_units_per_cell=map_model.native_units_per_cell,
        recording_frame_interval_seconds=0.2, cast_step_seconds=0.8, cast_movement_seconds=0.2,
        respawn_model_mode="global_redistribution", respawn_delay_source="test",
    )
    return map_model, world


def make_single_obstacle_env(
    spec: ObstacleSpec, *, episode_steps: int, seed: int, movement: tuple[MovementModel, ...],
) -> RecordedFarmingEnv:
    map_model, world = build_single_obstacle_world(spec, movement=movement)
    env = RecordedFarmingEnv(world, map_model=map_model, episode_steps=episode_steps)
    env.reset(seed=seed)
    for actor in env.actors[1:]:
        actor.alive = False
    return env


def waypoint_reachable(env: RecordedFarmingEnv, waypoint_native_xz: tuple[float, float]) -> bool:
    """Mechanical reachability check: finite geodesic distance from the
    player's current cell to the waypoint's cell. Used both as a
    pre-training verification gate and as a live per-episode guard
    (per the user's explicit sampling rule: never spawn/place into
    unreachable geometry and misinterpret the resulting failure as a
    policy defect)."""
    player_cell = env.map.native_to_layout_cell(env.player_x, env.player_z)
    waypoint_cell = env.map.native_to_layout_cell(*waypoint_native_xz)
    if player_cell is None or waypoint_cell is None:
        return False
    field = env._geodesic_field(player_cell)
    return math.isfinite(field.get(waypoint_cell, math.inf))


class SingleObstacleWrapper(gym.Wrapper):
    """Places actor[0] as a fixed-for-the-episode waypoint straight ahead
    of the player (heading=0.0, matching the wall's axis-aligned
    construction), with a single obstacle (per ObstacleSpec) between them.
    Reward = per-tick GEODESIC distance reduction (not euclidean --
    obstacles exist, so a straight-line metric would reward moving toward
    the target through the wall). Terminates successfully within
    SUCCESS_RADIUS_CELLS, unsuccessfully on physical contact. Steering-
    only: event forced to NONE.

    `spec_source=None` picks a fresh random gap_side + spec every reset
    (required for training a single policy that must condition on the
    actual obstacle geometry, not memorize one answer)."""

    def __init__(self, env: Any, *, movement: tuple[MovementModel, ...], spec_source=None, rng: np.random.Generator | None = None):
        super().__init__(env)
        self._movement = movement
        self._spec_source = spec_source
        self._rng = rng or np.random.default_rng()
        self.obstacle_spec: ObstacleSpec | None = None
        self._prev_geodesic = 0.0
        self._prev_contacts = 0
        self.initial_geodesic_cells: float = 0.0

    def _geodesic_distance_to_waypoint(self) -> float:
        base_env = self.env.unwrapped
        player_cell = base_env.map.native_to_layout_cell(base_env.player_x, base_env.player_z)
        actor = base_env.actors[0]
        waypoint_cell = base_env.map.native_to_layout_cell(actor.x, actor.z)
        if player_cell is None or waypoint_cell is None:
            return math.inf
        field = base_env._geodesic_field(player_cell)
        return float(field.get(waypoint_cell, math.inf))

    def reset(self, **kwargs):
        base_env = self.env.unwrapped
        if self._spec_source is not None:
            self.obstacle_spec = self._spec_source(self._rng)
        # The obstacle geometry varies per episode, but RecordedFarmingEnv
        # builds its `map` once at construction -- rebuild and swap it in
        # BEFORE reset() so the very first observation already reflects
        # this episode's wall, not a stale one from whichever spec the env
        # happened to be constructed with.
        new_map_model, _world = build_single_obstacle_world(self.obstacle_spec, movement=self._movement)
        base_env.map = new_map_model
        obs, info = self.env.reset(**kwargs)
        base_env.heading = FIXED_HEADING
        cell_size = base_env.map.native_units_per_cell
        native_distance = self.obstacle_spec.distance_cells * cell_size
        base_env.actors[0].x = base_env.player_x + math.cos(FIXED_HEADING) * native_distance
        base_env.actors[0].z = base_env.player_z + math.sin(FIXED_HEADING) * native_distance
        base_env.actors[0].alive = True
        for actor in base_env.actors[1:]:
            actor.alive = False

        raw = base_env._observation()
        if obs.shape[0] > raw.shape[0]:
            obs = np.concatenate([raw, obs[raw.shape[0]:]])
        else:
            obs = raw
        self._prev_geodesic = self._geodesic_distance_to_waypoint()
        self.initial_geodesic_cells = self._prev_geodesic
        self._prev_contacts = int(info.get("contacts", 0)) if isinstance(info, dict) else 0
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.int64).copy()
        action[1] = EVENT_NONE
        obs, _reward, terminated, truncated, info = self.env.step(action)
        contacts = int(info.get("contacts", 0))
        contact_this_tick = contacts > self._prev_contacts
        self._prev_contacts = contacts
        geodesic = self._geodesic_distance_to_waypoint()
        progress = self._prev_geodesic - geodesic if math.isfinite(geodesic) and math.isfinite(self._prev_geodesic) else 0.0
        self._prev_geodesic = geodesic

        if contact_this_tick:
            return obs, COLLISION_TERMINAL_REWARD, True, truncated, info
        # GEODESIC proximity, not euclidean: with a wall present, being
        # euclidean-close can still mean "on the wrong side of the wall"
        # (e.g. adjacent cells separated by a multi-cell-thick blocked
        # band) -- using euclidean here would let success trigger without
        # the waypoint actually having been reached via a valid path.
        if geodesic <= SUCCESS_RADIUS_CELLS:
            return obs, SUCCESS_TERMINAL_REWARD, True, truncated, info
        return obs, progress, terminated, truncated, info
