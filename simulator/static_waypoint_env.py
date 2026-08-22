"""2026-08-10: static-waypoint symmetry diagnostic (Phase E of the PPO
mechanics audit, run only after the bandit micro-test (C/D) passed).

No monsters, no respawns, no target selector -- a single controllable
actor represents a FIXED-for-the-episode waypoint, placed at a controlled
angle/distance relative to a FIXED heading set at reset. Tests whether PPO
learns genuine geometry-conditioned steering (LEFT for a left waypoint,
RIGHT for a right waypoint, STRAIGHT for a straight-ahead waypoint) in the
smallest possible environment that still uses the real steering
target-geometry feature pipeline -- isolating this from monster/target-
selection semantics AND from the real training curriculum's movement-model
turn-rate asymmetry.

CONFOUND REMOVED: measured 2026-08-10 that the real training curriculum's
movement model has LEFT turning ~15% more per tick than RIGHT on average
(folded-normal mean |turn|: LEFT=0.229 rad/13.1deg vs RIGHT=0.199 rad/
11.4deg -- see simulator/run_logs/archive/OVERNIGHT_20260809_PIPELINE.md). This module's
`build_symmetric_open_world` uses an explicitly equal-magnitude,
zero-noise (deterministic) turn model instead, so any handedness this
experiment reproduces cannot be blamed on that asymmetry -- it would have
to come from the network/optimization dynamics themselves.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Literal

import gymnasium as gym
import numpy as np

from .environment import RecordedFarmingEnv
from .map_model import MapModel
from .world_model import MovementModel, RecordedWorldModel

WaypointCondition = Literal["left30", "right30", "straight"]
CONDITIONS: tuple[WaypointCondition, ...] = ("left30", "right30", "straight")


@dataclass(frozen=True)
class WaypointSpec:
    """One generalized episode's placement, per the 2026-08-11
    generalization stage (randomized heading/position/bearing/distance,
    not just the original 3 fixed angles at a fixed 15-cell distance).

    `heading`: the player's fixed-for-the-episode heading (radians).
    `bearing`: waypoint direction RELATIVE to heading (radians) -- 0 is
      straight ahead, positive is left (matches the sin(rel_angle)>0
      convention verified in scratchpad_static_waypoint_symmetry_check.py).
    `distance`: waypoint distance from the player, in cells.
    `position_offset`: (dx, dz) in cells, added to wherever the underlying
      env naturally spawned the player -- map-agnostic (does not assume
      the open map's specific center), so the same sampler works once
      obstacle maps are introduced.
    """
    heading: float
    bearing: float
    distance: float
    position_offset: tuple[float, float] = (0.0, 0.0)


def sample_generalized_spec(
    rng: np.random.Generator,
    *,
    distance_range: tuple[float, float] = (8.0, 25.0),
    position_offset_radius_cells: float = 10.0,
) -> WaypointSpec:
    """Continuous heading/bearing/distance/position sampler for the
    generalization stage -- replaces the original 3-discrete-angle,
    fixed-15-cell-distance, fixed-heading, fixed-position task."""
    heading = float(rng.uniform(-math.pi, math.pi))
    bearing = float(rng.uniform(-math.pi, math.pi))
    distance = float(rng.uniform(*distance_range))
    offset_radius = float(rng.uniform(0.0, position_offset_radius_cells))
    offset_angle = float(rng.uniform(-math.pi, math.pi))
    position_offset = (offset_radius * math.cos(offset_angle), offset_radius * math.sin(offset_angle))
    return WaypointSpec(heading=heading, bearing=bearing, distance=distance, position_offset=position_offset)


def held_out_eval_specs(
    n: int, *, seed: int, distance_range: tuple[float, float] = (8.0, 25.0),
    position_offset_radius_cells: float = 10.0,
) -> list[WaypointSpec]:
    """A FIXED list of specs, generated once from a seed disjoint from any
    training RNG range, reused across every checkpoint's evaluation for
    consistency -- the held-out combinations the user asked for, not fresh
    random draws each eval call."""
    rng = np.random.default_rng(seed)
    return [
        sample_generalized_spec(rng, distance_range=distance_range, position_offset_radius_cells=position_offset_radius_cells)
        for _ in range(n)
    ]

_CONDITION_ANGLE_RADIANS: dict[WaypointCondition, float] = {
    "left30": math.radians(30.0),
    "right30": -math.radians(30.0),
    "straight": 0.0,
}
FIXED_HEADING = 0.0
WAYPOINT_DISTANCE_CELLS = 15.0
SUCCESS_RADIUS_CELLS = 2.0
COLLISION_TERMINAL_REWARD = -500.0
SUCCESS_TERMINAL_REWARD = 20.0
EVENT_NONE = 0
MAP_HALF_SIZE_CELLS = 40


SYMMETRIC_MOVEMENT: tuple[MovementModel, ...] = (
    MovementModel(100, 1.0, 0.0, 0.0, 0.0),    # FORWARD (STRAIGHT)
    MovementModel(100, 1.0, 0.0, 0.25, 0.0),   # FORWARD_LEFT
    MovementModel(100, 1.0, 0.0, -0.25, 0.0),  # FORWARD_RIGHT -- exact mirror of LEFT
    MovementModel(0, 0.0, 0.0, 0.0, 0.0),      # CAST_EVA (unused, event forced NONE)
    MovementModel(10, 1.0, 0.0, 0.0, 0.0),     # JUMP (unused)
)


def symmetrize_movement(movement: tuple[MovementModel, ...]) -> tuple[MovementModel, ...]:
    """Matched-scale symmetrization of a real (possibly asymmetric)
    movement tuple: LEFT/RIGHT turn magnitude and std are both replaced by
    the AVERAGE of their absolute values (preserving overall turning
    'power'/noise scale), with opposite signs; FORWARD/EVA/JUMP entries
    are left untouched. Used for the recorded-vs-symmetric matched
    ablation -- everything about the movement model except the L/R
    asymmetry itself stays identical to the recorded condition."""
    left, right = movement[1], movement[2]
    mean_turn = (abs(left.turn_mean_radians) + abs(right.turn_mean_radians)) / 2.0
    mean_std = (left.turn_std_radians + right.turn_std_radians) / 2.0
    dist_mean = (left.distance_mean_cells + right.distance_mean_cells) / 2.0
    dist_std = (left.distance_std_cells + right.distance_std_cells) / 2.0
    symmetric_left = MovementModel(left.samples, dist_mean, dist_std, mean_turn, mean_std)
    symmetric_right = MovementModel(right.samples, dist_mean, dist_std, -mean_turn, mean_std)
    return (movement[0], symmetric_left, symmetric_right, movement[3], movement[4])


def build_open_world(
    *, population: int = 1, movement: tuple[MovementModel, ...] = SYMMETRIC_MOVEMENT,
) -> tuple[MapModel, RecordedWorldModel]:
    """Fully open (obstacle-free) map with a caller-supplied movement
    model -- defaults to SYMMETRIC_MOVEMENT (explicitly matched-magnitude,
    zero-noise LEFT/RIGHT turns), so any observed handedness in that
    default configuration must come from the network/training dynamics,
    not sampling variance or a movement-model asymmetry. Pass the real
    curriculum's movement tuple (or `symmetrize_movement` applied to it)
    for the recorded-vs-symmetric matched ablation."""
    size = MAP_HALF_SIZE_CELLS * 2 + 1
    map_model = MapModel.from_arrays(np.ones((size, size), dtype=bool))
    center = MAP_HALF_SIZE_CELLS
    positions = tuple(map_model.layout_to_native(center, center) for _ in range(4))
    sections = tuple(positions for _ in range(3))
    world = RecordedWorldModel(
        schema_version=5, source_recordings=("static_waypoint",), section_count=2, hub_section=2,
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


def make_static_waypoint_env(
    *, episode_steps: int, seed: int = 0, movement: tuple[MovementModel, ...] = SYMMETRIC_MOVEMENT,
) -> RecordedFarmingEnv:
    map_model, world = build_open_world(movement=movement)
    env = RecordedFarmingEnv(world, map_model=map_model, episode_steps=episode_steps)
    env.reset(seed=seed)
    # Only actor index 0 is ever used as the waypoint; the rest exist only
    # because RecordedFarmingEnv's population bookkeeping expects them --
    # keep them permanently dead so they can never appear in the
    # observation or be mistaken for a second waypoint.
    for actor in env.actors[1:]:
        actor.alive = False
    return env


class StaticWaypointWrapper(gym.Wrapper):
    """Places actor[0] as a fixed-for-the-episode waypoint. Two mutually
    exclusive placement modes:

    `condition` (original, 2026-08-10): one of the 3 fixed angles
    (left30/right30/straight) at WAYPOINT_DISTANCE_CELLS, heading forced
    to FIXED_HEADING=0.0, player position untouched (wherever the env
    naturally spawned it). Kept for the mirror-symmetry check and the
    original matched movement-dynamics ablation -- do not remove.

    `spec_source` (2026-08-11 generalization stage): a callable returning
    a WaypointSpec (randomized heading/bearing/distance/position) each
    reset -- REQUIRED for training/evaluating a policy that must
    generalize beyond the 3 original discrete angles and single fixed
    distance/heading/position.

    Reward = per-tick euclidean-distance reduction to the waypoint (open
    map -> geodesic == euclidean); terminates successfully
    (SUCCESS_TERMINAL_REWARD) within SUCCESS_RADIUS_CELLS, unsuccessfully
    (COLLISION_TERMINAL_REWARD) on physical contact. Steering-only: event
    always forced to NONE, matching PureNavigationWrapper's methodology.

    `condition=None, spec_source=None` (original random-condition mode)
    picks a fresh random condition every reset (via the supplied `rng`).
    """

    def __init__(
        self,
        env: Any,
        *,
        condition: WaypointCondition | None = None,
        rng: np.random.Generator | None = None,
        spec_source: Callable[[], WaypointSpec] | None = None,
    ):
        super().__init__(env)
        if condition is not None and spec_source is not None:
            raise ValueError("condition and spec_source are mutually exclusive")
        self._fixed_condition = condition
        self._spec_source = spec_source
        self._rng = rng or np.random.default_rng()
        self.condition: WaypointCondition = condition or "straight"
        self.waypoint_spec: WaypointSpec | None = None
        self._prev_distance = 0.0
        self._prev_contacts = 0
        # Set at reset from the ACTUAL initial distance (which varies
        # under spec_source, unlike the old fixed WAYPOINT_DISTANCE_CELLS)
        # -- required for correct, obstacle-stage-ready path-efficiency
        # accounting (2026-08-11 correction: the old metric divided by the
        # fixed 15-cell constant even though success triggers inside
        # SUCCESS_RADIUS_CELLS, so it could exceed 1).
        self.initial_distance_cells: float = 0.0

    def _place_waypoint(self) -> None:
        base_env = self.env.unwrapped
        if self._spec_source is not None:
            spec = self._spec_source()
            self.waypoint_spec = spec
            anchor_x, anchor_z = base_env.player_x, base_env.player_z
            cell_size = base_env.map.native_units_per_cell
            base_env.player_x = anchor_x + spec.position_offset[0] * cell_size
            base_env.player_z = anchor_z + spec.position_offset[1] * cell_size
            base_env.heading = spec.heading
            angle = spec.heading + spec.bearing
            native_distance = spec.distance * cell_size
        else:
            base_env.heading = FIXED_HEADING
            angle = FIXED_HEADING + _CONDITION_ANGLE_RADIANS[self.condition]
            native_distance = WAYPOINT_DISTANCE_CELLS * base_env.map.native_units_per_cell
        base_env.actors[0].x = base_env.player_x + math.cos(angle) * native_distance
        base_env.actors[0].z = base_env.player_z + math.sin(angle) * native_distance
        base_env.actors[0].alive = True
        for actor in base_env.actors[1:]:
            actor.alive = False

    def _distance_to_waypoint_cells(self) -> float:
        base_env = self.env.unwrapped
        actor = base_env.actors[0]
        dx = actor.x - base_env.player_x
        dz = actor.z - base_env.player_z
        return math.hypot(dx, dz) / base_env.map.native_units_per_cell

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self._spec_source is None:
            self.condition = self._fixed_condition if self._fixed_condition is not None else self._rng.choice(CONDITIONS)
        self._place_waypoint()
        base_env = self.env.unwrapped
        # Recompute the raw observation to reflect the just-placed waypoint
        # and heading (the reset() call above already returned a
        # pre-placement observation); preserve whatever sidecar tail an
        # outer wrapper (e.g. NavigationHistoryWrapper) already appended.
        raw = base_env._observation()
        if obs.shape[0] > raw.shape[0]:
            obs = np.concatenate([raw, obs[raw.shape[0]:]])
        else:
            obs = raw
        self._prev_distance = self._distance_to_waypoint_cells()
        self.initial_distance_cells = self._prev_distance
        self._prev_contacts = int(info.get("contacts", 0)) if isinstance(info, dict) else 0
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.int64).copy()
        action[1] = EVENT_NONE
        obs, _reward, terminated, truncated, info = self.env.step(action)
        contacts = int(info.get("contacts", 0))
        contact_this_tick = contacts > self._prev_contacts
        self._prev_contacts = contacts
        distance = self._distance_to_waypoint_cells()
        progress = self._prev_distance - distance
        self._prev_distance = distance

        if contact_this_tick:
            return obs, COLLISION_TERMINAL_REWARD, True, truncated, info
        if distance <= SUCCESS_RADIUS_CELLS:
            return obs, SUCCESS_TERMINAL_REWARD, True, truncated, info
        return obs, progress, terminated, truncated, info
