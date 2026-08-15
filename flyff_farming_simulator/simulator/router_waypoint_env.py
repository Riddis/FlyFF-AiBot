"""2026-08-14: Beginner Navigation Training Mix, Part 1 -- a training-time
wrapper that branches per episode between the frozen open-waypoint task
(no obstacles, exactly reproducing the qualified lineage's own training
conditions) and a router-driven obstacle task (single-wall / two-wall,
frozen `plan_route`/`select_persistent_waypoint`/`TargetPersistenceController`
select a MOVING waypoint each tick, exactly mirroring the eval-only
`run_episode_general_router` tick order). See the approved plan
("Beginner Navigation Training Mix") for the full design rationale,
including two structural decisions made only after they were verified
against the actual current code (not assumed):

  1. Open episodes use `static_waypoint_env.build_open_world()` (81x81);
     obstacle episodes use the injected `world_builder` (in practice
     `build_multi_wall_world`, 91x91). No coordinate-frame unification is
     needed -- `MapModel.grid_origin = size // 2` auto-centers every map
     at native (0,0) regardless of grid size, verified directly against
     `simulator/map_model.py`. Forcing everything onto one map size would
     only have introduced an unforced observation-distribution shift
     (`normalized_position()`/`context_crop()` DO depend on map
     dimensions, even though the origin doesn't).
  2. Within one `step()` call, the progress-reward `distance_before`/
     `distance_after` pair is always measured against `self._current_
     target`, the ONE target already fixed when that call started
     (selected by the prior `reset()` or the prior `step()`'s tail) --
     never re-selected mid-call. This structurally prevents a target
     switch from being folded into the reward as fake progress, rather
     than relying on a re-anchoring rule to avoid it.

Obstacle-generation failure handling: a requested obstacle episode MUST
remain that mode. On an invalid draw (`final_native` out of bounds/
untraversable, or `plan_route` finds no route), resample another spec of
the SAME mode from the injected `obstacle_spec_source`, up to
`max_route_retries` times; if exhausted, raise -- never silently convert
to an open episode (that would corrupt the declared episode-mode mixture
contract).

This module never imports root-level scratchpads (the samplers and
`build_multi_wall_world` live there) -- all obstacle-geometry sources are
injected callables, matching `StaticWaypointWrapper`'s existing
`spec_source` convention.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Literal

import gymnasium as gym
import numpy as np

from .kinodynamic_route_planner import TargetPersistenceController, plan_route, select_persistent_waypoint
from .map_model import MapModel
from .single_obstacle_env import MAP_HALF_SIZE_CELLS, ObstacleSpec
from .static_waypoint_env import (
    COLLISION_TERMINAL_REWARD, EVENT_NONE, FIXED_HEADING, SUCCESS_RADIUS_CELLS, SUCCESS_TERMINAL_REWARD,
    StaticWaypointWrapper, WaypointSpec, build_open_world,
)
from .world_model import RecordedWorldModel

EpisodeMode = Literal["open", "single_wall", "two_wall"]

# Cited, not imported (this module cannot import root-level scratchpads):
# scratchpad_generalized_waypoint_train_reward_ablation.py's LIVING_COST =
# 0.0441, the data-derived constant used by the "both"/combined reward mode
# that trained the frozen starting checkpoint. Applying the SAME constant
# here (true-terminal-on-timeout + this subtraction) keeps the reward scale
# this continuation trains under identical to what the checkpoint already
# expects, avoiding a value-function distribution shift from the reward
# formulation alone.
LIVING_COST = 0.0441


@dataclass(frozen=True)
class ObstacleEpisodeSpec:
    """Duck-typed union input for obstacle-mode episodes -- built by the
    caller (a root-level training script) from RandomizedWallSpec or
    TwoWallSpec, normalized here so this module never imports either."""
    wall_specs: list[ObstacleSpec]  # 1 entry (single_wall) or 2 (two_wall)
    approach_heading_offset_radians: float
    distance_cells: float  # straight-line distance to the TRUE final target, along FIXED_HEADING


def _final_native_valid(map_model: MapModel, final_native: tuple[float, float]) -> bool:
    cell = map_model.native_to_layout_cell(final_native[0], final_native[1])
    if cell is None:
        return False
    return bool(map_model.traversable[cell[1], cell[0]])


class RouterMixedWaypointWrapper(gym.Wrapper):
    """Per-episode mode-branching training wrapper. See module docstring."""

    def __init__(
        self,
        env: Any,
        *,
        mode_source: Callable[[], EpisodeMode],
        open_spec_source: Callable[[], WaypointSpec],
        obstacle_spec_source: Callable[[EpisodeMode], ObstacleEpisodeSpec],
        world_builder: Callable[[list[ObstacleSpec]], tuple[MapModel, RecordedWorldModel]],
        episode_steps_by_mode: dict[EpisodeMode, int],
        living_cost: float = LIVING_COST,
        max_route_retries: int = 20,
    ) -> None:
        super().__init__(env)
        self._mode_source = mode_source
        self._obstacle_spec_source = obstacle_spec_source
        self._world_builder = world_builder
        self._episode_steps_by_mode = dict(episode_steps_by_mode)
        self._living_cost = float(living_cost)
        self._max_route_retries = int(max_route_retries)
        # Constructed once, wraps the SAME underlying env -- StaticWaypointWrapper.reset()
        # never rebuilds/replaces base_env.map itself (verified directly), so this delegation
        # cannot silently override this wrapper's own map-per-mode assignment.
        self._static_helper = StaticWaypointWrapper(env, spec_source=open_spec_source)

        self._mode: EpisodeMode = "open"
        self._route: list[Any] | None = None
        self._controller: TargetPersistenceController | None = None
        self._final_native: tuple[float, float] | None = None
        self._current_target: tuple[float, float] | None = None
        self._prev_contacts: int = 0

    # -- reset ---------------------------------------------------------

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict]:
        self._mode = self._mode_source()
        base_env = self.env.unwrapped

        if self._mode == "open":
            map_model, _world = build_open_world()
            base_env.map = map_model
            base_env.episode_steps = self._episode_steps_by_mode["open"]
            obs, info = self._static_helper.reset(**kwargs)
            return obs, info

        map_model, final_native, initial_heading, route = self._sample_valid_obstacle_episode()
        base_env.map = map_model
        base_env.episode_steps = self._episode_steps_by_mode[self._mode]
        obs, info = self.env.reset(**kwargs)
        for actor in base_env.actors[1:]:
            actor.alive = False
        base_env.heading = initial_heading
        base_env.actors[0].x, base_env.actors[0].z = final_native
        base_env.actors[0].alive = True

        self._route = route
        self._controller = TargetPersistenceController(map_model, final_native[0], final_native[1])
        self._final_native = final_native
        self._prev_contacts = 0

        candidate = select_persistent_waypoint(
            map_model, route, player_x=base_env.player_x, player_z=base_env.player_z, heading=base_env.heading,
        )
        if candidate is None:
            candidate = final_native
        target = self._controller.update(candidate, player_x=base_env.player_x, player_z=base_env.player_z, route=route)
        base_env.actors[0].x, base_env.actors[0].z = target
        self._current_target = target

        obs = self.env._augment(base_env._observation(), base_env.previous_steering)
        return obs, info

    def _sample_valid_obstacle_episode(self):
        for _attempt in range(self._max_route_retries):
            spec = self._obstacle_spec_source(self._mode)
            map_model, _world = self._world_builder(spec.wall_specs)
            cell_size = map_model.native_units_per_cell
            center_native = map_model.layout_to_native(MAP_HALF_SIZE_CELLS, MAP_HALF_SIZE_CELLS)
            final_native = (
                center_native[0] + math.cos(FIXED_HEADING) * spec.distance_cells * cell_size,
                center_native[1] + math.sin(FIXED_HEADING) * spec.distance_cells * cell_size,
            )
            if not _final_native_valid(map_model, final_native):
                continue
            initial_heading = FIXED_HEADING + spec.approach_heading_offset_radians
            route = plan_route(
                map_model, start_x=center_native[0], start_z=center_native[1], start_heading=initial_heading,
                destination_x=final_native[0], destination_z=final_native[1],
            )
            if len(route) < 2:
                continue
            return map_model, final_native, initial_heading, route
        raise RuntimeError(
            f"RouterMixedWaypointWrapper: exhausted {self._max_route_retries} same-mode resamples "
            f"for mode={self._mode!r} without finding a valid (in-bounds, routable) episode -- this is "
            f"a real generator/geometry problem, not something to silently fall back past."
        )

    # -- step ------------------------------------------------------------

    def step(self, action):
        if self._mode == "open":
            obs, reward, terminated, truncated, info = self._static_helper.step(action)
        else:
            obs, reward, terminated, truncated, info = self._obstacle_step(action)
        reward, terminated, truncated = self._apply_combined_transform(reward, terminated, truncated)
        # Pure instrumentation (transparency only, never gates/adjusts anything):
        # lets a training-side callback measure REALIZED per-mode transition-tick
        # fractions, since the declared EPISODE_MODE_PROBS contract is episode-
        # weighted, not transition-weighted -- see the training script.
        info = dict(info)
        info["episode_mode"] = self._mode
        return obs, reward, terminated, truncated, info

    def _obstacle_step(self, action):
        executed_action = np.array(action, dtype=np.int64, copy=True)
        executed_action[1] = EVENT_NONE

        base_env = self.env.unwrapped
        cell_size = base_env.map.native_units_per_cell
        pre_x, pre_z = base_env.player_x, base_env.player_z
        distance_before = math.hypot(self._current_target[0] - pre_x, self._current_target[1] - pre_z) / cell_size

        obs, _reward, terminated, truncated, info = self.env.step(executed_action)  # base env reward discarded

        contacts = int(info.get("contacts", 0))
        contact_this_tick = contacts > self._prev_contacts
        self._prev_contacts = contacts

        post_x, post_z = base_env.player_x, base_env.player_z
        final_distance = math.hypot(self._final_native[0] - post_x, self._final_native[1] - post_z) / cell_size

        if contact_this_tick:
            reward, terminated = COLLISION_TERMINAL_REWARD, True
        elif final_distance <= SUCCESS_RADIUS_CELLS:
            reward, terminated = SUCCESS_TERMINAL_REWARD, True
        else:
            distance_after = math.hypot(self._current_target[0] - post_x, self._current_target[1] - post_z) / cell_size
            reward = distance_before - distance_after
            if not truncated:
                candidate = select_persistent_waypoint(
                    base_env.map, self._route, player_x=post_x, player_z=post_z, heading=base_env.heading,
                )
                if candidate is None:
                    candidate = self._final_native
                target = self._controller.update(candidate, player_x=post_x, player_z=post_z, route=self._route)
                base_env.actors[0].x, base_env.actors[0].z = target
                self._current_target = target
                obs = self.env._augment(base_env._observation(), base_env.previous_steering)

        return obs, reward, terminated, truncated, info

    def _apply_combined_transform(self, reward: float, terminated: bool, truncated: bool) -> tuple[float, bool, bool]:
        if truncated and not terminated:
            terminated, truncated = True, False
        reward = reward - self._living_cost
        return reward, terminated, truncated
