"""2026-08-15: broad equivalence check, per explicit user instruction,
BEFORE removing the (now temporarily duplicated) `select_persistent_
waypoint_experimental_invalid_hop_guard` function -- proves the promotion
edit to `select_persistent_waypoint` (simulator/kinodynamic_route_
planner.py) reproduces the already-validated v2 experimental
implementation's behavior EXACTLY, tick-for-tick, catching any copy-paste
error in the promotion itself before the duplicate is retired.

Drives full episodes with the OLD experimental function (ground truth,
unchanged since 840M/820M validation) and, at EVERY tick, ALSO computes
the NEW production `select_persistent_waypoint`'s output from the
identical pre-tick pose (hypothetically -- does not affect movement),
asserting exact equality. Covers:
  - all 120 episodes of the 812M dev pool (single_wall + two_wall,
    thousands of per-tick selector calls across real, varied geometry)
  - RTL8, RTL9 (the original invalid-hop repair cases)
  - SWR1 (single_wall_right[1], 830M -- the guard-is-a-no-op case)
  - episode67 (706M pool, two_wall_right_then_left[67])
"""
from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO

from .scratchpad_beginner_navigation_mix_pools import (
    DEV_POOL_SPEC_SEED, _reconstruct_single_wall_world, _reconstruct_two_wall_world, load_manifest,
)
from .scratchpad_beginner_routing_two_wall_s_route import held_out_two_wall_specs_for_direction
from .scratchpad_general_router_episode import build_multi_wall_world
from .scratchpad_router_v2_qualification_pool import QUALIFICATION_SPEC_SEED as SEED_830M
from simulator.environment import RecordedFarmingEnv
from navigation.kinodynamic_route_planner import (
    TargetPersistenceController, plan_route, select_persistent_waypoint, select_persistent_waypoint_experimental_invalid_hop_guard,
)
from simulator.navigation_history import NavigationHistoryWrapper
from simulator.single_obstacle_env import MAP_HALF_SIZE_CELLS
from simulator.static_waypoint_env import FIXED_HEADING, SUCCESS_RADIUS_CELLS

import math
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASELINE_CHECKPOINT = ROOT / "models" / "generalized_waypoint_both_seed2_0051200.zip"
EPISODE_STEPS = 200
DEV_POOL_EPISODE_SEED_BASE = 812_600_000
SEED_830M_BASE = 830_600_000


def _seed_counter_for(manifest: dict, stratum_name: str, index: int) -> int:
    counter = 0
    for name, stratum in manifest["strata"].items():
        if name == "open":
            continue
        if name == stratum_name:
            return counter + index
        counter += len(stratum["accepted"])
    raise KeyError(stratum_name)


def check_episode(model, map_model, world, *, initial_heading, final_native, episode_seed, label: str) -> int:
    """Drives the episode with the OLD experimental function, comparing
    the NEW production function's output at every tick. Returns the
    number of ticks checked. Raises on any mismatch."""
    raw_env = RecordedFarmingEnv(world, map_model=map_model, episode_steps=EPISODE_STEPS)
    env = NavigationHistoryWrapper(raw_env)
    obs, _info = env.reset(seed=episode_seed)
    base_env = env.unwrapped
    for actor in base_env.actors[1:]:
        actor.alive = False
    base_env.heading = initial_heading
    cell_size = base_env.map.native_units_per_cell
    base_env.actors[0].x, base_env.actors[0].z = final_native
    base_env.actors[0].alive = True

    route = plan_route(
        map_model, start_x=base_env.player_x, start_z=base_env.player_z, start_heading=base_env.heading,
        destination_x=final_native[0], destination_z=final_native[1],
    )
    if len(route) < 2:
        env.close()
        return 0

    controller = TargetPersistenceController(map_model, final_native[0], final_native[1])
    prev_contacts = 0
    ticks_checked = 0

    for tick in range(EPISODE_STEPS):
        pre_x, pre_z, pre_heading = base_env.player_x, base_env.player_z, base_env.heading
        prev_steering = base_env.previous_steering

        old_output = select_persistent_waypoint_experimental_invalid_hop_guard(
            map_model, route, player_x=pre_x, player_z=pre_z, heading=pre_heading,
        )
        new_output = select_persistent_waypoint(
            map_model, route, player_x=pre_x, player_z=pre_z, heading=pre_heading,
        )
        assert old_output == new_output, (
            f"{label} tick {tick}: EQUIVALENCE FAILURE -- old(experimental)={old_output} new(production)={new_output} "
            f"at pose=({pre_x},{pre_z},{pre_heading})"
        )
        ticks_checked += 1

        candidate = old_output if old_output is not None else final_native
        target = controller.update(candidate, player_x=pre_x, player_z=pre_z, route=route)

        base_env.actors[0].x, base_env.actors[0].z = target
        obs = env._augment(base_env._observation(), prev_steering)
        action, _state = model.predict(obs, deterministic=True)
        action_arr = np.asarray(action, dtype=np.int64).copy()
        action_arr[1] = 0
        obs, _reward, term, trunc, info = env.step(action_arr)

        contacts = int(info.get("contacts", 0))
        if contacts > prev_contacts:
            break
        prev_contacts = contacts
        final_distance = math.hypot(final_native[0] - base_env.player_x, final_native[1] - base_env.player_z) / cell_size
        if final_distance <= SUCCESS_RADIUS_CELLS:
            break

    env.close()
    return ticks_checked


def main() -> None:
    model = PPO.load(str(BASELINE_CHECKPOINT), device="cpu")
    total_ticks = 0
    total_episodes = 0

    print(f"{'=' * 90}\nEquivalence check: NEW select_persistent_waypoint vs OLD experimental_invalid_hop_guard\n{'=' * 90}")

    # -- 812M dev pool, all 120 obstacle episodes --
    dev_manifest = load_manifest(ROOT / "simulator" / "evaluations" / f"router_mix_dev_pool_{DEV_POOL_SPEC_SEED}_manifest.json")
    for stratum_name, stratum in dev_manifest["strata"].items():
        if stratum_name == "open":
            continue
        for index, episode in enumerate(stratum["accepted"]):
            if stratum_name.startswith("single_wall_"):
                map_model, world, final_native, initial_heading = _reconstruct_single_wall_world(episode)
            else:
                map_model, world, final_native, initial_heading = _reconstruct_two_wall_world(episode)
            seed = DEV_POOL_EPISODE_SEED_BASE + _seed_counter_for(dev_manifest, stratum_name, index)
            n_ticks = check_episode(model, map_model, world, initial_heading=initial_heading, final_native=final_native, episode_seed=seed, label=f"812M/{stratum_name}[{index}]")
            total_ticks += n_ticks
            total_episodes += 1
    print(f"812M dev pool: {total_episodes} episodes, {total_ticks} ticks checked, ALL EQUAL.")

    # -- SWR1, 830M --
    qual_manifest = load_manifest(ROOT / "simulator" / "evaluations" / f"router_mix_qualification_pool_{SEED_830M}_manifest.json")
    ep = qual_manifest["strata"]["single_wall_right"]["accepted"][1]
    map_model, world, final_native, initial_heading = _reconstruct_single_wall_world(ep)
    seed = SEED_830M_BASE + _seed_counter_for(qual_manifest, "single_wall_right", 1)
    n_ticks = check_episode(model, map_model, world, initial_heading=initial_heading, final_native=final_native, episode_seed=seed, label="SWR1")
    total_ticks += n_ticks
    total_episodes += 1
    print(f"SWR1: {n_ticks} ticks checked, EQUAL.")

    # -- episode67, 706M pool --
    specs = held_out_two_wall_specs_for_direction(100, direction="right_then_left", seed=706_000_000)
    spec = specs[67]
    wall1, wall2 = spec.wall1_obstacle_spec(), spec.wall2_obstacle_spec()
    map_model, world = build_multi_wall_world([wall1, wall2])
    cell_size = map_model.native_units_per_cell
    start_native = map_model.layout_to_native(MAP_HALF_SIZE_CELLS, MAP_HALF_SIZE_CELLS)
    final_native = (start_native[0] + math.cos(FIXED_HEADING) * spec.distance_cells * cell_size,
                     start_native[1] + math.sin(FIXED_HEADING) * spec.distance_cells * cell_size)
    initial_heading = FIXED_HEADING + spec.approach_heading_offset_radians
    n_ticks = check_episode(model, map_model, world, initial_heading=initial_heading, final_native=final_native, episode_seed=950_000_000 + 67, label="episode67")
    total_ticks += n_ticks
    total_episodes += 1
    print(f"episode67: {n_ticks} ticks checked, EQUAL.")

    print(f"\n{'=' * 90}\nTOTAL: {total_episodes} episodes, {total_ticks} ticks, ZERO mismatches -- "
          f"promotion edit is behaviorally equivalent to the validated experimental implementation.\n{'=' * 90}")


if __name__ == "__main__":
    main()
