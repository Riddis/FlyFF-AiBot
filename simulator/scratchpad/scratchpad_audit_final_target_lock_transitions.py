"""2026-08-15: audit the FIRST FINAL_TARGET_LOCK transition in every
patched-baseline 830M obstacle episode, per explicit user instruction.
Investigation has moved downstream from the selector's own candidate
ranking (proven insufficient on its own -- single_wall_left[21] and
single_wall_right[1] are structural mirror images with opposite correct
answers, see evaluations/diagnose_fallback_ranking_candidates.json) to
whether TargetPersistenceController.FINAL_TARGET_LOCK itself is the
mechanism that turns a safe-but-lower-margin approach into a collision.

AUDIT ONLY -- no controller/selector code is touched. Runs the frozen
baseline checkpoint + the EXPERIMENTAL collision-free-fallback selector
(condition B exactly as it ran in the 830M A/B/C comparison) + the
UNMODIFIED TargetPersistenceController against all 240 obstacle episodes
in the frozen 830M manifest. For every episode that ever reaches
FINAL_TARGET_LOCK, records the state at the FIRST tick that transition
fires:

  - final_distance_cells: distance from player to the true final target
  - heading_error_deg: bearing-to-final minus player heading, normalized
    to [-180, 180] -- same convention as RecordedFarmingEnv._bearing_to
    (simulator/environment.py:656-657): atan2(dz, dx) relative to heading
  - previous_steering: base_env.previous_steering BEFORE this tick's step
    (what fed this tick's observation)
  - current_steering: the deterministic PPO action taken THIS tick
  - held_target_bearing_deg: bearing to the newly-held target (== final
    target bearing, since locking sets target=destination exactly, but
    computed independently for completeness/verification)
  - direct_segment_clearance_to_final: _direct_hop_min_clearance from the
    player's exact pose to the final target (the same value the
    controller itself just used to decide to lock)
  - player_clearance_cells: player's own local clearance
    (_clearance_cells_native at the player's position, not directional)
  - previous_controller_reason: the reason recorded on the PRECEDING tick
    (what state led into this lock)
  - ticks_since_last_target_switch: current tick minus the tick index of
    the previous target switch (any reason, not just lock)
  - outcome: the episode's final recorded outcome (success/collision/
    timeout) from the SAME trajectory this lock transition belongs to

No variable "speed" state exists in this movement model to record --
movement_kernel.PATH_LENGTH_CELLS_PER_TICK is a fixed constant per tick
(confirmed by direct read), not a per-tick variable quantity.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from .scratchpad_beginner_navigation_mix_pools import (
    _reconstruct_single_wall_world, _reconstruct_two_wall_world, load_manifest,
)
from .scratchpad_router_patch_qualification_pool import QUALIFICATION_SPEC_SEED
from simulator.environment import RecordedFarmingEnv
from navigation.kinodynamic_route_planner import (
    TargetPersistenceController, TargetSwitchReason, _clearance_cells_native, _direct_hop_min_clearance,
    _normalize_angle, plan_route, select_persistent_waypoint_experimental_collision_free_fallback,
)
from simulator.navigation_history import NavigationHistoryWrapper
from simulator.static_waypoint_env import SUCCESS_RADIUS_CELLS

ROOT = Path(__file__).resolve().parents[2]
BASELINE_CHECKPOINT = ROOT / "models" / "generalized_waypoint_both_seed2_0051200.zip"
EPISODE_STEPS = 200
DEV_POOL_EPISODE_SEED_BASE = 830_600_000
STEERING_NAMES = {0: "STRAIGHT", 1: "LEFT", 2: "RIGHT"}


def _bearing_to(x: float, z: float, player_x: float, player_z: float, heading: float) -> float:
    """Matches RecordedFarmingEnv's own bearing convention exactly
    (simulator/environment.py:656-657): atan2(dz, dx), normalized relative
    to the player's current heading."""
    target = math.atan2(z - player_z, x - player_x)
    return _normalize_angle(target - heading)


def run_episode_with_lock_audit(model, map_model, world, *, initial_heading, final_native, episode_seed):
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
        return "planner_failure_no_route_found", None

    controller = TargetPersistenceController(map_model, final_native[0], final_native[1])

    prev_contacts = 0
    outcome = "timeout"
    lock_record = None
    prev_reason = None
    last_switch_tick = 0

    for tick in range(EPISODE_STEPS):
        pre_x, pre_z, pre_heading = base_env.player_x, base_env.player_z, base_env.heading
        prev_steering = base_env.previous_steering

        candidate = select_persistent_waypoint_experimental_collision_free_fallback(
            map_model, route, player_x=pre_x, player_z=pre_z, heading=pre_heading,
        )
        if candidate is None:
            candidate = final_native
        switches_before = controller.target_switches
        target = controller.update(candidate, player_x=pre_x, player_z=pre_z, route=route)
        reason = controller.last_switch_reason
        switched_this_tick = controller.target_switches != switches_before

        entered_lock_this_tick = (reason == TargetSwitchReason.FINAL_TARGET_LOCK and lock_record is None
                                   and (prev_reason != TargetSwitchReason.FINAL_TARGET_LOCK))

        base_env.actors[0].x, base_env.actors[0].z = target
        obs = env._augment(base_env._observation(), prev_steering)
        action, _state = model.predict(obs, deterministic=True)
        a = int(action[0])
        action_arr = np.asarray(action, dtype=np.int64).copy()
        action_arr[1] = 0

        if entered_lock_this_tick:
            direct_clearance_to_final = _direct_hop_min_clearance(map_model, pre_x, pre_z, final_native[0], final_native[1])
            lock_record = {
                "tick": tick,
                "final_distance_cells": round(math.hypot(final_native[0] - pre_x, final_native[1] - pre_z) / cell_size, 3),
                "heading_error_deg": round(math.degrees(_bearing_to(final_native[0], final_native[1], pre_x, pre_z, pre_heading)), 2),
                "previous_steering": STEERING_NAMES[int(prev_steering)],
                "current_steering": STEERING_NAMES[a],
                "held_target_bearing_deg": round(math.degrees(_bearing_to(target[0], target[1], pre_x, pre_z, pre_heading)), 2),
                "direct_segment_clearance_to_final": round(direct_clearance_to_final, 3),
                "player_clearance_cells": round(_clearance_cells_native(map_model, pre_x, pre_z), 3),
                "previous_controller_reason": prev_reason.value if prev_reason else None,
                "ticks_since_last_target_switch": tick - last_switch_tick,
            }

        if switched_this_tick:
            last_switch_tick = tick
        prev_reason = reason

        obs, _reward, term, trunc, info = env.step(action_arr)
        contacts = int(info.get("contacts", 0))
        if contacts > prev_contacts:
            outcome = "collision"
            break
        prev_contacts = contacts
        final_distance = math.hypot(final_native[0] - base_env.player_x, final_native[1] - base_env.player_z) / cell_size
        if final_distance <= SUCCESS_RADIUS_CELLS:
            outcome = "success"
            break

    env.close()
    if lock_record is not None:
        lock_record["outcome"] = outcome
    return outcome, lock_record


def main() -> None:
    baseline_model = PPO.load(str(BASELINE_CHECKPOINT), device="cpu")
    manifest = load_manifest(ROOT / "simulator" / "evaluations" / f"router_mix_qualification_pool_{QUALIFICATION_SPEC_SEED}_manifest.json")

    seed_counter = 0
    results = []
    n_episodes = 0
    n_reached_lock = 0
    for stratum_name, stratum in manifest["strata"].items():
        if stratum_name == "open":
            continue
        for index, episode in enumerate(stratum["accepted"]):
            n_episodes += 1
            episode_seed = DEV_POOL_EPISODE_SEED_BASE + seed_counter
            seed_counter += 1
            if stratum_name.startswith("single_wall_"):
                map_model, world, final_native, initial_heading = _reconstruct_single_wall_world(episode)
            else:
                map_model, world, final_native, initial_heading = _reconstruct_two_wall_world(episode)
            outcome, lock_record = run_episode_with_lock_audit(
                baseline_model, map_model, world, initial_heading=initial_heading, final_native=final_native,
                episode_seed=episode_seed,
            )
            if lock_record is not None:
                n_reached_lock += 1
                lock_record["stratum"] = stratum_name
                lock_record["index"] = index
                lock_record["label"] = f"{stratum_name}[{index}]"
                results.append(lock_record)
            if (n_episodes % 40) == 0:
                print(f"  ...{n_episodes}/240 episodes processed, {n_reached_lock} reached FINAL_TARGET_LOCK so far")

    print(f"\n{'=' * 90}\n{n_reached_lock}/{n_episodes} obstacle episodes reached FINAL_TARGET_LOCK at least once\n{'=' * 90}")

    by_outcome: dict[str, int] = {}
    for r in results:
        by_outcome[r["outcome"]] = by_outcome.get(r["outcome"], 0) + 1
    print(f"Outcome breakdown among lock-reaching episodes: {by_outcome}")

    swr1 = next((r for r in results if r["label"] == "single_wall_right[1]"), None)
    print(f"\nSWR1 lock record: {json.dumps(swr1, indent=2) if swr1 else 'DID NOT REACH LOCK (unexpected)'}")

    successes = [r for r in results if r["outcome"] == "success"]
    for field in ["final_distance_cells", "heading_error_deg", "direct_segment_clearance_to_final",
                  "player_clearance_cells", "ticks_since_last_target_switch"]:
        vals = [r[field] for r in successes]
        if vals:
            print(f"\n{field} among {len(vals)} SUCCESSFUL lock transitions: "
                  f"min={min(vals):.2f} max={max(vals):.2f} mean={sum(vals)/len(vals):.2f}"
                  + (f"  |  SWR1={swr1[field]:.2f}" if swr1 else ""))

    steering_counts_success = {}
    for r in successes:
        key = (r["previous_steering"], r["current_steering"])
        steering_counts_success[key] = steering_counts_success.get(key, 0) + 1
    print(f"\n(previous_steering, current_steering) among successful lock transitions: {steering_counts_success}")
    if swr1:
        print(f"SWR1's (previous_steering, current_steering): ({swr1['previous_steering']}, {swr1['current_steering']})")

    out_path = ROOT / "simulator" / "evaluations" / "audit_final_target_lock_transitions.json"
    out_path.write_text(json.dumps({"n_obstacle_episodes": n_episodes, "n_reached_lock": n_reached_lock, "results": results}, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
