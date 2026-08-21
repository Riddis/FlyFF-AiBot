"""2026-08-15: one causal diagnostic, per explicit user instruction.
NOT a proposed production fix -- this evaluation-only controller variant
suppresses BOTH of TargetPersistenceController.update()'s FINAL_TARGET_LOCK
branches (the initial lock-in check AND the "already locked" short-circuit
on subsequent ticks) while leaving every other line -- the patched
selector, the INITIAL/KEEP_CURRENT/CURRENT_REACHED_OR_PASSED/
CURRENT_UNSAFE/BETTER_FORWARD_TARGET hysteresis logic -- byte-identical to
production. Runs ONLY on single_wall_right[1] (the one 830M episode B
regressed).

Question: if the exact same patched trajectory is prevented from ever
entering final-lock mode, does the collision disappear?
  - If yes: the downstream lock transition is causally necessary for this
    regression (the selector's low-margin pick alone isn't sufficient to
    explain the collision).
  - If it still collides: the earlier trajectory divergence (the selector
    itself) is the real issue, and final-lock is merely where the
    symptom becomes visible.

No controller/selector production code is modified.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from scratchpad_beginner_navigation_mix_pools import _reconstruct_single_wall_world, load_manifest
from scratchpad_diagnose_single_wall_right_1_regression import DEV_POOL_EPISODE_SEED_BASE, INDEX, STRATUM, _seed_counter_for
from scratchpad_router_patch_qualification_pool import QUALIFICATION_SPEC_SEED
from simulator.environment import RecordedFarmingEnv
from simulator.kinodynamic_route_planner import (
    DESIRED_CLEARANCE_CELLS, TargetPersistenceController, TargetSwitchReason, _direct_hop_min_clearance,
    _nearest_route_index, plan_route, select_persistent_waypoint_experimental_collision_free_fallback,
)
from simulator.navigation_history import NavigationHistoryWrapper
from simulator.static_waypoint_env import SUCCESS_RADIUS_CELLS

ROOT = Path(__file__).resolve().parents[1]
BASELINE_CHECKPOINT = ROOT / "models" / "generalized_waypoint_both_seed2_0051200.zip"
EPISODE_STEPS = 200
STEERING_NAMES = {0: "STRAIGHT", 1: "LEFT", 2: "RIGHT"}


class NoFinalLockController(TargetPersistenceController):
    """EVAL-ONLY diagnostic variant -- see module docstring. Identical to
    TargetPersistenceController.update() except both FINAL_TARGET_LOCK
    branches are removed; falls straight through to the normal
    INITIAL/CURRENT_REACHED_OR_PASSED/CURRENT_UNSAFE/BETTER_FORWARD_TARGET/
    KEEP_CURRENT hysteresis using `candidate` (the patched selector's own
    output) exactly as production does for every OTHER decision. Never
    imported by any production code path."""

    def update(self, candidate: tuple[float, float], *, player_x: float, player_z: float, route) -> tuple[float, float]:
        cell_size = self.map_model.native_units_per_cell

        if self.previous_target is None:
            self.previous_target = candidate
            self.target_switches += 1
            self._set_reason(TargetSwitchReason.INITIAL)
            return candidate

        player_idx = _nearest_route_index(route, player_x, player_z)
        prev_idx = _nearest_route_index(route, self.previous_target[0], self.previous_target[1])
        dist_to_prev = math.hypot(self.previous_target[0] - player_x, self.previous_target[1] - player_z) / cell_size
        prev_reached_or_passed = dist_to_prev <= self.REACH_RADIUS_CELLS or player_idx >= prev_idx

        if prev_reached_or_passed:
            self.previous_target = candidate
            self.target_switches += 1
            self._set_reason(TargetSwitchReason.CURRENT_REACHED_OR_PASSED)
            return candidate

        direct_clearance_to_prev = _direct_hop_min_clearance(
            self.map_model, player_x, player_z, self.previous_target[0], self.previous_target[1],
        )
        prev_still_safe = direct_clearance_to_prev >= DESIRED_CLEARANCE_CELLS
        if not prev_still_safe:
            self.previous_target = candidate
            self.target_switches += 1
            self._set_reason(TargetSwitchReason.CURRENT_UNSAFE)
            return candidate

        progress_prev = math.hypot(self.previous_target[0] - self.destination[0], self.previous_target[1] - self.destination[1]) / cell_size
        progress_candidate = math.hypot(candidate[0] - self.destination[0], candidate[1] - self.destination[1]) / cell_size
        if progress_prev - progress_candidate >= self.PROGRESS_IMPROVEMENT_MARGIN_CELLS:
            self.previous_target = candidate
            self.target_switches += 1
            self._set_reason(TargetSwitchReason.BETTER_FORWARD_TARGET)
            return candidate

        self._set_reason(TargetSwitchReason.KEEP_CURRENT)
        return self.previous_target


def run_traced(model, map_model, world, *, initial_heading, final_native, episode_seed, controller_cls, label: str):
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
    assert len(route) >= 2, f"{label}: no route found"
    controller = controller_cls(map_model, final_native[0], final_native[1])

    ticks = []
    prev_contacts = 0
    outcome = "timeout"

    for tick in range(EPISODE_STEPS):
        pre_x, pre_z, pre_heading = base_env.player_x, base_env.player_z, base_env.heading
        prev_steering = base_env.previous_steering

        candidate = select_persistent_waypoint_experimental_collision_free_fallback(
            map_model, route, player_x=pre_x, player_z=pre_z, heading=pre_heading,
        )
        if candidate is None:
            candidate = final_native
        target = controller.update(candidate, player_x=pre_x, player_z=pre_z, route=route)
        reason = controller.last_switch_reason.value if controller.last_switch_reason else None

        base_env.actors[0].x, base_env.actors[0].z = target
        obs = env._augment(base_env._observation(), prev_steering)
        action, _state = model.predict(obs, deterministic=True)
        a = int(action[0])
        action_arr = np.asarray(action, dtype=np.int64).copy()
        action_arr[1] = 0
        obs, _reward, term, trunc, info = env.step(action_arr)

        post_x, post_z = base_env.player_x, base_env.player_z
        contacts = int(info.get("contacts", 0))
        contact_this_tick = contacts > prev_contacts
        prev_contacts = contacts
        final_distance = math.hypot(final_native[0] - post_x, final_native[1] - post_z) / cell_size

        ticks.append({
            "tick": tick, "pre_pose": (round(pre_x, 3), round(pre_z, 3), round(math.degrees(pre_heading), 2)),
            "controller_target": (round(target[0], 3), round(target[1], 3)), "controller_reason": reason,
            "steering": STEERING_NAMES[a], "final_distance_cells": round(final_distance, 3), "contact": contact_this_tick,
        })

        if contact_this_tick:
            outcome = "collision"
            break
        if final_distance <= SUCCESS_RADIUS_CELLS:
            outcome = "success"
            break

    env.close()
    return outcome, ticks


def main() -> None:
    baseline_model = PPO.load(str(BASELINE_CHECKPOINT), device="cpu")
    manifest = load_manifest(ROOT / "evaluations" / f"router_mix_qualification_pool_{QUALIFICATION_SPEC_SEED}_manifest.json")
    counter = _seed_counter_for(manifest, STRATUM, INDEX)
    episode_seed = DEV_POOL_EPISODE_SEED_BASE + counter
    episode = manifest["strata"][STRATUM]["accepted"][INDEX]
    map_model, world, final_native, initial_heading = _reconstruct_single_wall_world(episode)

    results = {}
    for controller_cls, label in [(TargetPersistenceController, "B_with_final_lock"), (NoFinalLockController, "B_final_lock_SUPPRESSED")]:
        outcome, ticks = run_traced(
            baseline_model, map_model, world, initial_heading=initial_heading, final_native=final_native,
            episode_seed=episode_seed, controller_cls=controller_cls, label=label,
        )
        results[label] = {"outcome": outcome, "ticks": ticks}
        print(f"\n{'=' * 100}\n{label}: outcome={outcome} ticks_run={len(ticks)}\n{'=' * 100}")
        for row in ticks:
            print(f"  tick={row['tick']:3d} steer={row['steering']:8s} "
                  f"controller_target={row['controller_target']} reason={row['controller_reason']:24s} "
                  f"dist={row['final_distance_cells']:.2f} contact={row['contact']}")

    print(f"\n{'=' * 100}\nCAUSAL VERDICT\n{'=' * 100}")
    with_lock = results["B_with_final_lock"]["outcome"]
    suppressed = results["B_final_lock_SUPPRESSED"]["outcome"]
    print(f"With FINAL_TARGET_LOCK (production behavior): {with_lock}")
    print(f"With FINAL_TARGET_LOCK suppressed (eval-only): {suppressed}")
    if with_lock == "collision" and suppressed != "collision":
        print("Collision DISAPPEARS when final-lock is suppressed -- the lock transition is causally necessary for this regression.")
    elif with_lock == "collision" and suppressed == "collision":
        print("Collision PERSISTS even with final-lock suppressed -- the earlier trajectory divergence (selector pick) is the real issue; final-lock is not the cause.")
    else:
        print("Unexpected: production run did not collide in this rerun -- check for nondeterminism before drawing a conclusion.")

    out_path = ROOT / "evaluations" / "diagnose_swr1_final_lock_suppressed.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
