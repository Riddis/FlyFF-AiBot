"""Corrected proof-of-mechanism experiment (v2), fixing two bugs found in
the v1 script (see run_logs/OVERNIGHT_20260809_PIPELINE.md):
1. Target position must come from a FRESH replay that stops exactly at the
   tick being examined -- never from mutating position on an env that has
   already been replayed past that point (stale _nearest_reachable_actor_id/
   _best_group_actor_id).
2. Compare what the BEAM WOULD ACTUALLY CHOOSE under each target angle
   (target_angle only affects scoring tie-break among safety-gated
   candidates), not a crude raw-angle classification.

Also reports the coarse path's clearance profile for the next ~40 cells, so
a "same immediate direction, but hugs more open space" case can be told
apart from "no coarse-routing information at all here".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from farming.actions import SteeringAction
from simulator.local_clearance import sample_heading_relative_clearance
from simulator.steering_oracle import (
    _beam_search_first_action, _oracle_steering_decision_v3, DEFAULT_ROBUST_SIGMA,
    DEFAULT_BEAM_DEPTH, DEFAULT_BEAM_WIDTH, DEFAULT_CONTINUATION_DEPTH,
)
from simulator.synthetic import iter_variant_environments
from simulator.curriculum_manifests import load_heldout_manifest
from simulator.scripted_policies import _obstacle_aware_target_angle, _event_for
from scratchpad_coarse_route_proof_of_mechanism import (
    compute_clearance_field, coarse_route, waypoint_direction, _target_position,
)

SIGMA = DEFAULT_ROBUST_SIGMA
EPISODE_SECONDS = 150.0
MAX_ACTIONS = 1000

# (layout, seed) -> list of check_ticks (tick right before each fallback streak begins)
CHECKPOINTS_BY_EPISODE = {
    ("03_early_irregular_plain_typical_fast", 0): [162],
    ("03_early_irregular_plain_typical_fast", 1): [52, 518],
}


def main() -> None:
    manifest = load_heldout_manifest("evaluations/manifests/oracle_fresh_confirmation.json")
    results = []

    for (layout_name, seed), check_ticks in CHECKPOINTS_BY_EPISODE.items():
        check_ticks_sorted = sorted(check_ticks)
        entry, env = next(iter(iter_variant_environments(
            manifest.curriculum_path, stage=manifest.stage, seed=seed, episode_steps=MAX_ACTIONS,
            episode_seconds=EPISODE_SECONDS, variant_name=layout_name,
        )))
        obs, _ = env.reset(seed=seed)
        prev_action = None
        next_checkpoint_idx = 0
        tick = 0
        while next_checkpoint_idx < len(check_ticks_sorted):
            target_tick = check_ticks_sorted[next_checkpoint_idx]
            while tick < target_tick:
                action, _ = _oracle_steering_decision_v3(
                    env, sigma=SIGMA, beam_depth=DEFAULT_BEAM_DEPTH, beam_width=DEFAULT_BEAM_WIDTH,
                    previous_action=prev_action, stage=manifest.stage, continuation_depth=DEFAULT_CONTINUATION_DEPTH,
                )
                obs, r, term, trunc, info = env.step(np.asarray([int(action), int(_event_for(env))], dtype=np.int64))
                prev_action = action
                tick += 1

            # Now at target_tick's decision point, via genuine fresh replay --
            # env._nearest_reachable_actor_id etc. are causally correct here.
            obstacle_aware_angle = _obstacle_aware_target_angle(env)
            target = _target_position(env)
            map_model = env.map
            coarse_angle = None
            path_clearances = None
            if target is not None:
                coarse_angle = waypoint_direction(map_model, env.player_x, env.player_z, env.heading, target[0], target[1])
                clearance_field = compute_clearance_field(map_model.traversable)
                start_cell = map_model.native_to_layout_cell(env.player_x, env.player_z)
                target_cell = map_model.native_to_layout_cell(target[0], target[1])
                if start_cell and target_cell:
                    path = coarse_route(map_model.traversable, clearance_field, start_cell, target_cell)
                    if path:
                        path_clearances = [int(clearance_field[y, x]) for x, y in path[:40]]

            def beam_choice(target_angle):
                clearance_raw = sample_heading_relative_clearance(env.map, env.player_x, env.player_z, env.heading)
                clearance_by_action = {
                    SteeringAction.STRAIGHT: clearance_raw["forward"], SteeringAction.LEFT: clearance_raw["left"],
                    SteeringAction.RIGHT: clearance_raw["right"],
                }
                chosen = _beam_search_first_action(
                    env.map, env.model.movement, env.player_x, env.player_z, env.heading,
                    sigma=SIGMA, depth=DEFAULT_BEAM_DEPTH, beam_width=DEFAULT_BEAM_WIDTH, previous_action=prev_action,
                    target_angle=target_angle, clearance=clearance_by_action, continuation_depth=DEFAULT_CONTINUATION_DEPTH,
                )
                return SteeringAction(chosen).name if chosen is not None else None

            obstacle_aware_choice = beam_choice(obstacle_aware_angle)
            coarse_choice = beam_choice(coarse_angle) if coarse_angle is not None else None
            agrees = obstacle_aware_choice == coarse_choice

            print(f"{layout_name}/seed{seed} tick{target_tick}: "
                  f"obstacle_aware_angle={obstacle_aware_angle} -> beam={obstacle_aware_choice} | "
                  f"coarse_angle={coarse_angle} -> beam={coarse_choice} | AGREE={agrees}", flush=True)
            if path_clearances:
                print(f"  coarse path clearance profile (next 40 cells): {path_clearances}", flush=True)

            results.append({
                "layout": layout_name, "seed": seed, "check_tick": target_tick,
                "obstacle_aware_angle": obstacle_aware_angle, "obstacle_aware_beam_choice": obstacle_aware_choice,
                "coarse_angle": coarse_angle, "coarse_beam_choice": coarse_choice, "agrees": agrees,
                "coarse_path_clearance_profile": path_clearances,
            })
            next_checkpoint_idx += 1
        env.close()

    (ROOT / "evaluations" / "coarse_route_proof_of_mechanism_v2.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8",
    )
    n_disagree = sum(1 for r in results if not r["agrees"])
    print(f"\n=== CORRECTED SUMMARY: {n_disagree}/{len(results)} decision points where the properly-computed "
          f"coarse route's beam choice disagrees with the standard reactive target's beam choice ===", flush=True)
    print("\nSaved.", flush=True)


if __name__ == "__main__":
    main()
