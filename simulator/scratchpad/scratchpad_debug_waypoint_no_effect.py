"""Debug why waypoint-steering produced IDENTICAL contact counts to the
historical replay in scratchpad_coarse_route_rollout_verification.py --
identical, not just similar, which is suspicious since a genuinely
different first action should make every subsequent tick diverge.
Cheap: replays only up to check_tick (not the full 40-tick window), then
inspects the beam's actual candidate set under both target angles directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from farming.actions import SteeringAction
from simulator.local_clearance import sample_heading_relative_clearance
from simulator.steering_oracle import (
    _beam_search_first_action, _oracle_steering_decision_v3, _robust_envelope_safe, _terminal_viability, _CANDIDATES,
    DEFAULT_ROBUST_SIGMA, DEFAULT_BEAM_DEPTH, DEFAULT_BEAM_WIDTH, DEFAULT_CONTINUATION_DEPTH,
)
from simulator.synthetic import iter_variant_environments
from simulator.curriculum_manifests import load_heldout_manifest
from simulator.scripted_policies import _obstacle_aware_target_angle, _event_for
from .scratchpad_coarse_route_proof_of_mechanism import waypoint_direction, _target_position

SIGMA = DEFAULT_ROBUST_SIGMA
EPISODE_SECONDS = 150.0
MAX_ACTIONS = 1000

CASES = [
    ("03_early_irregular_plain_typical_fast", 0, 162),
    ("03_early_irregular_plain_typical_fast", 1, 52),
]

manifest = load_heldout_manifest("simulator/evaluations/manifests/oracle_fresh_confirmation.json")

for layout_name, seed, check_tick in CASES:
    entry, env = next(iter(iter_variant_environments(
        manifest.curriculum_path, stage=manifest.stage, seed=seed, episode_steps=MAX_ACTIONS,
        episode_seconds=EPISODE_SECONDS, variant_name=layout_name,
    )))
    obs, _ = env.reset(seed=seed)
    prev_action = None
    for tick in range(check_tick):
        action, _ = _oracle_steering_decision_v3(
            env, sigma=SIGMA, beam_depth=DEFAULT_BEAM_DEPTH, beam_width=DEFAULT_BEAM_WIDTH,
            previous_action=prev_action, stage=manifest.stage, continuation_depth=DEFAULT_CONTINUATION_DEPTH,
        )
        obs, r, term, trunc, info = env.step(np.asarray([int(action), int(_event_for(env))], dtype=np.int64))
        prev_action = action

    # Now at check_tick's decision point -- inspect directly.
    obstacle_aware_angle = _obstacle_aware_target_angle(env)
    target = _target_position(env)
    coarse_angle = waypoint_direction(env.map, env.player_x, env.player_z, env.heading, target[0], target[1]) if target else None

    print(f"\n=== {layout_name}/seed{seed} tick{check_tick} ===", flush=True)
    print(f"obstacle_aware target_angle={obstacle_aware_angle}", flush=True)
    print(f"coarse waypoint target_angle={coarse_angle}", flush=True)
    print(f"player_x={env.player_x:.2f} player_z={env.player_z:.2f} heading={env.heading:.3f}", flush=True)

    for label, target_angle in (("obstacle_aware", obstacle_aware_angle), ("coarse_waypoint", coarse_angle)):
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
        # How many depth-1 immediate candidates are robustly safe at all?
        robust_immediate = [a for a in _CANDIDATES if _robust_envelope_safe(env.map, env.model.movement, env.player_x, env.player_z, env.heading, a, sigma=SIGMA)[0]]
        print(f"  [{label}] target_angle={target_angle} -> beam_chosen={SteeringAction(chosen).name if chosen is not None else None} "
              f"robust_immediate_candidates={[SteeringAction(a).name for a in robust_immediate]}", flush=True)
    env.close()
