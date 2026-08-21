"""Check whether the coarse clearance-aware route, computed from an EARLIER
point in the approach (well before the local beam's ~8-tick horizon could
see the eventual clearance=1 dead end), would have avoided the pocket that
trapped 12_early_open_center_high_bursty seed1 for 450+ ticks.

This is a much clearer test case than the original 3-point check: clearance
declines monotonically (34 -> 13 -> 17 -> 12 -> 9 -> 1) over ticks 100-155,
well outside the beam's effective horizon, before the agent gets fully
pinned. Diagnostic only.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simulator.curriculum_manifests import load_heldout_manifest
from .scratchpad_diagnose_v3_terminal_gate_onsets import record_trace
from .scratchpad_coarse_route_proof_of_mechanism import compute_clearance_field, coarse_route, _target_position
from simulator.synthetic import iter_variant_environments
from simulator.scripted_policies import _event_for
from simulator.steering_oracle import _oracle_steering_decision_v3, DEFAULT_ROBUST_SIGMA, DEFAULT_BEAM_DEPTH, DEFAULT_BEAM_WIDTH, DEFAULT_CONTINUATION_DEPTH
import numpy as np

manifest = load_heldout_manifest("simulator/evaluations/manifests/oracle_fresh_confirmation.json")
LAYOUT, SEED = "12_early_open_center_high_bursty", 1
CHECK_TICKS = [100, 130, 140]

entry, env = next(iter(iter_variant_environments(
    manifest.curriculum_path, stage=manifest.stage, seed=SEED, episode_steps=1000,
    episode_seconds=150.0, variant_name=LAYOUT,
)))
obs, _ = env.reset(seed=SEED)
prev_action = None
tick = 0
map_model = env.map
clearance_field = compute_clearance_field(map_model.traversable)

for target_tick in CHECK_TICKS:
    while tick < target_tick:
        action, _ = _oracle_steering_decision_v3(
            env, sigma=DEFAULT_ROBUST_SIGMA, beam_depth=DEFAULT_BEAM_DEPTH, beam_width=DEFAULT_BEAM_WIDTH,
            previous_action=prev_action, stage=manifest.stage, continuation_depth=DEFAULT_CONTINUATION_DEPTH,
        )
        obs, r, term, trunc, info = env.step(np.asarray([int(action), int(_event_for(env))], dtype=np.int64))
        prev_action = action
        tick += 1

    target = _target_position(env)
    print(f"\ntick {tick}: player=({env.player_x:.1f},{env.player_z:.1f}) target={target}", flush=True)
    if target is None:
        print("  no reachable target", flush=True)
        continue
    start_cell = map_model.native_to_layout_cell(env.player_x, env.player_z)
    target_cell = map_model.native_to_layout_cell(target[0], target[1])
    if start_cell is None or target_cell is None:
        print("  target/start cell out of bounds", flush=True)
        continue
    path = coarse_route(map_model.traversable, clearance_field, start_cell, target_cell)
    if not path:
        print("  no coarse route found", flush=True)
        continue
    clearances = [int(clearance_field[y, x]) for x, y in path]
    min_clearance = min(clearances)
    print(f"  coarse route length={len(path)} cells, min_clearance_along_route={min_clearance}, "
          f"clearance_profile(first 30)={clearances[:30]}", flush=True)
    # Does the coarse route pass anywhere near the eventual trap cell (5,100)?
    trap_cell = (5, 100)
    near_trap = any(abs(cx - trap_cell[0]) <= 3 and abs(cy - trap_cell[1]) <= 3 for cx, cy in path)
    print(f"  route passes within 3 cells of the eventual trap {trap_cell}: {near_trap}", flush=True)

env.close()
print("\nDONE", flush=True)
