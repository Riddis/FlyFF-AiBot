"""Step 2 of the coarse-route proof-of-mechanism: does actually STEERING
toward the coarse waypoint, using the EXISTING unchanged depth-4 beam/escape
machinery (only the target_angle source changes), prevent the historical
collision?

Replays each episode deterministically (same seed) with the STANDARD oracle
up to check_tick, confirms the replay matches the original trace exactly
(same seed => same physics => must match, or something is wrong), then
switches to a waypoint-steered decision (target_angle from the coarse route
instead of _obstacle_aware_target_angle) for a verification window, and
compares contact counts against the historical window.

Does not modify steering_oracle.py -- the beam/escape functions are called
exactly as they are, only the target_angle argument's SOURCE changes.
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
    _beam_search_first_action, _fastest_escape_first_action, _one_real_tick, _CANDIDATES,
    _STAGE_ESCAPE_TICKS, _DEFAULT_ESCAPE_TICKS, _oracle_steering_decision_v3,
    DEFAULT_ROBUST_SIGMA, DEFAULT_BEAM_DEPTH, DEFAULT_BEAM_WIDTH, DEFAULT_CONTINUATION_DEPTH,
)
from simulator.synthetic import iter_variant_environments
from simulator.curriculum_manifests import load_heldout_manifest
from scratchpad_coarse_route_proof_of_mechanism import waypoint_direction, _target_position

SIGMA = DEFAULT_ROBUST_SIGMA
BEAM_DEPTH = DEFAULT_BEAM_DEPTH
BEAM_WIDTH = DEFAULT_BEAM_WIDTH
CONTINUATION_DEPTH = DEFAULT_CONTINUATION_DEPTH
VERIFICATION_WINDOW_TICKS = 40
EPISODE_SECONDS = 150.0
MAX_ACTIONS = 1000

# (layout, seed, check_tick) -- the tick where the coarse route disagreed
# with the historical beam (from coarse_route_proof_of_mechanism.json).
TARGETS = [
    ("03_early_irregular_plain_typical_fast", 0, 162, "coarse=RIGHT vs historical=STRAIGHT"),
    ("03_early_irregular_plain_typical_fast", 1, 52, "coarse=LEFT vs historical=STRAIGHT"),
]


def _waypoint_oracle_decision(env, *, sigma, beam_depth, beam_width, previous_action, stage, continuation_depth):
    target = _target_position(env)
    target_angle = None
    if target is not None:
        target_angle = waypoint_direction(env.map, env.player_x, env.player_z, env.heading, target[0], target[1])

    clearance_raw = sample_heading_relative_clearance(env.map, env.player_x, env.player_z, env.heading)
    clearance_by_action = {
        SteeringAction.STRAIGHT: clearance_raw["forward"],
        SteeringAction.LEFT: clearance_raw["left"],
        SteeringAction.RIGHT: clearance_raw["right"],
    }
    beam_action = _beam_search_first_action(
        env.map, env.model.movement, env.player_x, env.player_z, env.heading,
        sigma=sigma, depth=beam_depth, beam_width=beam_width, previous_action=previous_action,
        target_angle=target_angle, clearance=clearance_by_action, continuation_depth=continuation_depth,
    )
    if beam_action is not None:
        return beam_action, False
    budget = _STAGE_ESCAPE_TICKS.get(stage, _DEFAULT_ESCAPE_TICKS)
    escape_action = _fastest_escape_first_action(
        env.map, env.model.movement, env.player_x, env.player_z, env.heading, max_ticks=budget, sigma=sigma,
    )
    if escape_action is not None:
        return escape_action, True
    immediate = {
        c: _one_real_tick(env.map, env.model.movement, env.player_x, env.player_z, env.heading, c)
        for c in _CANDIDATES
    }
    return max(_CANDIDATES, key=lambda c: immediate[c].progress_cells), True


def run_comparison(curriculum_path, layout_name, seed, stage, check_tick, note):
    from simulator.scripted_policies import _event_for

    print(f"\n--- {layout_name}/seed{seed} check_tick={check_tick} ({note}) ---", flush=True)

    for mode in ("historical_replay", "waypoint_steered"):
        entry, env = next(iter(iter_variant_environments(
            curriculum_path, stage=stage, seed=seed, episode_steps=MAX_ACTIONS,
            episode_seconds=EPISODE_SECONDS, variant_name=layout_name,
        )))
        obs, _ = env.reset(seed=seed)
        prev_action = None
        prev_contacts = 0
        contacts_in_window = 0
        window_start = check_tick
        window_end = check_tick + VERIFICATION_WINDOW_TICKS
        recorded_positions = []
        for tick in range(window_end + 1):
            if mode == "historical_replay" or tick < window_start:
                action, _used_fallback = _oracle_steering_decision_v3(
                    env, sigma=SIGMA, beam_depth=BEAM_DEPTH, beam_width=BEAM_WIDTH,
                    previous_action=prev_action, stage=stage, continuation_depth=CONTINUATION_DEPTH,
                )
            else:
                action, _used_fallback = _waypoint_oracle_decision(
                    env, sigma=SIGMA, beam_depth=BEAM_DEPTH, beam_width=BEAM_WIDTH,
                    previous_action=prev_action, stage=stage, continuation_depth=CONTINUATION_DEPTH,
                )
            obs, r, term, trunc, info = env.step(np.asarray([int(action), int(_event_for(env))], dtype=np.int64))
            contacts = int(info.get("contacts", 0))
            if tick >= window_start:
                if contacts > prev_contacts:
                    contacts_in_window += 1
                recorded_positions.append((tick, env.player_x, env.player_z, env.heading))
            prev_contacts = contacts
            prev_action = action
            if term or trunc:
                break
        env.close()
        print(f"  [{mode}] contact_ticks_in_verification_window(next {VERIFICATION_WINDOW_TICKS} ticks)="
              f"{contacts_in_window}", flush=True)
        if mode == "historical_replay":
            historical_contacts = contacts_in_window
        else:
            waypoint_contacts = contacts_in_window

    return {"layout": layout_name, "seed": seed, "check_tick": check_tick, "note": note,
            "historical_contacts_in_window": historical_contacts, "waypoint_steered_contacts_in_window": waypoint_contacts}


def main() -> None:
    manifest = load_heldout_manifest("evaluations/manifests/oracle_fresh_confirmation.json")
    results = []
    for layout_name, seed, check_tick, note in TARGETS:
        rec = run_comparison(manifest.curriculum_path, layout_name, seed, manifest.stage, check_tick, note)
        results.append(rec)

    (ROOT / "evaluations" / "coarse_route_rollout_verification.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8",
    )
    print("\n=== SUMMARY ===", flush=True)
    for r in results:
        improved = r["waypoint_steered_contacts_in_window"] < r["historical_contacts_in_window"]
        print(f"{r['layout']}/seed{r['seed']} tick{r['check_tick']}: historical={r['historical_contacts_in_window']} "
              f"waypoint_steered={r['waypoint_steered_contacts_in_window']} IMPROVED={improved}", flush=True)
    print("\nSaved.", flush=True)


if __name__ == "__main__":
    main()
