"""Step 3: matched evaluation of target-selection hysteresis (Step 2)
against the pre-hysteresis baseline, on identical seeds. Per the user's
spec: the catastrophic 517-tick-lock episode, several high-switch-rate
episodes, and several low-switch-rate controls (from the Step 1 thrashing
measurement).

Reports: distinct collision events, contact ticks, max contact run,
fallback rate, longest fallback streak, target switches (raw + material),
kills, coverage (unique cells), stagnation.

Only the environment's target_hysteresis_enabled flag differs between the
two conditions -- sigma, beam depth/width, continuation_depth, and
everything else in steering_oracle.py stays exactly as-is, per the standing
instruction to isolate this one variable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from simulator.curriculum_manifests import load_heldout_manifest
from simulator.milestone_evaluator import _contact_event_stats
from simulator.movement_classification import classify_episode_movement
from simulator.synthetic import iter_variant_environments
from simulator.scripted_policies import _event_for
from simulator.steering_oracle import (
    _oracle_steering_decision_v3, DEFAULT_ROBUST_SIGMA, DEFAULT_BEAM_DEPTH,
    DEFAULT_BEAM_WIDTH, DEFAULT_CONTINUATION_DEPTH,
)

SIGMA = DEFAULT_ROBUST_SIGMA
EPISODE_SECONDS = 150.0
MAX_ACTIONS = 1000

# From Step 1's thrashing measurement (evaluations/target_thrashing_aggregate_report.json)
TARGET_EPISODES = [
    ("12_early_open_center_high_bursty", 1, "catastrophic 517-tick lock"),
    ("06_early_broad_lobes_high_bursty", 0, "high switch rate (41.2/100t)"),
    ("02_early_open_field_high_bursty", 0, "high switch rate (40.5/100t)"),
    ("03_early_irregular_plain_typical_fast", 0, "low switch rate (24.8/100t) control"),
    ("10_early_wide_neck_high_bursty", 1, "low switch rate (25.5/100t) control"),
]


def run_episode(curriculum_path, layout_name, seed, stage, hysteresis_enabled):
    entry, env = next(iter(iter_variant_environments(
        curriculum_path, stage=stage, seed=seed, episode_steps=MAX_ACTIONS,
        episode_seconds=EPISODE_SECONDS, variant_name=layout_name,
    )))
    obs, _ = env.reset(seed=seed)
    env.target_hysteresis_enabled = hysteresis_enabled
    prev_action = None
    prev_contacts = 0
    prev_target_id = None
    steering_choices, unique_cells_trace, total_distance_trace, contacts_trace = [], [], [], []
    fallback_trace = []
    switches = 0
    info = {}
    for _tick in range(MAX_ACTIONS):
        action, used_fallback = _oracle_steering_decision_v3(
            env, sigma=SIGMA, beam_depth=DEFAULT_BEAM_DEPTH, beam_width=DEFAULT_BEAM_WIDTH,
            previous_action=prev_action, stage=stage, continuation_depth=DEFAULT_CONTINUATION_DEPTH,
        )
        cur_target_id = env._nearest_reachable_actor_id
        if cur_target_id is None:
            cur_target_id = env._best_group_actor_id
        if cur_target_id != prev_target_id:
            switches += 1
        prev_target_id = cur_target_id

        steering_choices.append(int(action))
        obs, r, term, trunc, info = env.step(np.asarray([int(action), int(_event_for(env))], dtype=np.int64))
        unique_cells_trace.append(int(info["unique_cells"]))
        total_distance_trace.append(float(info["total_distance_cells"]))
        contacts_trace.append(int(info["contacts"]))
        fallback_trace.append(bool(used_fallback))
        prev_action = action
        if term or trunc:
            break
    env.close()

    movement = classify_episode_movement(
        steering_choices=steering_choices, unique_cells_trace=unique_cells_trace, total_distance_trace=total_distance_trace,
    )
    contact_stats = _contact_event_stats(contacts_trace)

    fallback_ticks = sum(1 for f in fallback_trace if f)
    longest_fallback_streak = 0
    cur = 0
    for f in fallback_trace:
        cur = cur + 1 if f else 0
        longest_fallback_streak = max(longest_fallback_streak, cur)

    return {
        "steps": len(steering_choices), "total_kills": int(info.get("total_kills", 0)),
        "unique_cells": int(info.get("unique_cells", 0)), "target_switches": switches,
        "target_switches_per_100_ticks": 100.0 * switches / max(1, len(steering_choices)),
        "fallback_ticks": fallback_ticks, "fallback_rate": fallback_ticks / max(1, len(steering_choices)),
        "longest_fallback_streak": longest_fallback_streak,
        **movement, **contact_stats,
    }


def main():
    manifest = load_heldout_manifest("simulator/evaluations/manifests/oracle_fresh_confirmation.json")
    results = []
    for layout_name, seed, note in TARGET_EPISODES:
        baseline = run_episode(manifest.curriculum_path, layout_name, seed, manifest.stage, hysteresis_enabled=False)
        hysteresis = run_episode(manifest.curriculum_path, layout_name, seed, manifest.stage, hysteresis_enabled=True)
        rec = {"layout": layout_name, "seed": seed, "note": note, "baseline": baseline, "hysteresis": hysteresis}
        results.append(rec)
        print(f"\n--- {layout_name}/seed{seed} [{note}] ---", flush=True)
        print(f"  baseline:   events={baseline['distinct_contact_events']} ticks={baseline['total_contact_ticks']} "
              f"max_consec={baseline['max_consecutive_contact_ticks']} fallback_rate={baseline['fallback_rate']:.3f} "
              f"longest_fallback={baseline['longest_fallback_streak']} switches={baseline['target_switches']} "
              f"({baseline['target_switches_per_100_ticks']:.1f}/100t) kills={baseline['total_kills']} "
              f"unique_cells={baseline['unique_cells']} stagn={baseline['physical_stagnation']}", flush=True)
        print(f"  hysteresis: events={hysteresis['distinct_contact_events']} ticks={hysteresis['total_contact_ticks']} "
              f"max_consec={hysteresis['max_consecutive_contact_ticks']} fallback_rate={hysteresis['fallback_rate']:.3f} "
              f"longest_fallback={hysteresis['longest_fallback_streak']} switches={hysteresis['target_switches']} "
              f"({hysteresis['target_switches_per_100_ticks']:.1f}/100t) kills={hysteresis['total_kills']} "
              f"unique_cells={hysteresis['unique_cells']} stagn={hysteresis['physical_stagnation']}", flush=True)

    out_path = ROOT / "simulator" / "evaluations" / "matched_eval_target_hysteresis.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    print("\n=== SUMMARY ===", flush=True)
    total_base_events = sum(r["baseline"]["distinct_contact_events"] for r in results)
    total_hyst_events = sum(r["hysteresis"]["distinct_contact_events"] for r in results)
    total_base_ticks = sum(r["baseline"]["total_contact_ticks"] for r in results)
    total_hyst_ticks = sum(r["hysteresis"]["total_contact_ticks"] for r in results)
    total_base_switches = sum(r["baseline"]["target_switches"] for r in results)
    total_hyst_switches = sum(r["hysteresis"]["target_switches"] for r in results)
    print(f"total distinct events: baseline={total_base_events} hysteresis={total_hyst_events}", flush=True)
    print(f"total contact ticks:    baseline={total_base_ticks} hysteresis={total_hyst_ticks}", flush=True)
    print(f"total target switches:  baseline={total_base_switches} hysteresis={total_hyst_switches}", flush=True)
    print("\nSaved.", flush=True)


if __name__ == "__main__":
    main()
