"""Fresh, untouched confirmation evaluation for the terminal-gated v3
steering oracle (escape-BFS robust-safety fix + continuation_depth=4), per
the 2026-08-09 user correction: the early_heldout/unseen_templates/challenge
pools are development pools now (they directly informed the terminal gate,
the escape-BFS fix, and the continuation_depth selection), so a materially-
improving result there makes the oracle a CANDIDATE teacher, not a qualified
one. This evaluates on evaluations/manifests/oracle_fresh_confirmation.json
(12 layouts, seed base 23,000,000, confirmed disjoint from every seed range
used by training/heldout/unseen_templates/challenge), with a stricter bar
than Basic's own later graduation tolerance: essentially zero physical
collision events, reported PER EPISODE (not just per-layout aggregates),
since "near zero" needs per-episode visibility to actually verify.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from simulator.milestone_evaluator import _contact_event_stats
from simulator.movement_classification import classify_episode_movement
from simulator.steering_oracle import SteeringOracleTeacherV3
from simulator.synthetic import iter_variant_environments
from simulator.curriculum_manifests import load_heldout_manifest

SIGMA = 1.5
EVAL_SEEDS = [0, 1]
EPISODE_SECONDS = 150.0
MAX_ACTIONS = 1000

manifest = load_heldout_manifest("evaluations/manifests/oracle_fresh_confirmation.json")

results = []
for layout_name in manifest.layouts:
    for seed in EVAL_SEEDS:
        entry, env = next(iter(iter_variant_environments(
            manifest.curriculum_path, stage=manifest.stage, seed=seed, episode_steps=MAX_ACTIONS,
            episode_seconds=EPISODE_SECONDS, variant_name=layout_name,
        )))
        obs, _ = env.reset(seed=seed)
        oracle = SteeringOracleTeacherV3(stage=manifest.stage, sigma=SIGMA)

        steering_choices, unique_cells_trace, total_distance_trace, contacts_trace = [], [], [], []
        info = {}
        for _tick in range(MAX_ACTIONS):
            cmd = oracle.command(env)
            steering_choices.append(int(cmd.steering))
            obs, r, term, trunc, info = env.step(np.asarray([int(cmd.steering), int(cmd.event)], dtype=np.int64))
            unique_cells_trace.append(int(info["unique_cells"]))
            total_distance_trace.append(float(info["total_distance_cells"]))
            contacts_trace.append(int(info["contacts"]))
            if term or trunc:
                break
        env.close()

        movement = classify_episode_movement(
            steering_choices=steering_choices, unique_cells_trace=unique_cells_trace, total_distance_trace=total_distance_trace,
        )
        contact_stats = _contact_event_stats(contacts_trace)
        rec = {
            "layout": layout_name, "seed": seed, "steps": len(steering_choices),
            "total_kills": int(info.get("total_kills", 0)), "unique_cells": int(info.get("unique_cells", 0)),
            "kills_per_simulated_hour": float(info.get("total_kills", 0)) * 3600.0 / max(1e-9, float(info.get("elapsed_seconds", 0.0))),
            "fallback_rate": oracle.fallback_rate,
            **movement, **contact_stats,
        }
        results.append(rec)
        flag = "CLEAN" if not rec["any_contact"] else "CONTACT"
        print(f"[{flag}] {layout_name}/seed{seed}: distinct_events={rec['distinct_contact_events']} "
              f"contact_ticks={rec['total_contact_ticks']} max_consec={rec['max_consecutive_contact_ticks']} "
              f"stagn={rec['physical_stagnation']} kills={rec['total_kills']} unique_cells={rec['unique_cells']} "
              f"fallback_rate={rec['fallback_rate']:.3f}", flush=True)

out_path = ROOT / "evaluations" / "oracle_fresh_confirmation_qualification.json"
out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
print(f"\nSaved: {out_path}", flush=True)

n = len(results)
any_contact = sum(1 for r in results if r["any_contact"])
total_events = sum(r["distinct_contact_events"] for r in results)
total_ticks = sum(r["total_contact_ticks"] for r in results)
max_consec = max(r["max_consecutive_contact_ticks"] for r in results)
stagn = sum(1 for r in results if r["physical_stagnation"])
zero_kill = sum(1 for r in results if r["total_kills"] == 0)
clean_episodes = [r for r in results if not r["any_contact"]]

print(f"\n=== FRESH CONFIRMATION SUMMARY (n={n} episodes, {len(manifest.layouts)} layouts x {len(EVAL_SEEDS)} seeds) ===", flush=True)
print(f"episodes_with_any_contact: {any_contact}/{n} ({100*any_contact/n:.1f}%)", flush=True)
print(f"clean (zero-contact) episodes: {len(clean_episodes)}/{n} ({100*len(clean_episodes)/n:.1f}%)", flush=True)
print(f"total_distinct_collision_events: {total_events} (mean {total_events/n:.2f}/episode)", flush=True)
print(f"total_contact_ticks: {total_ticks}", flush=True)
print(f"max_consecutive_contact_ticks (worst episode): {max_consec}", flush=True)
print(f"physical_stagnation_episodes: {stagn}", flush=True)
print(f"zero_kill_episodes: {zero_kill}", flush=True)
print(f"median_kills_per_hour: {np.median([r['kills_per_simulated_hour'] for r in results]):.0f}", flush=True)
print(f"median_unique_cells: {np.median([r['unique_cells'] for r in results]):.0f}", flush=True)
print(f"median_fallback_rate: {np.median([r['fallback_rate'] for r in results]):.3f}", flush=True)
print("\nGRAND TOTAL DONE", flush=True)
