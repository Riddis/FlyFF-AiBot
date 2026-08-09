"""Aggregate the 8 target-thrashing measurement shards into the full Step-1
report requested: switches/100 ticks, lookback-window switch counts before
onsets and fallback-streak starts, fallback-duration vs pre-streak switch
count, dead-vs-live-target switch breakdown, material vs raw switches.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np

all_episodes = []
for shard_idx in range(8):
    path = ROOT / "evaluations" / f"target_thrashing_measurement_shard{shard_idx}.json"
    all_episodes.extend(json.loads(path.read_text(encoding="utf-8")))
missing_path = ROOT / "evaluations" / "target_thrashing_measurement_shard_missing.json"
if missing_path.exists():
    all_episodes.extend(json.loads(missing_path.read_text(encoding="utf-8")))

print(f"Total episodes aggregated: {len(all_episodes)}\n")

# --- Whole-episode switch rates ---
switch_rates = [e["switches_per_100_ticks"] for e in all_episodes]
material_rates = [e["material_switches_per_100_ticks"] for e in all_episodes]
print("=== Whole-episode switch rates ===")
print(f"switches/100ticks: median={np.median(switch_rates):.1f} mean={np.mean(switch_rates):.1f} "
      f"min={min(switch_rates):.1f} max={max(switch_rates):.1f}")
print(f"material switches/100ticks: median={np.median(material_rates):.1f} mean={np.mean(material_rates):.1f}")

total_dead = sum(e["dead_target_switches"] for e in all_episodes)
total_live = sum(e["live_target_switches"] for e in all_episodes)
print(f"\ndead-target switches (old target died): {total_dead}")
print(f"live-target switches (old target still valid, preference-driven): {total_live}")
print(f"fraction preference-driven: {100*total_live/(total_dead+total_live):.1f}%")

# --- Lookback window analysis: switches before onsets ---
print("\n=== Switches in the N ticks BEFORE each collision onset (pooled across all episodes) ===")
for w in (5, 10, 20):
    all_vals = []
    all_material_vals = []
    for e in all_episodes:
        all_vals.extend(e["onset_lookback_switches"][str(w)])
        all_material_vals.extend(e["onset_lookback_material_switches"][str(w)])
    if all_vals:
        print(f"  window={w}: n_onsets={len(all_vals)} median_switches={np.median(all_vals):.1f} "
              f"mean_switches={np.mean(all_vals):.2f} median_material={np.median(all_material_vals):.1f} "
              f"mean_material={np.mean(all_material_vals):.2f} "
              f"pct_with_zero_switches={100*sum(1 for v in all_vals if v==0)/len(all_vals):.1f}%")

# --- Lookback window analysis: switches before fallback-streak starts ---
print("\n=== Switches in the N ticks BEFORE each fallback-streak entry (pooled) ===")
for w in (5, 10, 20):
    all_vals = []
    all_material_vals = []
    for e in all_episodes:
        all_vals.extend(e["streak_lookback_switches"][str(w)])
        all_material_vals.extend(e["streak_lookback_material_switches"][str(w)])
    if all_vals:
        print(f"  window={w}: n_streaks={len(all_vals)} median_switches={np.median(all_vals):.1f} "
              f"mean_switches={np.mean(all_vals):.2f} median_material={np.median(all_material_vals):.1f} "
              f"mean_material={np.mean(all_material_vals):.2f} "
              f"pct_with_zero_switches={100*sum(1 for v in all_vals if v==0)/len(all_vals):.1f}%")

# --- Fallback duration vs pre-streak switch count (10-tick window) ---
print("\n=== Fallback-streak duration vs switches in the preceding 10 ticks ===")
pairs = []
for e in all_episodes:
    durations = e["fallback_streak_durations"]
    pre_switches = e["streak_lookback_switches"]["10"]
    for d, s in zip(durations, pre_switches):
        pairs.append((s, d))
if pairs:
    high = [d for s, d in pairs if s >= 2]
    low = [d for s, d in pairs if s < 2]
    print(f"streaks preceded by >=2 switches in prior 10 ticks: n={len(high)} "
          f"median_duration={np.median(high) if high else 'n/a'} mean_duration={np.mean(high) if high else 'n/a':.1f}")
    print(f"streaks preceded by <2 switches in prior 10 ticks:  n={len(low)} "
          f"median_duration={np.median(low) if low else 'n/a'} mean_duration={np.mean(low) if low else 'n/a':.1f}")
    corr = np.corrcoef([s for s, d in pairs], [d for s, d in pairs])[0, 1] if len(pairs) > 1 else float("nan")
    print(f"correlation(pre-streak switches, streak duration): {corr:.3f}")

# --- Per-episode table for reference ---
print("\n=== Per-episode summary ===")
for e in sorted(all_episodes, key=lambda x: (x["layout"], x["seed"])):
    print(f"  {e['layout']}/seed{e['seed']}: switches={e['total_switches']} "
          f"({e['switches_per_100_ticks']:.1f}/100t) onsets={e['n_onsets']} streaks={e['n_fallback_streaks']} "
          f"streak_durations={e['fallback_streak_durations']}")

out_path = ROOT / "evaluations" / "target_thrashing_aggregate_report.json"
out_path.write_text(json.dumps(all_episodes, indent=2, default=str), encoding="utf-8")
print(f"\nSaved aggregate: {out_path}")
