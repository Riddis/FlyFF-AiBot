"""Fills the 2-episode gap left by a sharding mismatch (shard0's smoke test
used n_shards=24 while the full run used n_shards=8, leaving indices 8 and
16 -- 05_early_broad_lobes_typical_fast seed0 and 09_early_wide_neck_
typical_fast seed0 -- uncovered)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulator.curriculum_manifests import load_heldout_manifest
from scratchpad_measure_target_thrashing import run_episode, analyze_episode

manifest = load_heldout_manifest("evaluations/manifests/oracle_fresh_confirmation.json")
MISSING = [("05_early_broad_lobes_typical_fast", 0), ("09_early_wide_neck_typical_fast", 0)]

results = []
for layout_name, seed in MISSING:
    trace = run_episode(manifest.curriculum_path, layout_name, seed, manifest.stage)
    analysis = analyze_episode(layout_name, seed, trace)
    results.append(analysis)
    print(f"{layout_name}/seed{seed}: switches={analysis['total_switches']} "
          f"({analysis['switches_per_100_ticks']:.1f}/100t) onsets={analysis['n_onsets']} "
          f"fallback_streaks={analysis['n_fallback_streaks']}", flush=True)

out_path = ROOT / "evaluations" / "target_thrashing_measurement_shard_missing.json"
out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
print(f"Saved: {out_path}", flush=True)
print("SHARD DONE", flush=True)
