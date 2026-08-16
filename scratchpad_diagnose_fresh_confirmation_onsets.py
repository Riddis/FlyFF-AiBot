"""Cheap, targeted causal diagnosis on the WORST fresh-confirmation-pool
episodes (not all 24 -- that would cost as much as the aborted broad run).
Question: does the SAME mechanism (fallback persistence / genuinely-cornered,
established on the tuning pools) still dominate on genuinely fresh geometry,
or is something new happening that the continuation_depth=4 fix doesn't
address? terminal_gate only -- no plain_v3 comparison needed for this
question. Diagnostic only, does not modify steering_oracle.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from simulator.steering_oracle import _CANDIDATES, _robust_envelope_safe, DEFAULT_ROBUST_SIGMA
from simulator.curriculum_manifests import load_heldout_manifest
from scratchpad_diagnose_v3_terminal_gate_onsets import record_trace, find_onsets, analyze_onset

SIGMA = DEFAULT_ROBUST_SIGMA

# The two worst fresh-pool episodes seen so far (7 and 8 distinct events).
TARGETS = [
    ("03_early_irregular_plain_typical_fast", 0, "7 events in fresh qualification"),
    ("03_early_irregular_plain_typical_fast", 1, "8 events in fresh qualification"),
]


def main() -> None:
    manifest = load_heldout_manifest("evaluations/manifests/oracle_fresh_confirmation.json")
    all_results = []
    zero_robust = 0
    some_robust = 0
    from collections import Counter
    category_counts = Counter()

    for layout_name, seed, note in TARGETS:
        trace, live_env = record_trace(manifest.curriculum_path, layout_name, seed, manifest.stage, "terminal_gate")
        onsets = find_onsets(trace)
        map_model = live_env.map
        movement_models = live_env.model.movement
        print(f"{layout_name}/seed{seed}[{note}]: {len(onsets)} onsets", flush=True)
        for onset_idx in onsets:
            rec = analyze_onset(live_env, trace, onset_idx)
            category_counts[rec["category"]] += 1
            s = trace[onset_idx]
            robust_count = sum(
                1 for a in _CANDIDATES
                if _robust_envelope_safe(map_model, movement_models, s["x"], s["z"], s["heading"], a, sigma=SIGMA)[0]
            )
            if robust_count == 0:
                zero_robust += 1
            else:
                some_robust += 1
            all_results.append({
                "layout": layout_name, "seed": seed, "onset_tick": onset_idx,
                "category": rec["category"], "fallback_streak_ticks_before_onset": rec["fallback_streak_ticks_before_onset"],
                "robust_count_at_onset": robust_count,
            })
        live_env.close()

    total = zero_robust + some_robust
    print(f"\n=== Fresh-pool worst-case onset diagnosis (n={total} onsets) ===", flush=True)
    print("Classification:", flush=True)
    for cat, n in category_counts.most_common():
        print(f"  {cat}: {n} ({100*n/total:.1f}%)", flush=True)
    print(f"Robust options at collision tick: zero={zero_robust} ({100*zero_robust/total:.1f}%) "
          f"some={some_robust} ({100*some_robust/total:.1f}%)", flush=True)

    (ROOT / "evaluations" / "oracle_fresh_confirmation_onset_diagnosis.json").write_text(
        json.dumps(all_results, indent=2, default=str), encoding="utf-8",
    )
    print("\nSaved.", flush=True)


if __name__ == "__main__":
    main()
