"""Targeted follow-up to the 2026-08-09 escape-BFS robust-safety fix: that fix
only marginally reduced total contact ticks (913->869 on the 14-episode
matched set) and left the onset classification almost unchanged (98.2% still
C_fallback_persistence). This script answers the specific question needed to
decide the next step: at the exact pre-collision tick of each onset, was ANY
of the 3 actions robustly safe at all?

- If "genuinely cornered" (zero robust options) dominates: the fix cannot
  help there by construction (it explicitly falls back to old behavior in
  that case) -- points at an UPSTREAM routing/lookahead issue (the beam is
  driving into corners tight enough that nothing downstream can avoid
  contact), not a downstream escape-execution bug.
- If a robust option existed but contact still happened anyway: points at a
  bug in the fix's execution path or a residual stochastic-tail issue, not
  lookahead depth.

Diagnostic only -- does not modify steering_oracle.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from simulator.steering_oracle import _CANDIDATES, _robust_envelope_safe, DEFAULT_ROBUST_SIGMA
from simulator.curriculum_manifests import load_heldout_manifest
from scratchpad_diagnose_v3_terminal_gate_onsets import record_trace, find_onsets, TARGET_LAYOUTS

SIGMA = DEFAULT_ROBUST_SIGMA


def main() -> None:
    zero_robust = 0
    some_robust = 0
    details = []
    for manifest_path, layout_name, note in TARGET_LAYOUTS:
        manifest = load_heldout_manifest(manifest_path)
        for seed in (0, 1):
            trace, live_env = record_trace(manifest.curriculum_path, layout_name, seed, manifest.stage, "terminal_gate")
            onsets = find_onsets(trace)
            map_model = live_env.map
            movement_models = live_env.model.movement
            for onset_idx in onsets:
                s = trace[onset_idx]
                robust_count = sum(
                    1 for a in _CANDIDATES
                    if _robust_envelope_safe(map_model, movement_models, s["x"], s["z"], s["heading"], a, sigma=SIGMA)[0]
                )
                if robust_count == 0:
                    zero_robust += 1
                else:
                    some_robust += 1
                details.append({
                    "layout": layout_name, "note": note, "seed": seed, "onset_tick": onset_idx,
                    "robust_count_at_onset": robust_count,
                })
            live_env.close()
            print(f"{layout_name}[{note}]/seed{seed}: onsets={len(onsets)} done", flush=True)

    total = zero_robust + some_robust
    print(f"\n=== Robust-origin-at-collision-tick summary (n={total} onsets) ===", flush=True)
    print(f"  zero robust options (genuinely cornered): {zero_robust} ({100*zero_robust/total:.1f}%)", flush=True)
    print(f"  at least one robust option existed:        {some_robust} ({100*some_robust/total:.1f}%)", flush=True)

    import json
    (ROOT / "evaluations" / "oracle_robust_origin_at_onset_diagnosis.json").write_text(
        json.dumps(details, indent=2, default=str), encoding="utf-8",
    )
    print("\nSaved.", flush=True)


if __name__ == "__main__":
    main()
