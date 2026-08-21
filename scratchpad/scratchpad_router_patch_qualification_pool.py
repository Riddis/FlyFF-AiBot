"""2026-08-14: fresh, larger qualification pool for the router-patch
comparison (A = frozen 0051200 + OLD selector, B = frozen 0051200 +
PATCHED selector, C = seed100@56320 + PATCHED selector), per explicit
user instruction. 812M is saturated (both baseline+patch and trained+
patch already reach 120/120 there) and can no longer discriminate between
candidates -- this pool roughly doubles the obstacle evidence.

QUALIFICATION_SPEC_SEED = 830_000_000
  single-wall: 40/side x 3 sides = 120
  two-wall:    60/direction x 2 directions = 120
  open:        40
  TOTAL = 280, obstacle slice = 240

Reuses build_manifest/save_manifest UNCHANGED from
scratchpad_beginner_navigation_mix_pools.py -- same prevalidation
(final_native in-bounds + plan_route succeeds), same same-mode
resampling on rejection, same fixed deterministic stream IDs (the
hash()-randomization bug already fixed there). Manifest is materialized
in full BEFORE any policy/router evaluation touches it -- this script
only builds and freezes it; scratchpad_router_patch_qualification_compare.py
does the actual A/B/C evaluation.
"""
from __future__ import annotations

from pathlib import Path

from scratchpad_beginner_navigation_mix_pools import build_manifest, save_manifest

ROOT = Path(__file__).resolve().parents[1]

QUALIFICATION_SPEC_SEED = 830_000_000
N_PER_SIDE = 40
N_PER_DIRECTION = 60
N_OPEN = 40


def main() -> None:
    path = ROOT / "evaluations" / f"router_mix_qualification_pool_{QUALIFICATION_SPEC_SEED}_manifest.json"
    print(f"{'=' * 90}\nBuilding router-patch qualification manifest (spec_seed={QUALIFICATION_SPEC_SEED})\n{'=' * 90}")
    manifest = build_manifest(QUALIFICATION_SPEC_SEED, n_per_side=N_PER_SIDE, n_per_direction=N_PER_DIRECTION, n_open=N_OPEN)
    save_manifest(manifest, path)
    total = sum(len(s["accepted"]) for s in manifest["strata"].values())
    obstacle_total = sum(len(s["accepted"]) for name, s in manifest["strata"].items() if name != "open")
    print(f"\nTotal episodes: {total} (obstacle slice: {obstacle_total}). No policy/router has been run against this pool yet.")


if __name__ == "__main__":
    main()
