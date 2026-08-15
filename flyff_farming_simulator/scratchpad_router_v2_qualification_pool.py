"""2026-08-15: fresh 840M qualification pool for select_persistent_
waypoint_experimental_invalid_hop_guard ("v2"), per explicit user
instruction, after v2 passed development validation on 830M/812M/640M/
663M/26-fixtures (collisions AND planner failures, both checked --
evaluations/router_v2_guarded_development_validation.json). 830M/812M/
640M/663M/26-fixtures are now development data; this pool has never been
inspected by anything.

QUALIFICATION_SPEC_SEED = 840_000_000
  single-wall: 40/side x 3 sides = 120
  two-wall:    60/direction x 2 directions = 120
  open:        40
  TOTAL = 280, obstacle slice = 240

Same structure as 830M (scratchpad_router_patch_qualification_pool.py),
same build_manifest/save_manifest (unchanged, fixed deterministic stream
IDs, same geometry-only prevalidation), different seed. Manifest is
materialized in full BEFORE any policy/selector evaluation touches it --
this script only builds and freezes it plus records checksums tying the
qualification result to an exact code state; scratchpad_router_v2_
qualification_840M.py does the actual A-vs-D evaluation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scratchpad_beginner_navigation_mix_pools import build_manifest, save_manifest

ROOT = Path(__file__).resolve().parent

QUALIFICATION_SPEC_SEED = 840_000_000
N_PER_SIDE = 40
N_PER_DIRECTION = 60
N_OPEN = 40

BASELINE_CHECKPOINT = ROOT / "models" / "generalized_waypoint_both_seed2_0051200.zip"
ROUTER_SOURCE = ROOT / "simulator" / "kinodynamic_route_planner.py"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    manifest_path = ROOT / "evaluations" / f"router_mix_qualification_pool_{QUALIFICATION_SPEC_SEED}_manifest.json"
    print(f"{'=' * 90}\nBuilding v2 qualification manifest (spec_seed={QUALIFICATION_SPEC_SEED})\n{'=' * 90}")
    manifest = build_manifest(QUALIFICATION_SPEC_SEED, n_per_side=N_PER_SIDE, n_per_direction=N_PER_DIRECTION, n_open=N_OPEN)
    save_manifest(manifest, manifest_path)
    total = sum(len(s["accepted"]) for s in manifest["strata"].values())
    obstacle_total = sum(len(s["accepted"]) for name, s in manifest["strata"].items() if name != "open")
    print(f"\nTotal episodes: {total} (obstacle slice: {obstacle_total}). No policy/selector has been run against this pool yet.")

    checksums = {
        "manifest_spec_seed": QUALIFICATION_SPEC_SEED,
        "manifest_sha256": _sha256_file(manifest_path),
        "baseline_checkpoint_path": str(BASELINE_CHECKPOINT.relative_to(ROOT)),
        "baseline_checkpoint_sha256": _sha256_file(BASELINE_CHECKPOINT),
        "router_source_path": str(ROUTER_SOURCE.relative_to(ROOT)),
        "router_source_sha256": _sha256_file(ROUTER_SOURCE),
    }
    checksums_path = ROOT / "evaluations" / f"router_mix_qualification_pool_{QUALIFICATION_SPEC_SEED}_checksums.json"
    checksums_path.write_text(json.dumps(checksums, indent=2), encoding="utf-8")
    print(f"\nChecksums recorded to {checksums_path}:")
    for k, v in checksums.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
