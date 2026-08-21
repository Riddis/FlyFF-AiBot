"""Build a FRESH, untouched confirmation manifest for the terminal-gated v3
steering oracle (escape-BFS robust-safety fix + continuation_depth=4).

2026-08-09 correction from the user: the early_heldout / early_heldout_
unseen_templates / early_challenge pools that motivated the terminal gate,
the escape-BFS fix, and the continuation_depth selection are now
DEVELOPMENT pools, not final qualification pools -- an oracle that scores
well there is a CANDIDATE teacher, not a qualified one. Before any oracle
label touches Basic, evaluate on layouts/seeds that have not influenced any
of these design decisions.

All existing early-stage curricula (training, heldout, unseen_templates,
factorial_probe) use per-variant seeds in the ~20,268,xxx-21,324,xxx range
(confirmed by direct inspection, not assumed). This uses seed base
23,000,000 -- clearly outside that entire range, so every layout here is
disjoint from every seed this oracle's design decisions have ever seen.

Covers all 6 templates used throughout this session's tuning work (open_field,
irregular_plain, broad_lobes, split_field, wide_neck, open_center) at 2
density/respawn combinations each (typical_fast, high_bursty), obstacle_level=0
(matching this session's "early" stage convention), stage="early".
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simulator.synthetic import generate_curriculum_from_plan
from simulator import curriculum_manifests

OUT_DIR = ROOT / "simulator" / "curricula" / "synthetic_curriculum_oracle_fresh_confirmation"
FRESH_SEED_BASE = 23_000_000

TEMPLATES = ("open_field", "irregular_plain", "broad_lobes", "split_field", "wide_neck", "open_center")


def main() -> None:
    plan = []
    for template in TEMPLATES:
        plan.append(("early", template, "typical", "fast", 0))
        plan.append(("early", template, "high", "bursty", 0))

    curriculum_path = generate_curriculum_from_plan(
        OUT_DIR, plan, seed=FRESH_SEED_BASE, overwrite=True,
        curriculum_name="Oracle fresh confirmation pool (2026-08-09, untouched by tuning)",
    )
    print(f"Generated: {curriculum_path}")

    layouts = tuple(v["name"] for v in json.loads(curriculum_path.read_text())["variants"])
    manifest = curriculum_manifests.HeldoutManifest(
        stage="early",
        curriculum_path="simulator/curricula/synthetic_curriculum_oracle_fresh_confirmation/curriculum.json",
        layouts=layouts,
        notes=(
            "FRESH confirmation pool for the terminal-gated v3 steering oracle "
            "(escape-BFS robust-safety fix + continuation_depth=4), built "
            "2026-08-09 specifically because early_heldout/early_heldout_"
            "unseen_templates/early_challenge became development pools during "
            "this oracle's tuning (they directly informed the terminal gate, "
            "the escape-BFS fix, and the continuation_depth selection). Covers "
            "all 6 templates used during tuning at 2 density/respawn combos each "
            f"(typical_fast, high_bursty), obstacle_level=0, stage=early. Seed "
            f"base {FRESH_SEED_BASE} -- disjoint from every seed range used by "
            "training, early_heldout, early_heldout_unseen_templates, and "
            "early_challenge (all confirmed to fall within ~20,268,xxx-"
            "21,324,xxx). Standard for this pool is stricter than Basic's own "
            "later recovery-assisted graduation tolerance: essentially zero "
            "physical collision events, since this evaluates a collision-"
            "avoidance TEACHER, not a training-wheel policy."
        ),
    )
    out_path = curriculum_manifests.save_manifest(
        manifest, ROOT / "simulator" / "evaluations" / "manifests" / "oracle_fresh_confirmation.json"
    )
    print(f"Saved manifest: {out_path}")
    print(f"Layouts ({len(layouts)}): {layouts}")

    curriculum_manifests.assert_disjoint_from_training(
        manifest, "simulator/curricula/synthetic_curriculum/curriculum.json", manifest_root="."
    )
    print("Confirmed disjoint from the early-stage training curriculum.")


if __name__ == "__main__":
    main()
