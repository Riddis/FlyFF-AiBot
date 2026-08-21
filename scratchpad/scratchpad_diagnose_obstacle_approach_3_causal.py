"""2026-08-15: narrowly-scoped paired causal diagnostic on the exact
frozen obstacle_approach[3] spec/seed, per explicit user instruction.
DIAGNOSTIC ONLY -- does not become a qualification gate, does not cause
any policy/router/pool change.

Question: obstacle_approach[3] collided starting from an extreme -177.36
deg heading (the target was at +176.35 deg -- essentially the worst-case
opposite direction), with a wall present nearby but NOT blocking the
direct player->target segment (confirmed by the geometry audit). Is this
"extreme-heading x nearby-obstacle interaction" (the wall's presence
mattered) or "pure heading-recovery failure" (the wall was incidental,
same collision would happen in the open)?

Three conditions, same checkpoint/router/EVA/target logic as the real
baseline run, differing ONLY in the controlled variable:
  A. ORIGINAL: wall present, heading = the actual -177.36 deg the real
     episode used.
  B. WALL REMOVED, SAME HEADING: identical spec minus the wall, heading
     still forced to -177.36 deg. If the collision disappears, the wall
     was causally necessary -- an interaction, not a pure heading bug.
  C. WALL PRESENT, ALIGNED HEADING: identical spec (wall included),
     heading forced to point directly at the target (~0 deg relative
     bearing, minimal turn required). If this succeeds cleanly, the
     extreme heading was what set up the risky trajectory in the first
     place.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from stable_baselines3 import PPO

from scratchpad_monster_approach_baseline_pool import FULL_POOL_SPEC_SEED, load_manifest, spec_from_episode_dict
from scratchpad_monster_approach_baseline_eval import run_monster_approach_episode

ROOT = Path(__file__).resolve().parents[1]
BASELINE_CHECKPOINT = ROOT / "models" / "generalized_waypoint_both_seed2_0051200.zip"
ORIGINAL_SEED = 850_800_043  # exact seed obstacle_approach[3] used in the corrected baseline run (850_800_000 + seed_counter=43)


def _spec_dict_from(spec, *, wall_specs=None, heading_override_radians=None) -> dict:
    return {
        "category": spec.category,
        "monster_offsets": [list(o) for o in spec.monster_offsets],
        "wall_specs": list(spec.wall_specs) if wall_specs is None else list(wall_specs),
        "heading_override_radians": spec.heading_override_radians if heading_override_radians is None else heading_override_radians,
    }


def main() -> None:
    model = PPO.load(str(BASELINE_CHECKPOINT), device="cpu")
    manifest = load_manifest(ROOT / "evaluations" / f"monster_approach_baseline_pool_{FULL_POOL_SPEC_SEED}_manifest.json")
    episode_dict = manifest["strata"]["obstacle_approach"]["accepted"][3]
    spec = spec_from_episode_dict("obstacle_approach", episode_dict)

    # Exact real value the episode actually used (not a rounded
    # approximation) -- pulled directly from the corrected baseline's
    # persisted raw_episodes.
    corrected_result = json.loads((ROOT / "evaluations" / f"monster_approach_baseline_{FULL_POOL_SPEC_SEED}_result_corrected.json").read_text(encoding="utf-8"))
    ep3_raw = next(e for e in corrected_result["raw_episodes"] if e["category"] == "obstacle_approach" and e["index"] == 3)
    ACTUAL_HEADING = float(ep3_raw["initial_heading_radians"])
    ALIGNED_HEADING = spec.monster_offsets[0][1]  # the spec's own bearing offset -- forcing heading here gives EXACTLY 0 relative bearing (player is at the polar-offset origin)

    conditions = {
        "A_original_wall_present_extreme_heading": _spec_dict_from(spec, heading_override_radians=ACTUAL_HEADING),
        "B_wall_removed_same_extreme_heading": _spec_dict_from(spec, wall_specs=(), heading_override_radians=ACTUAL_HEADING),
        "C_wall_present_aligned_heading": _spec_dict_from(spec, heading_override_radians=ALIGNED_HEADING),
    }

    print(f"{'=' * 100}\nCAUSAL DIAGNOSTIC: obstacle_approach[3] -- wall x heading interaction\n{'=' * 100}")
    print(f"Original spec: {json.dumps(episode_dict, indent=2)}")
    print(f"ACTUAL_HEADING (rad/deg): {ACTUAL_HEADING} / {__import__('math').degrees(ACTUAL_HEADING):.2f}")
    print(f"ALIGNED_HEADING (rad/deg): {ALIGNED_HEADING} / {__import__('math').degrees(ALIGNED_HEADING):.2f}")

    results = {}
    for name, ep_dict in conditions.items():
        result = run_monster_approach_episode(model, "obstacle_approach", ep_dict, episode_seed=ORIGINAL_SEED, verbose=True)
        results[name] = result
        print(f"\n--- {name} ---")
        print(f"  outcome={result.outcome} ticks={result.ticks} contact_ticks={result.contact_ticks}")
        for row in result.trace:
            print(f"    {row}")

    print(f"\n{'=' * 100}\nSUMMARY\n{'=' * 100}")
    for name, r in results.items():
        print(f"  {name}: outcome={r.outcome}")

    a_outcome = results["A_original_wall_present_extreme_heading"].outcome
    b_outcome = results["B_wall_removed_same_extreme_heading"].outcome
    c_outcome = results["C_wall_present_aligned_heading"].outcome

    print(f"\n{'=' * 100}")
    if a_outcome == "collision" and b_outcome != "collision":
        print("VERDICT: wall removal ELIMINATES the collision (same extreme heading) -- the wall was CAUSALLY NECESSARY.")
        print("         This supports 'extreme-heading x nearby-obstacle interaction', not a pure heading-recovery failure.")
    elif a_outcome == "collision" and b_outcome == "collision":
        print("VERDICT: collision PERSISTS with the wall removed -- the wall was NOT necessary for this failure.")
        print("         This points to a pure heading-recovery issue, with the wall incidental in the original episode.")
    else:
        print(f"VERDICT: unexpected -- condition A did not reproduce a collision (outcome={a_outcome}). Investigate before concluding.")

    if c_outcome == "collision":
        print("Additionally: the SAME wall with an ALIGNED (minimal-turn) heading ALSO collided -- the wall alone is more hazardous than the direct-segment-clear check suggested.")
    else:
        print(f"Additionally: the SAME wall with an ALIGNED (minimal-turn) heading did NOT collide (outcome={c_outcome}) -- consistent with the wall being safe under a normal approach heading.")
    print(f"{'=' * 100}")

    out_path = ROOT / "evaluations" / "diagnose_obstacle_approach_3_causal.json"
    out_path.write_text(json.dumps({
        name: {k: v for k, v in r.__dict__.items() if k != "trace"} | {"trace": r.trace}
        for name, r in results.items()
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
