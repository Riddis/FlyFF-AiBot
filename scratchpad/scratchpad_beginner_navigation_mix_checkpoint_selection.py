"""2026-08-14: Beginner Navigation Training Mix, Part 4 -- mechanical,
predeclared checkpoint eligibility + selection, applied exactly as
specified in the approved plan, across all 3 replicates' full checkpoint
histories.

Eligibility (all required):
  1. No open-waypoint regression: exactly success_rate==1.0, collision_
     rate==0.0 on held_out_eval_specs(40, seed=777_000_000).
  2. Real, paired improvement on the 120-episode dev pool: baseline vs.
     candidate on the identical episodes, repaired > regressed by >=3,
     AND candidate's collision-episode set is a STRICT SUBSET of
     baseline's.
  3. Timeout/contact classifications are diagnostics only, never gating.

This pool has zero timeouts and zero planner failures for baseline and
every trained checkpoint (verified directly from the saved eval logs) --
so "repaired"/"regressed" reduce exactly to the collision-episode-key SET
DIFFERENCE (repaired = baseline_collisions - candidate_collisions,
regressed = candidate_collisions - baseline_collisions), no separate
paired replay needed.

Ranking among eligible candidates: lowest obstacle-dev-pool collision
rate -> highest success rate -> highest mean_path_efficiency -> lower
mean ticks-to-success -> earlier checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTINUATION_SEEDS = [100, 102, 108]
BASELINE_EVAL_PATH = ROOT / "evaluations" / "router_mix_dev_pool_baseline_eval.json"


def _collision_set(keys: list[str]) -> set[str]:
    return set(keys)


def main() -> None:
    baseline = json.loads(BASELINE_EVAL_PATH.read_text(encoding="utf-8"))
    baseline_collisions = _collision_set(baseline["collision_episode_keys"])
    print(f"Baseline (frozen seed2@0051200) dev-pool collisions ({len(baseline_collisions)}): {sorted(baseline_collisions)}")
    print(f"Baseline: n={baseline['n_total']} success_rate={baseline['combined_summary']['success_rate']:.4f} "
          f"timeout_rate={baseline['combined_summary']['timeout_rate']} planner_failure_rate={baseline['combined_summary']['planner_failure_rate']}")
    assert baseline["combined_summary"]["timeout_rate"] == 0.0
    assert baseline["combined_summary"]["planner_failure_rate"] == 0.0

    all_candidates = []
    for seed in CONTINUATION_SEEDS:
        log_path = ROOT / "evaluations" / f"generalized_waypoint_router_mix_seed{seed}_checkpoint_evals.json"
        entries = json.loads(log_path.read_text(encoding="utf-8"))
        for entry in entries:
            obs = entry["eval"]["obstacle_dev_pool"]
            reg = entry["eval"]["open_regression_777M"]
            summary = obs["combined_summary"]
            assert summary["timeout_rate"] == 0.0 and summary["planner_failure_rate"] == 0.0, (
                f"seed={seed} steps={entry['num_timesteps']}: non-zero timeout/planner_failure rate -- "
                f"the collision-set-difference shortcut for repaired/regressed is invalid here, "
                f"a full paired replay would be required"
            )
            candidate_collisions = _collision_set(obs["collision_episode_keys"])
            repaired = baseline_collisions - candidate_collisions
            regressed = candidate_collisions - baseline_collisions
            gate1_open_regression = reg["success_rate"] == 1.0 and reg["collision_rate"] == 0.0
            gate2_margin = (len(repaired) - len(regressed)) >= 3
            # STRICT subset, per the declared rule -- `.issubset()` also
            # accepts equal sets, which is not what "strict subset" means.
            # 2026-08-14 MISTAKES.md: code previously used .issubset() while
            # separately computing (and never using) a correctly-strict
            # variable -- didn't change eligibility in this dataset (the
            # margin gate already excludes equal-set cases) but the code
            # didn't match the declared rule.
            gate2_subset = candidate_collisions < baseline_collisions
            eligible = gate1_open_regression and gate2_margin and gate2_subset
            all_candidates.append({
                "seed": seed, "steps": entry["num_timesteps"],
                "collision_rate": summary["collision_rate"], "success_rate": summary["success_rate"],
                "mean_path_efficiency": summary["mean_path_efficiency"], "mean_ticks_to_success": summary["mean_ticks_to_success"],
                "collisions": sorted(candidate_collisions), "repaired": sorted(repaired), "regressed": sorted(regressed),
                "repaired_minus_regressed": len(repaired) - len(regressed),
                "gate1_open_regression": gate1_open_regression, "gate2_margin_ge_3": gate2_margin,
                "gate2_strict_subset": gate2_subset, "eligible": eligible,
            })

    print(f"\n{'=' * 100}\nALL CANDIDATES ({len(all_candidates)} checkpoints across {len(CONTINUATION_SEEDS)} replicates)\n{'=' * 100}")
    for c in sorted(all_candidates, key=lambda c: (c["seed"], c["steps"])):
        print(f"  seed={c['seed']:4d} steps={c['steps']:6d} collision_rate={c['collision_rate']:.4f} "
              f"success_rate={c['success_rate']:.4f} repaired={len(c['repaired'])} regressed={len(c['regressed'])} "
              f"(delta={c['repaired_minus_regressed']:+d}) subset={c['gate2_strict_subset']} "
              f"open_ok={c['gate1_open_regression']} ELIGIBLE={c['eligible']}")

    eligible_candidates = [c for c in all_candidates if c["eligible"]]
    ranked = sorted(eligible_candidates, key=lambda c: (
        c["collision_rate"], -c["success_rate"],
        -(c["mean_path_efficiency"] if c["mean_path_efficiency"] is not None else -1.0),
        c["mean_ticks_to_success"] if c["mean_ticks_to_success"] is not None else float("inf"),
        c["steps"],
    ))

    selected = ranked[0] if ranked else None

    print(f"\n{'=' * 100}\nELIGIBILITY RESULT\n{'=' * 100}")
    if not eligible_candidates:
        print("NO ELIGIBLE CANDIDATE. No checkpoint satisfies all three gates (repaired-regressed >= 3 with a")
        print("strict-subset collision set, plus perfect open-waypoint regression).")
        best_by_delta = sorted(all_candidates, key=lambda c: -c["repaired_minus_regressed"])[:5]
        print("\nBest 5 by repaired-minus-regressed (for diagnostic visibility only, NOT eligible):")
        for c in best_by_delta:
            print(f"    seed={c['seed']} steps={c['steps']}: delta={c['repaired_minus_regressed']:+d} "
                  f"collisions={c['collisions']} subset={c['gate2_strict_subset']}")
    else:
        print(f"{len(eligible_candidates)} eligible candidate(s). Selected: seed={selected['seed']} steps={selected['steps']}")

    rule = (
        "Eligibility: (1) open_regression success==1.0,collision==0.0; (2) repaired-regressed>=3 AND "
        "candidate_collision_set subset-of baseline_collision_set; (3) timeout/contact classification "
        "diagnostic-only. Ranking: collision_rate asc, success_rate desc, mean_path_efficiency desc, "
        "mean_ticks_to_success asc, steps asc."
    )
    output = {
        "rule": rule,
        "baseline_collisions": sorted(baseline_collisions),
        "all_candidates": all_candidates,
        "ranked_eligible_candidates": ranked,
        "selected": selected,
    }
    out_path = ROOT / "evaluations" / "router_mix_checkpoint_selection_result.json"
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
