"""2026-08-15: fresh, untouched 840M qualification of v2 (`select_
persistent_waypoint_experimental_invalid_hop_guard`), per explicit user
instruction, code state frozen (no further selector/controller/PPO
changes since the development-validation gate passed -- evaluations/
router_v2_guarded_development_validation.json). 830M/812M/640M/663M/26-
fixtures are development data; this manifest (evaluations/router_mix_
qualification_pool_840000000_manifest.json) has never been inspected by
anything before this script runs.

RESULT AT THE TIME THIS RAN (do not re-derive from memory -- read the
saved file): v2 QUALIFIED. 7->3 collisions, strict subset, zero new,
success 232->236, planner failures exactly equal (0=0), open identical
40/40. Full result: evaluations/router_v2_qualification_840000000_
result.json. v2 was subsequently PROMOTED into production `select_
persistent_waypoint` (2026-08-15) -- see that function's own docstring
in simulator/kinodynamic_route_planner.py for the full evidence chain.

POST-PROMOTION REPRODUCIBILITY NOTE: condition "A" below imports
`select_persistent_waypoint_legacy_pre_v2` from `scratchpad_legacy_
qualified_selector.py` (a frozen, fidelity-checked archival copy of the
PRE-promotion three-tier algorithm), NOT the production `select_
persistent_waypoint` name -- production now contains the promoted v2
design, and `select_persistent_waypoint_experimental_invalid_hop_guard`
is now just an alias to it, so importing either of those two names for
"A" would silently test "new vs new" and produce a meaningless tie if
this script is ever rerun. Do not change this back to importing
`select_persistent_waypoint` for A.

Compares exactly two systems, no trained PPO:
  A = frozen 0051200 + select_persistent_waypoint_legacy_pre_v2 (frozen
      archival copy of the PRE-promotion qualified selector)
  D = frozen 0051200 + select_persistent_waypoint_experimental_invalid_
      hop_guard ("v2" -- now also production, via alias)
Both are called via run_episode_general_router's explicit `selector_fn`
parameter (2026-08-15 addition) -- NOT monkeypatching -- per explicit
user instruction to minimize mutable indirection at qualification time.

All 280 manifest episodes are evaluated: the 240 obstacle episodes AND
the manifest's own 40 open episodes (not the older 777M pool).

Predeclared qualification rule (frozen before evaluation, verbatim):
  OPEN: D outcomes identical to A, preferably both 40/40 success/0
    collisions.
  OBSTACLE: D success count >= A; D collision set (subset-of) A collision
    set; IF A has >=1 collision, D collision set must be a STRICT subset
    of A's.
  PLANNER: D planner-failure set (subset-of) A's (expect exact equality).
  No retries, no regenerated manifest, no episode substitution, no
  threshold changes after outcomes are visible.
  If A scores 240/240 obstacle successes, 840M is NON-DISCRIMINATING for
  the safety-improvement claim -- report that explicitly, do not
  regenerate hunting for failures.
  If A and D simply tie despite A having failures, KEEP A -- v2 has not
  earned promotion from a tie.

Does not touch 820M. No training. Stops after reporting -- the next
step (freezing the selected system against 820M, or declaring 840M
non-discriminating and deciding separately whether another pool is
warranted) is a separate, later decision.
"""
from __future__ import annotations

import json
from pathlib import Path

from stable_baselines3 import PPO

from scratchpad_beginner_navigation_mix_pools import eval_obstacle_manifest, load_manifest
from scratchpad_historical_reproduction_guard import verify_historical_snapshot
from scratchpad_legacy_qualified_selector import select_persistent_waypoint_legacy_pre_v2
from scratchpad_router_v2_guarded_development_validation import eval_open_stratum
from scratchpad_router_v2_qualification_pool import QUALIFICATION_SPEC_SEED
from simulator.kinodynamic_route_planner import select_persistent_waypoint_experimental_invalid_hop_guard

ROOT = Path(__file__).resolve().parents[1]
BASELINE_CHECKPOINT = ROOT / "models" / "generalized_waypoint_both_seed2_0051200.zip"
OBSTACLE_EPISODE_SEED_BASE = 840_600_000
OPEN_EPISODE_SEED_BASE = 840_700_000


def main() -> None:
    verify_historical_snapshot(
        extra_files=(f"evaluations/router_mix_qualification_pool_{QUALIFICATION_SPEC_SEED}_manifest.json",),
    )
    manifest = load_manifest(ROOT / "evaluations" / f"router_mix_qualification_pool_{QUALIFICATION_SPEC_SEED}_manifest.json")
    checksums = json.loads((ROOT / "evaluations" / f"router_mix_qualification_pool_{QUALIFICATION_SPEC_SEED}_checksums.json").read_text(encoding="utf-8"))
    model = PPO.load(str(BASELINE_CHECKPOINT), device="cpu")

    print(f"{'=' * 90}\n840M QUALIFICATION: A (qualified) vs D (v2 guarded), 280 episodes, frozen code state\n{'=' * 90}")
    print(f"Checksums: {checksums}")

    print(f"\n{'=' * 90}\nOBSTACLE (240 episodes)\n{'=' * 90}")
    a_obstacle = eval_obstacle_manifest(model, manifest, episode_seed_base=OBSTACLE_EPISODE_SEED_BASE, selector_fn=select_persistent_waypoint_legacy_pre_v2)
    d_obstacle = eval_obstacle_manifest(model, manifest, episode_seed_base=OBSTACLE_EPISODE_SEED_BASE, selector_fn=select_persistent_waypoint_experimental_invalid_hop_guard)
    a_s, d_s = a_obstacle["combined_summary"], d_obstacle["combined_summary"]
    print(f"  A: n={a_obstacle['n_total']} success={a_s['success_rate']:.4f} collision={a_s['collision_rate']:.4f} "
          f"planner_fail={a_s['planner_failure_rate']:.4f} collisions={a_obstacle['collision_episode_keys']}")
    print(f"  D: n={d_obstacle['n_total']} success={d_s['success_rate']:.4f} collision={d_s['collision_rate']:.4f} "
          f"planner_fail={d_s['planner_failure_rate']:.4f} collisions={d_obstacle['collision_episode_keys']}")

    print(f"\n{'=' * 90}\nOPEN (40 episodes, manifest's own stratum)\n{'=' * 90}")
    a_open = eval_open_stratum(model, manifest, episode_seed_base=OPEN_EPISODE_SEED_BASE)
    d_open = eval_open_stratum(model, manifest, episode_seed_base=OPEN_EPISODE_SEED_BASE)
    print(f"  A: {a_open}")
    print(f"  D: {d_open}")

    # -- predeclared qualification rule, applied mechanically --
    print(f"\n{'=' * 90}\nPREDECLARED QUALIFICATION RULE\n{'=' * 90}")

    open_identical = a_open == d_open
    open_perfect = a_open["success_rate"] == 1.0 and a_open["collision_rate"] == 0.0 and d_open["success_rate"] == 1.0 and d_open["collision_rate"] == 0.0
    print(f"OPEN: A={a_open} D={d_open} -> identical={open_identical} both_perfect={open_perfect}")

    a_coll = set(a_obstacle["collision_episode_keys"])
    d_coll = set(d_obstacle["collision_episode_keys"])
    a_success = round(a_s["success_rate"] * a_obstacle["n_total"])
    d_success = round(d_s["success_rate"] * d_obstacle["n_total"])
    a_pf_set = set(a_obstacle["planner_failure_episode_keys"])
    d_pf_set = set(d_obstacle["planner_failure_episode_keys"])

    non_discriminating = len(a_coll) == 0
    if non_discriminating:
        print(f"\nA has ZERO obstacle collisions (240/240 success) -- 840M is NON-DISCRIMINATING for the "
              f"safety-improvement claim. Per instruction: do NOT regenerate hunting for failures. "
              f"Preserving this result; whether another untouched pool is warranted is a separate decision.")
        d_qualifies = None
    else:
        collision_subset = d_coll.issubset(a_coll)
        collision_strict = d_coll < a_coll
        success_ok = d_success >= a_success
        planner_subset = d_pf_set.issubset(a_pf_set)
        d_qualifies = collision_subset and collision_strict and success_ok and planner_subset
        print(f"OBSTACLE: A_collisions={sorted(a_coll)} D_collisions={sorted(d_coll)} A_success={a_success} D_success={d_success} "
              f"A_planner_fail={sorted(a_pf_set)} D_planner_fail={sorted(d_pf_set)}")
        print(f"  collision_subset={collision_subset} collision_STRICT_subset={collision_strict} success_ge_a={success_ok} "
              f"planner_subset={planner_subset} planner_EXACTLY_equal={a_pf_set == d_pf_set}")
        if d_coll == a_coll:
            print("  D and A collision sets are IDENTICAL (a tie) -- per instruction, KEEP A. v2 has not earned promotion here.")
        print(f"\nD QUALIFIES OVER A: {d_qualifies}")

    print(f"\n{'=' * 90}\nRESULT: {'v2 QUALIFIES on 840M' if d_qualifies else ('840M NON-DISCRIMINATING' if non_discriminating else 'v2 DOES NOT QUALIFY on 840M')}\n{'=' * 90}")
    print("Stopping here per instruction -- 820M not touched, nothing trained. Next step is a separate decision.")

    output = {
        "checksums": checksums,
        "obstacle": {"A": a_obstacle, "D": d_obstacle},
        "open": {"A": a_open, "D": d_open},
        "rule_evaluation": {
            "open_identical": open_identical, "open_both_perfect": open_perfect,
            "non_discriminating": non_discriminating,
            "a_collisions": sorted(a_coll), "d_collisions": sorted(d_coll),
            "a_success": a_success, "d_success": d_success,
            "a_planner_failures": sorted(a_pf_set), "d_planner_failures": sorted(d_pf_set),
            "d_qualifies": d_qualifies,
        },
    }
    out_path = ROOT / "evaluations" / f"router_v2_qualification_{QUALIFICATION_SPEC_SEED}_result.json"
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
