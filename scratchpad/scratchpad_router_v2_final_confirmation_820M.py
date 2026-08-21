"""2026-08-15: ONE-SHOT final confirmation of v2 (`select_persistent_
waypoint_experimental_invalid_hop_guard`) against the already-frozen
820M pool (`evaluations/router_mix_final_pool_820000000_manifest.json`,
121 episodes: 51 single-wall + 50 two-wall + 20 open -- materialized
before Phase B ever began, untouched by any of this investigation's
selector work, deliberately preserved through the entire router-patch
detour). Per explicit user instruction: 820M is NOT regenerated (its
documented 121-episode count, from an original "25/side" planning
assumption vs. GAP_SIDES having 3 values, stands as-is).

Compares exactly two systems, no trained PPO, explicit selector_fn (no
monkeypatching):
  A = frozen 0051200 + select_persistent_waypoint (qualified, production)
  D = frozen 0051200 + select_persistent_waypoint_experimental_invalid_
      hop_guard ("v2", 840M-qualified: 7->3 collisions, strict subset,
      success 232->236, planner failures exactly equal at 0, open
      identical 40/40 -- evaluations/router_v2_qualification_
      840000000_result.json)

Predeclared FINAL rule (frozen before this script sees 820M, verbatim
from explicit user instruction -- stricter than 840M's qualification
rule, closing the "repairs collisions but silently moves failure
elsewhere" hole):

  OPEN: D per-episode outcomes IDENTICAL to A (not just equal counts).
  OBSTACLE:
    - D success >= A
    - D collision set (subset-of) A collision set
    - D total non-success episode set (subset-of) A total non-success
      episode set (collision UNION timeout UNION planner_failure)
    - D timeout set (subset-of) A timeout set
    IF A has >=1 collision: D collision set must be a STRICT subset.
  PLANNER: D planner-failure set (subset-of) A planner-failure set.

  No retries. No substitutions. No regenerated final manifest. No
  threshold changes after outcomes are visible -- this script runs
  ONCE.

  If A has ZERO obstacle collisions: interpreted primarily as a NON-
  REGRESSION confirmation (D must introduce no new failure of any
  kind) -- the strict-improvement claim was already independently
  demonstrated on 840M and is not re-required here.

Does not train anything. Does not modify either selector. Reports the
paired result; promoting v2 into production selector semantics is a
SEPARATE, later decision.

RESULT AT THE TIME THIS RAN (do not re-derive from memory -- read the
saved file): v2 PASSED. 2->1 collisions, strict subset, zero new, zero
timeouts either side, zero planner failures either side, open per-
episode-identical (20/20 both). Full result: evaluations/router_v2_
final_confirmation_820000000_result.json. v2 was subsequently PROMOTED
into production `select_persistent_waypoint` (2026-08-15).

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
"""
from __future__ import annotations

import json
from pathlib import Path

from stable_baselines3 import PPO

from scratchpad_beginner_navigation_mix_pools import FINAL_CONFIRMATION_SPEC_SEED, eval_obstacle_manifest, load_manifest
from scratchpad_historical_reproduction_guard import verify_historical_snapshot
from scratchpad_legacy_qualified_selector import select_persistent_waypoint_legacy_pre_v2
from scratchpad_router_v2_guarded_development_validation import eval_open_stratum
from simulator.kinodynamic_route_planner import select_persistent_waypoint_experimental_invalid_hop_guard

ROOT = Path(__file__).resolve().parents[1]
BASELINE_CHECKPOINT = ROOT / "models" / "generalized_waypoint_both_seed2_0051200.zip"
OBSTACLE_EPISODE_SEED_BASE = 820_600_000
OPEN_EPISODE_SEED_BASE = 820_700_000


def main() -> None:
    verify_historical_snapshot(
        extra_files=(f"evaluations/router_mix_final_pool_{FINAL_CONFIRMATION_SPEC_SEED}_manifest.json",),
    )
    manifest = load_manifest(ROOT / "evaluations" / f"router_mix_final_pool_{FINAL_CONFIRMATION_SPEC_SEED}_manifest.json")
    model = PPO.load(str(BASELINE_CHECKPOINT), device="cpu")

    print(f"{'=' * 90}\n820M FINAL CONFIRMATION: A (qualified) vs D (v2 guarded), 121 episodes, ONE SHOT\n{'=' * 90}")

    print(f"\n{'=' * 90}\nOBSTACLE (101 episodes: 51 single-wall + 50 two-wall)\n{'=' * 90}")
    a_obstacle = eval_obstacle_manifest(model, manifest, episode_seed_base=OBSTACLE_EPISODE_SEED_BASE, selector_fn=select_persistent_waypoint_legacy_pre_v2)
    d_obstacle = eval_obstacle_manifest(model, manifest, episode_seed_base=OBSTACLE_EPISODE_SEED_BASE, selector_fn=select_persistent_waypoint_experimental_invalid_hop_guard)
    a_s, d_s = a_obstacle["combined_summary"], d_obstacle["combined_summary"]
    print(f"  A: n={a_obstacle['n_total']} success={a_s['success_rate']:.4f} collisions={a_obstacle['collision_episode_keys']} "
          f"timeouts={a_obstacle['timeout_episode_keys']} planner_fail={a_obstacle['planner_failure_episode_keys']}")
    print(f"  D: n={d_obstacle['n_total']} success={d_s['success_rate']:.4f} collisions={d_obstacle['collision_episode_keys']} "
          f"timeouts={d_obstacle['timeout_episode_keys']} planner_fail={d_obstacle['planner_failure_episode_keys']}")

    print(f"\n{'=' * 90}\nOPEN (20 episodes, manifest's own stratum)\n{'=' * 90}")
    a_open = eval_open_stratum(model, manifest, episode_seed_base=OPEN_EPISODE_SEED_BASE)
    d_open = eval_open_stratum(model, manifest, episode_seed_base=OPEN_EPISODE_SEED_BASE)
    print(f"  A: {a_open}")
    print(f"  D: {d_open}")

    # -- predeclared FINAL rule --
    print(f"\n{'=' * 90}\nPREDECLARED FINAL RULE\n{'=' * 90}")

    open_identical = a_open["episode_outcomes"] == d_open["episode_outcomes"]
    open_both_perfect = a_open["successes"] == a_open["n"] and d_open["successes"] == d_open["n"]
    print(f"OPEN: per-episode outcomes identical={open_identical} (A={a_open['episode_outcomes']})")
    print(f"      both_perfect(40/40-equivalent)={open_both_perfect}")

    a_coll = set(a_obstacle["collision_episode_keys"])
    d_coll = set(d_obstacle["collision_episode_keys"])
    a_timeout = set(a_obstacle["timeout_episode_keys"])
    d_timeout = set(d_obstacle["timeout_episode_keys"])
    a_pf = set(a_obstacle["planner_failure_episode_keys"])
    d_pf = set(d_obstacle["planner_failure_episode_keys"])
    a_nonsuccess = a_coll | a_timeout | a_pf
    d_nonsuccess = d_coll | d_timeout | d_pf
    a_success = round(a_s["success_rate"] * a_obstacle["n_total"])
    d_success = round(d_s["success_rate"] * d_obstacle["n_total"])

    success_ok = d_success >= a_success
    collision_subset = d_coll.issubset(a_coll)
    nonsuccess_subset = d_nonsuccess.issubset(a_nonsuccess)
    timeout_subset = d_timeout.issubset(a_timeout)
    planner_subset = d_pf.issubset(a_pf)

    zero_collisions_on_a = len(a_coll) == 0
    if zero_collisions_on_a:
        collision_strict = True  # vacuous requirement, not enforced
        print("A has ZERO obstacle collisions on 820M -- interpreting as a NON-REGRESSION confirmation "
              "(strict-improvement already independently demonstrated on 840M, not re-required here).")
    else:
        collision_strict = d_coll < a_coll

    d_passes = success_ok and collision_subset and nonsuccess_subset and timeout_subset and planner_subset and collision_strict and open_identical

    print(f"OBSTACLE: A_collisions={sorted(a_coll)} D_collisions={sorted(d_coll)}")
    print(f"          A_timeouts={sorted(a_timeout)} D_timeouts={sorted(d_timeout)}")
    print(f"          A_planner_fail={sorted(a_pf)} D_planner_fail={sorted(d_pf)}")
    print(f"          A_success={a_success} D_success={d_success}")
    print(f"  success_ge_a={success_ok} collision_subset={collision_subset} "
          f"collision_STRICT(if_A_has_any)={collision_strict} nonsuccess_subset={nonsuccess_subset} "
          f"timeout_subset={timeout_subset} planner_subset={planner_subset}")

    print(f"\n{'=' * 90}\nRESULT: {'v2 PASSES 820M final confirmation' if d_passes else 'v2 DOES NOT PASS 820M final confirmation'}\n{'=' * 90}")
    print("Not promoting anything -- that is a separate, later decision. No further changes made.")

    output = {
        "obstacle": {"A": a_obstacle, "D": d_obstacle},
        "open": {"A": a_open, "D": d_open},
        "rule_evaluation": {
            "open_identical": open_identical, "open_both_perfect": open_both_perfect,
            "zero_collisions_on_a": zero_collisions_on_a,
            "a_collisions": sorted(a_coll), "d_collisions": sorted(d_coll),
            "a_timeouts": sorted(a_timeout), "d_timeouts": sorted(d_timeout),
            "a_planner_failures": sorted(a_pf), "d_planner_failures": sorted(d_pf),
            "a_success": a_success, "d_success": d_success,
            "success_ok": success_ok, "collision_subset": collision_subset, "collision_strict": collision_strict,
            "nonsuccess_subset": nonsuccess_subset, "timeout_subset": timeout_subset, "planner_subset": planner_subset,
            "d_passes": d_passes,
        },
    }
    out_path = ROOT / "evaluations" / f"router_v2_final_confirmation_{FINAL_CONFIRMATION_SPEC_SEED}_result.json"
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
