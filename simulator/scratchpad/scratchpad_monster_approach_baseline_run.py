"""2026-08-15: FULL post-router-fix "complete bot" baseline run, per
explicit user instruction. Frozen 0051200 navigator + promoted
production router, steering-only, scripted EVA, multi-kill mode for
multi-target strata. 120 episodes, 6 strata, spec_seed=850_000_000
(evaluations/monster_approach_baseline_pool_850000000_manifest.json --
frozen, geometry-prevalidated only, never resampled by PPO outcome).

2026-08-15 CORRECTED (per user review of the first pass): persists FULL
per-episode raw results (not just aggregated), separately reports
death-driven retargets vs genuine live-target/hysteresis switches
(previously conflated as one "target_switches" number), records the
ACTUAL achieved initial heading/target-bearing per episode (not the
spec's nominal parameter -- env.reset() still randomizes heading for
strata without an explicit override), and empirically verifies the
kill-count-reached invariant (total_kills >= KILL_COUNT_TARGET for every
"killed_target_count_reached" episode) rather than just asserting it in
the runner. Same frozen manifest, same seeds, same movement/routing/EVA/
target-selection -- these are evaluator/reporting corrections only.

DESCRIPTIVE BASELINE, NOT a pass/fail qualification -- no thresholds are
applied or invented after seeing results. RecoveryController is NOT used
in this primary run. No training. No router/controller changes.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from stable_baselines3 import PPO

from .scratchpad_monster_approach_baseline_eval import KILL_COUNT_TARGET, MULTI_TARGET_CATEGORIES, run_monster_approach_episode
from .scratchpad_monster_approach_baseline_pool import FULL_POOL_SPEC_SEED, load_manifest

ROOT = Path(__file__).resolve().parents[2]
BASELINE_CHECKPOINT = ROOT / "models" / "generalized_waypoint_both_seed2_0051200.zip"
EPISODE_SEED_BASE = 850_800_000  # UNCHANGED from the first pass -- identical seeds, deterministic replay


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values), "mean": round(statistics.mean(values), 3), "median": round(statistics.median(values), 3),
        "min": round(min(values), 3), "max": round(max(values), 3),
    }


def _switch_to_kill_success(result) -> tuple[int, int]:
    """For each target_change_event, did a kill occur in that segment
    (strictly after this switch's tick, before the NEXT switch's tick or
    episode end)? Returns (successes, total_switches)."""
    if not result.target_change_events:
        return 0, 0
    switch_ticks = [e["tick"] for e in result.target_change_events] + [result.ticks]
    successes = 0
    for i, event in enumerate(result.target_change_events):
        window_start, window_end = event["tick"], switch_ticks[i + 1]
        if any(window_start < kt <= window_end for kt in result.kill_ticks):
            successes += 1
    return successes, len(result.target_change_events)


def _result_to_dict(category: str, index: int, r) -> dict:
    d = {k: v for k, v in r.__dict__.items() if k != "trace"}
    d["category"] = category
    d["index"] = index
    return d


def main() -> None:
    model = PPO.load(str(BASELINE_CHECKPOINT), device="cpu")
    manifest = load_manifest(ROOT / "simulator" / "evaluations" / f"monster_approach_baseline_pool_{FULL_POOL_SPEC_SEED}_manifest.json")

    print(f"{'=' * 100}\nCORRECTED FULL MONSTER-APPROACH BASELINE (deterministic replay, same manifest/seeds)\n{'=' * 100}")

    per_stratum_raw: dict[str, list] = {}
    all_raw_episodes: list[dict] = []
    seed_counter = 0
    for category, stratum in manifest["strata"].items():
        results = []
        for index, episode_dict in enumerate(stratum["accepted"]):
            seed = EPISODE_SEED_BASE + seed_counter
            seed_counter += 1
            result = run_monster_approach_episode(model, category, episode_dict, episode_seed=seed, verbose=False)
            results.append((index, result))
            all_raw_episodes.append(_result_to_dict(category, index, result))
        per_stratum_raw[category] = results
        outcomes = {}
        for _i, r in results:
            outcomes[r.outcome] = outcomes.get(r.outcome, 0) + 1
        print(f"  {category:20s} ({len(results)} episodes): {outcomes}")

    # -- empirical invariant check: EVERY killed_target_count_reached
    # episode must have total_kills >= KILL_COUNT_TARGET. Computed fresh
    # here from the persisted raw data, not trusted from the runner's own
    # internal assertion alone. --
    print(f"\n{'=' * 100}\nEMPIRICAL INVARIANT CHECK: killed_target_count_reached => total_kills >= {KILL_COUNT_TARGET}\n{'=' * 100}")
    invariant_violations = []
    for category, results in per_stratum_raw.items():
        for index, r in results:
            if r.outcome == "killed_target_count_reached" and r.total_kills < KILL_COUNT_TARGET:
                invariant_violations.append((category, index, r.total_kills))
    if invariant_violations:
        print(f"  VIOLATIONS FOUND: {invariant_violations}")
    else:
        n_checked = sum(1 for cat in MULTI_TARGET_CATEGORIES for _i, r in per_stratum_raw[cat] if r.outcome == "killed_target_count_reached")
        print(f"  PASSED -- all {n_checked} killed_target_count_reached episodes independently verified to have total_kills >= {KILL_COUNT_TARGET}.")

    print(f"\n{'=' * 100}\nPER-STRATUM METRICS\n{'=' * 100}")
    report: dict = {"spec_seed": FULL_POOL_SPEC_SEED, "episode_seed_base": EPISODE_SEED_BASE, "strata": {}}

    for category, results in per_stratum_raw.items():
        n = len(results)
        outcome_keys: dict[str, list[str]] = {}
        for index, r in results:
            outcome_keys.setdefault(r.outcome, []).append(f"({category!r}, {index})")

        total_kills = sum(r.total_kills for _i, r in results)
        episodes_with_kill = sum(1 for _i, r in results if r.total_kills > 0)
        episodes_reaching_range = sum(1 for _i, r in results if r.ticks_to_first_range is not None)
        ticks_to_first_range = [r.ticks_to_first_range for _i, r in results if r.ticks_to_first_range is not None]

        ticks_per_kill: list[float] = []
        for _i, r in results:
            prev = 0
            for kt in r.kill_ticks:
                ticks_per_kill.append(kt - prev)
                prev = kt

        total_contacts = sum(len(r.contact_ticks) for _i, r in results)
        episodes_with_collision = sum(1 for _i, r in results if r.outcome == "collision")
        episodes_timeout = sum(1 for _i, r in results if r.outcome == "timeout")
        episodes_planner_failure = sum(1 for _i, r in results if r.outcome == "planner_failure")
        episodes_stuck = sum(1 for _i, r in results if r.outcome == "stuck")
        total_stuck_ticks = sum(len(r.stuck_trigger_ticks) for _i, r in results)
        episodes_ran_out_of_targets = sum(1 for _i, r in results if r.outcome == "ran_out_of_targets")

        # Switches, split by cause -- the corrected headline metric.
        all_events = [e for _i, r in results for e in r.target_change_events]
        death_driven = [e for e in all_events if e["reason"] == "death_driven_retarget"]
        live_switches = [e for e in all_events if e["reason"] == "live_hysteresis_switch"]
        total_switches = len(all_events)
        total_replans = sum(r.replans for _i, r in results)

        switch_successes = 0
        switch_total = 0
        for _i, r in results:
            s, t = _switch_to_kill_success(r)
            switch_successes += s
            switch_total += t

        all_drift = [d for _i, r in results for d in r.target_drift_segments_cells]
        headings = [r.initial_heading_radians for _i, r in results if r.initial_heading_radians is not None]
        bearings_deg = [round(__import__("math").degrees(r.initial_target_bearing_relative_radians), 2)
                         for _i, r in results if r.initial_target_bearing_relative_radians is not None]

        stratum_report = {
            "n_episodes": n,
            "outcome_counts": {k: len(v) for k, v in outcome_keys.items()},
            "outcome_episode_keys": outcome_keys,
            "total_kills": total_kills,
            "kill_rate_episodes_with_ge1_kill": round(episodes_with_kill / n, 3),
            "mean_kills_per_episode": round(total_kills / n, 3),
            "episodes_reaching_eva_range": episodes_reaching_range,
            "episodes_reaching_eva_range_rate": round(episodes_reaching_range / n, 3),
            "ticks_to_first_range": _stats(ticks_to_first_range),
            "ticks_per_kill": _stats(ticks_per_kill),
            "total_contact_ticks": total_contacts,
            "episodes_with_collision": episodes_with_collision,
            "episodes_timeout": episodes_timeout,
            "episodes_planner_failure": episodes_planner_failure,
            "episodes_stuck": episodes_stuck,
            "total_stuck_trigger_ticks": total_stuck_ticks,
            "episodes_ran_out_of_targets": episodes_ran_out_of_targets,
            "total_target_switches": total_switches,
            "death_driven_retargets": len(death_driven),
            "live_hysteresis_switches": len(live_switches),
            "mean_target_switches_per_episode": round(total_switches / n, 3),
            "total_replans": total_replans,
            "switch_to_kill_success": {"successes": switch_successes, "total_switches": switch_total,
                                        "rate": round(switch_successes / switch_total, 3) if switch_total else None},
            "target_drift_cells": _stats(all_drift),
            "initial_target_bearing_deg": _stats(bearings_deg),
        }
        report["strata"][category] = stratum_report

        print(f"\n--- {category} ---")
        print(f"  outcomes: {stratum_report['outcome_counts']}")
        print(f"  kills: total={total_kills} rate(>=1 kill)={stratum_report['kill_rate_episodes_with_ge1_kill']} mean/ep={stratum_report['mean_kills_per_episode']}")
        print(f"  reached EVA range: {episodes_reaching_range}/{n} ({stratum_report['episodes_reaching_eva_range_rate']})")
        print(f"  ticks_to_first_range: {stratum_report['ticks_to_first_range']}")
        print(f"  ticks_per_kill: {stratum_report['ticks_per_kill']}")
        print(f"  collisions={episodes_with_collision} timeouts={episodes_timeout} planner_fail={episodes_planner_failure} "
              f"stuck={episodes_stuck} ran_out_of_targets={episodes_ran_out_of_targets}")
        print(f"  switches: total={total_switches} (death_driven_retarget={len(death_driven)}, live_hysteresis_switch={len(live_switches)}) replans={total_replans}")
        print(f"  switch_to_kill_success={stratum_report['switch_to_kill_success']}")
        print(f"  target_drift_cells: {stratum_report['target_drift_cells']}")
        print(f"  initial_target_bearing_deg (ACTUAL, not nominal spec): {stratum_report['initial_target_bearing_deg']}")

    # -- overall summary --
    print(f"\n{'=' * 100}\nOVERALL SUMMARY\n{'=' * 100}")
    all_n = sum(s["n_episodes"] for s in report["strata"].values())
    all_kills = sum(s["total_kills"] for s in report["strata"].values())
    all_collisions = sum(s["episodes_with_collision"] for s in report["strata"].values())
    all_timeouts = sum(s["episodes_timeout"] for s in report["strata"].values())
    all_planner_fail = sum(s["episodes_planner_failure"] for s in report["strata"].values())
    all_stuck = sum(s["episodes_stuck"] for s in report["strata"].values())
    all_ran_out = sum(s["episodes_ran_out_of_targets"] for s in report["strata"].values())
    all_switches = sum(s["total_target_switches"] for s in report["strata"].values())
    all_death_driven = sum(s["death_driven_retargets"] for s in report["strata"].values())
    all_live_switches = sum(s["live_hysteresis_switches"] for s in report["strata"].values())
    all_replans = sum(s["total_replans"] for s in report["strata"].values())
    print(f"Total episodes: {all_n}  Total kills: {all_kills}  Collisions: {all_collisions}  "
          f"Timeouts: {all_timeouts}  Planner failures: {all_planner_fail}  Stuck: {all_stuck}  Ran out of targets: {all_ran_out}")
    print(f"Total switches: {all_switches} (death_driven_retarget={all_death_driven}, live_hysteresis_switch={all_live_switches})  Total replans: {all_replans}")
    assert all_switches == all_replans, f"switches ({all_switches}) != replans ({all_replans}) -- every switch to a non-None target must trigger exactly one replan"
    print(f"  (verified: total switches == total replans, {all_switches} == {all_replans})")

    report["overall"] = {
        "n_episodes": all_n, "total_kills": all_kills, "episodes_with_collision": all_collisions,
        "episodes_timeout": all_timeouts, "episodes_planner_failure": all_planner_fail, "episodes_stuck": all_stuck,
        "episodes_ran_out_of_targets": all_ran_out, "total_switches": all_switches,
        "death_driven_retargets": all_death_driven, "live_hysteresis_switches": all_live_switches,
        "total_replans": all_replans, "invariant_violations": invariant_violations,
    }
    report["raw_episodes"] = all_raw_episodes

    out_path = ROOT / "simulator" / "evaluations" / f"monster_approach_baseline_{FULL_POOL_SPEC_SEED}_result_corrected.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved corrected full baseline report (with raw per-episode data) to {out_path}")


if __name__ == "__main__":
    main()
