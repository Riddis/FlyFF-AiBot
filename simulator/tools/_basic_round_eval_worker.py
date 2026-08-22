"""Standalone worker: evaluates one Basic milestone checkpoint (milestone
evaluator + raw diagnostic) out-of-process, so RUN_CANONICAL_BASIC.py's main
loop can move on to the next round's DAgger collection without waiting on
this. Fire-and-forget from the main loop's perspective -- results are
written to the same report paths the main loop used to write inline.

The main loop no longer auto-stops on milestone-eval-derived alarms (only
on the cheap, synchronously-available DAgger-BC head-collapse check) --
this worker prints "!!! ALARM" lines for a human/monitor to catch and
manually abort the run on, per explicit instruction that trades the
automatic gate for wall-clock speed.

Run with: python _basic_round_eval_worker.py <checkpoint_path> <round_idx>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}][round{sys.argv[2]}-eval-worker] {msg}", flush=True)


def main() -> None:
    checkpoint_path = sys.argv[1]
    round_idx = int(sys.argv[2])

    from stable_baselines3 import PPO

    from RUN_CANONICAL_BASIC import (
        DAGGER_EPISODE_SECONDS,
        DAGGER_MAX_ACTIONS,
        EVAL_DIR,
        MILESTONE_EVAL_CURRICULUM,
        MILESTONE_EVAL_EPISODE_SECONDS,
        MILESTONE_EVAL_LAYOUTS,
        MILESTONE_EVAL_MAX_ACTIONS,
        MILESTONE_EVAL_SEEDS,
        RAW_DIAGNOSTIC_HELDOUT_MANIFEST,
        RAW_DIAGNOSTIC_SEEDS,
        RECOVERY_ALARM_DOMINANT_LAYOUT_SHARE,
        RECOVERY_ALARM_INTERVENTION_TICKS_FRACTION,
    )
    from simulator.basic_milestone_evaluator import evaluate_basic_milestone_parallel
    from simulator.beginner_transition import zero_shot_raw_diagnostic_parallel

    log(f"Evaluating checkpoint {checkpoint_path} for round {round_idx}...")

    log("Running Basic milestone evaluator (assisted-mode metrics)...")
    milestone_report = evaluate_basic_milestone_parallel(
        checkpoint_path, MILESTONE_EVAL_CURRICULUM, MILESTONE_EVAL_LAYOUTS, seeds=MILESTONE_EVAL_SEEDS,
        episode_seconds=MILESTONE_EVAL_EPISODE_SECONDS, max_actions=MILESTONE_EVAL_MAX_ACTIONS, n_workers=4,
    )
    report_path = EVAL_DIR / f"canonical_basic_milestone_{round_idx:03d}_report.json"
    report_path.write_text(json.dumps(milestone_report, indent=2, default=str), encoding="utf-8")

    log(f"  intervention_count: {milestone_report['intervention_count']}")
    log(f"  intervention_ticks_fraction: {milestone_report['intervention_ticks_fraction']}")
    log(f"  contacts_per_step: {milestone_report['contacts_per_step']}")
    log(f"  mean_displacement_per_tick: {milestone_report['mean_displacement_per_tick']}")
    log(f"  steering_disagreement_rate: {milestone_report['steering_disagreement_rate']}")
    log(f"  event_disagreement_rate: {milestone_report['event_disagreement_rate']}")
    log(f"  gave_up_episode_fraction: {milestone_report['gave_up_episode_fraction']}")
    log(f"  dominant_layout_intervention_share: {milestone_report['dominant_layout_intervention_share']}")

    log("Running raw (recovery-off) diagnostic (informational only)...")
    diagnostic = zero_shot_raw_diagnostic_parallel(
        checkpoint_path, heldout_manifest_path=RAW_DIAGNOSTIC_HELDOUT_MANIFEST,
        seeds=RAW_DIAGNOSTIC_SEEDS, episode_seconds=DAGGER_EPISODE_SECONDS, max_actions=DAGGER_MAX_ACTIONS, n_workers=4,
    )
    diagnostic_path = EVAL_DIR / f"canonical_basic_milestone_{round_idx:03d}_raw_diagnostic.json"
    diagnostic_path.write_text(json.dumps(diagnostic, indent=2, default=str), encoding="utf-8")
    for layout, stats in diagnostic["per_layout"].items():
        log(f"  raw[{layout}]: stagnation={stats['physical_stagnation_episodes']}/{stats['n_episodes']} "
            f"contacts/100={stats['mean_contacts_per_100_distance']:.2f}")

    alarms = []
    if (milestone_report["intervention_ticks_fraction"]
            and milestone_report["intervention_ticks_fraction"]["median"] >= RECOVERY_ALARM_INTERVENTION_TICKS_FRACTION):
        alarms.append(f"recovery firing on {milestone_report['intervention_ticks_fraction']['median']:.1%} of ticks (near-constant)")
    if milestone_report["dominant_layout_intervention_share"] >= RECOVERY_ALARM_DOMINANT_LAYOUT_SHARE:
        alarms.append(f"one layout accounts for {milestone_report['dominant_layout_intervention_share']:.1%} of interventions")
    if milestone_report["gave_up_episode_fraction"] >= 0.5:
        alarms.append(f"{milestone_report['gave_up_episode_fraction']:.1%} of episodes ended in recovery giving up")

    if alarms:
        log(f"!!! ALARM round {round_idx}: {alarms}")
    else:
        log(f"Round {round_idx} eval clean, no alarms.")


if __name__ == "__main__":
    main()
