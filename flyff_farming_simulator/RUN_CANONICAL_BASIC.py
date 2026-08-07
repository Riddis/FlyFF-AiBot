"""The first real fresh Basic training run for the canonical lineage.

Fresh current-architecture initialization (no historical weights) -> human
BC bootstrap (all 8 canonical recordings, causal 923->925 temporal
sidecars) -> repeated rounds of {recovery-assisted DAgger collection on the
broad-template sibling pool, supervised update, milestone checkpoint,
assisted-mode milestone evaluation, informational raw diagnostic} -> stop
and report. No PPO anywhere in this script -- see simulator/
basic_environment.py's module docstring for why. Every saved checkpoint
gets a run_provenance.json sidecar.

Run with: python RUN_CANONICAL_BASIC.py
"""

from __future__ import annotations

import glob
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
EVAL_DIR = ROOT / "evaluations"
MODELS_DIR.mkdir(exist_ok=True)
EVAL_DIR.mkdir(exist_ok=True)

SEED = 0
TRAINING_CURRICULUM = str(ROOT / "synthetic_curriculum" / "curriculum.json")
DAGGER_CURRICULUM = str(ROOT / "synthetic_curriculum_phase2_dagger_siblings_v2" / "curriculum.json")
DAGGER_LAYOUTS = [
    "01_early_open_field_typical_fast", "02_early_open_field_low_fast", "03_early_open_field_high_typical",
    "04_early_wide_neck_high_typical", "05_early_wide_neck_typical_bursty",
    "06_early_irregular_plain_typical_fast", "07_early_split_field_typical_fast",
]
MILESTONE_EVAL_CURRICULUM = str(ROOT / "synthetic_curriculum_navigation_calibration" / "curriculum.json")
MILESTONE_EVAL_LAYOUTS = [
    "01_early_open_field_typical_fast", "03_early_irregular_plain_typical_fast",
    "05_early_broad_lobes_typical_fast", "07_early_wide_neck_typical_fast",
    "09_early_split_field_typical_fast", "11_early_open_center_typical_fast",
]  # one per template, for breadth without every round costing 12 layouts
MILESTONE_EVAL_SEEDS = [0, 1, 2]
RAW_DIAGNOSTIC_HELDOUT_MANIFEST = str(ROOT / "evaluations" / "manifests" / "early_heldout.json")
RAW_DIAGNOSTIC_SEEDS = [0, 1]

N_ROUNDS = 6
DAGGER_EPISODE_SECONDS = 90.0
DAGGER_MAX_ACTIONS = 400
MILESTONE_EVAL_EPISODE_SECONDS = 90.0
MILESTONE_EVAL_MAX_ACTIONS = 400

RECOVERY_ALARM_INTERVENTION_TICKS_FRACTION = 0.85  # "firing almost constantly"
RECOVERY_ALARM_DOMINANT_LAYOUT_SHARE = 0.85  # "one layout dominating failures"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def check_no_nan(model, where: str) -> None:
    import torch
    for name, param in model.policy.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            raise RuntimeError(f"NaN/Inf detected in {name} at {where} -- stopping, not continuing through this.")


def main() -> None:
    from simulator.basic_environment import collect_basic_dagger_dataset, save_basic_dagger_dataset
    from simulator.basic_training import (
        bootstrap_policy_from_human_recordings,
        bootstrap_steering_from_teacher,
        build_fresh_basic_policy,
        build_human_bootstrap_dataset,
        canonical_checkpoint_name,
        collect_simulator_teacher_dataset,
        save_checkpoint_with_provenance,
    )
    from simulator.demonstrations import export_demonstrations
    from simulator.map_model import MapModel
    from simulator.navigation_dataset import MiningConfig
    from simulator.navigation_history import NavigationHistoryWrapper
    from simulator.synthetic import iter_variant_environments

    training_recordings = sorted(glob.glob(str(ROOT / "recordings" / "training" / "*.zip")))
    eva_only_recordings = sorted(glob.glob(str(ROOT / "recordings" / "eva_only" / "*.zip")))
    log(f"Human recordings: {len(training_recordings)} direct_keyboard, {len(eva_only_recordings)} eva_only")

    # ---------------------------------------------------------------
    log("=== Stage 1: export human demonstrations ===")
    # ---------------------------------------------------------------
    demo_path = EVAL_DIR / "canonical_basic_human_demos.npz"
    if demo_path.exists():
        log(f"Reusing existing export: {demo_path}")
    else:
        map_data = MapModel.load()
        export_demonstrations(
            training_recordings, demo_path, map_model=map_data, eva_only_recording_paths=eva_only_recordings,
        )
    demo_data = np.load(demo_path, allow_pickle=False)
    log(f"Human demonstration samples: {demo_data['observations'].shape[0]}, sessions: {len(set(demo_data['session_index'].tolist()))}")

    bootstrap_dataset_path = EVAL_DIR / "canonical_basic_bootstrap_dataset.npz"
    build_human_bootstrap_dataset(demo_path, bootstrap_dataset_path)
    log(f"Bootstrap dataset (925-dim, neutral recent_contact): {bootstrap_dataset_path}")

    eval_processes: list[subprocess.Popen] = []
    eval_worker_script = ROOT / "_basic_round_eval_worker.py"

    def dispatch_round_eval(checkpoint_path: Path, round_idx: int) -> None:
        log(f"Dispatching round {round_idx} evaluation asynchronously (milestone evaluator + raw diagnostic) "
            f"in parallel with the next round's DAgger collection -- watch for '!!! ALARM round {round_idx}' "
            "in this worker's own log output; the main loop no longer blocks on or auto-stops from these checks.")
        proc = subprocess.Popen(
            [sys.executable, str(eval_worker_script), str(checkpoint_path), str(round_idx)],
            cwd=str(ROOT),
        )
        eval_processes.append(proc)

    # ---------------------------------------------------------------
    # Resume support: if earlier milestone checkpoints already exist (e.g.
    # this script was restarted mid-run to pick up the async-eval change),
    # skip straight to the next un-collected round instead of redoing the
    # fresh init + bootstrap (which is deterministic under SEED=0 and would
    # just reproduce identical results at real wall-clock cost).
    # ---------------------------------------------------------------
    existing_milestones: dict[int, Path] = {}
    for p in MODELS_DIR.glob("canonical_basic_milestone_*.zip"):
        m = re.match(r"canonical_basic_milestone_(\d+)\.zip", p.name)
        if m:
            existing_milestones[int(m.group(1))] = p
    completed_round = max(existing_milestones) if existing_milestones else 0

    if completed_round > 0:
        # ---------------------------------------------------------------
        log(f"=== Resuming: found completed round {completed_round} checkpoint, skipping fresh init/bootstrap ===")
        # ---------------------------------------------------------------
        previous_checkpoint = existing_milestones[completed_round]
        from stable_baselines3 import PPO
        model = PPO.load(str(previous_checkpoint), device="cpu")
        log(f"Loaded {previous_checkpoint} as the resume starting point.")
        resume_report_path = EVAL_DIR / f"canonical_basic_milestone_{completed_round:03d}_report.json"
        if not resume_report_path.exists():
            log(f"Round {completed_round} has no eval report yet (was mid-flight when interrupted) -- "
                "dispatching its evaluation now before resuming collection.")
            dispatch_round_eval(previous_checkpoint, completed_round)
        first_round = completed_round + 1
    else:
        # ---------------------------------------------------------------
        log("=== Stage 2: fresh current-architecture initialization ===")
        # ---------------------------------------------------------------
        entry, probe_env = next(iter(iter_variant_environments(
            TRAINING_CURRICULUM, stage="early", seed=SEED, episode_steps=10, episode_seconds=5.0,
        )))
        wrapped_probe_env = NavigationHistoryWrapper(probe_env)
        model = build_fresh_basic_policy(wrapped_probe_env, seed=SEED, device="cpu")
        wrapped_probe_env.close()
        log(f"Fresh policy built. observation_space={model.observation_space}. starting_checkpoint=NONE (verified fresh).")

        # ---------------------------------------------------------------
        log("=== Stage 3a: steering bootstrap from the scripted simulator teacher ===")
        # ---------------------------------------------------------------
        # Human recordings do NOT bootstrap steering -- see basic_training.py's
        # module docstring. Known project precedent: when steering was
        # restricted to the compact target-geometry representation, human
        # steering-label accuracy fell to ~31.6% while scripted-teacher
        # steering/target-angle correlations/simulated farming stayed healthy.
        # Re-confirmed directly on the real canonical human dataset before this
        # run: all 6 geometry features correlate <=0.05 with recorded human
        # steering across 2684 valid samples, loss stuck flat across 60 epochs
        # regardless of class-balanced weighting -- a real, already-known
        # property of the representation, not a bug to chase further here.
        teacher_dataset_path = EVAL_DIR / "canonical_basic_teacher_dataset.npz"
        teacher_dataset = collect_simulator_teacher_dataset(
            DAGGER_CURRICULUM, DAGGER_LAYOUTS, samples=6000, episode_seconds=DAGGER_EPISODE_SECONDS,
            max_actions=DAGGER_MAX_ACTIONS, seed=SEED, output_path=teacher_dataset_path,
        )
        log(f"Teacher dataset: {teacher_dataset['observations'].shape[0]} samples, "
            f"steering_counts={teacher_dataset['steering_counts']}, event_counts={teacher_dataset['event_counts']}")

        steering_result = bootstrap_steering_from_teacher(
            model, teacher_dataset, epochs=20, learning_rate=3e-4, batch_size=128, seed=SEED, progress_every_seconds=20.0,
        )
        check_no_nan(model, "after teacher steering bootstrap")
        log(f"Teacher steering BC done. train={steering_result['train_samples']} val={steering_result['validation_samples']}")
        log(f"  final epoch loss: {steering_result['history'][-1]}")
        log(f"  steering accuracy before/after: "
            f"{steering_result['before']['gate']['heads']['steering']['accuracy']:.3f} -> "
            f"{steering_result['after']['gate']['heads']['steering']['accuracy']:.3f}")
        log(f"  angle correlation (expect positive vs LEFT, negative vs RIGHT): {steering_result['angle_correlation']}")
        corr = steering_result["angle_correlation"]
        if not (corr["corr_sin_angle_vs_is_left"] and corr["corr_sin_angle_vs_is_left"] > 0
                and corr["corr_sin_angle_vs_is_right"] and corr["corr_sin_angle_vs_is_right"] < 0):
            raise RuntimeError(
                f"Teacher steering bootstrap did not produce sign-correct target-angle correlations: {corr} "
                "-- stopping. This is the same scripted-teacher path that worked historically; a failure here "
                "is a real regression, not the known human-steering mismatch, and needs its own diagnosis."
            )

        # ---------------------------------------------------------------
        log("=== Stage 3b: event/EVA bootstrap from human recordings ===")
        # ---------------------------------------------------------------
        event_result = bootstrap_policy_from_human_recordings(
            model, bootstrap_dataset_path, train_heads=("event",), epochs=20, learning_rate=3e-4, batch_size=128,
            validation_fraction=0.15, seed=SEED, progress_every_seconds=20.0,
        )
        check_no_nan(model, "after human event bootstrap")
        log(f"Human event BC done. train={event_result['train_samples']} val={event_result['validation_samples']}")
        log(f"  final epoch losses: {event_result['history'][-1]}")
        log(f"  event accuracy before/after: "
            f"{event_result['before']['gate']['heads']['event']['accuracy']:.3f} -> "
            f"{event_result['after']['gate']['heads']['event']['accuracy']:.3f}")
        log(f"  steering accuracy unchanged (frozen) before/after: "
            f"{event_result['before']['gate']['heads']['steering']['accuracy']:.3f} -> "
            f"{event_result['after']['gate']['heads']['steering']['accuracy']:.3f}")
        event_conf = event_result["after"]["gate"]["heads"]["event"].get("per_class")
        if event_conf:
            log(f"  event per-class (val): {event_conf}")

        if event_result["after"]["gate"]["heads"]["event"]["accuracy"] < 0.3:
            raise RuntimeError(
                f"Event head accuracy is only {event_result['after']['gate']['heads']['event']['accuracy']:.3f} "
                "after human bootstrap -- stopping rather than proceeding with a possibly-broken event head."
            )

        bootstrap_checkpoint = MODELS_DIR / f"{canonical_checkpoint_name('basic', 'bootstrap')}.zip"
        save_checkpoint_with_provenance(
            model, bootstrap_checkpoint, stage="basic", milestone="bootstrap", seeds=SEED,
            config={
                "steering_source": "scripted_teacher", "steering_epochs": 20, "steering_samples": teacher_dataset["observations"].shape[0],
                "event_source": "human_recordings", "event_epochs": 20,
            },
            curriculum_path=DAGGER_CURRICULUM, recording_paths=training_recordings + eva_only_recordings,
            starting_checkpoint=None,
            extra={
                "steering_result_summary": {
                    "train_samples": steering_result["train_samples"], "angle_correlation": steering_result["angle_correlation"],
                },
                "event_result_summary": {"train_samples": event_result["train_samples"]},
            },
        )
        log(f"Saved: {bootstrap_checkpoint} (+ provenance)")
        previous_checkpoint = bootstrap_checkpoint
        first_round = 1

    all_round_reports = []
    summary_path = EVAL_DIR / "canonical_basic_run_summary.json"
    if summary_path.exists():
        try:
            all_round_reports = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            all_round_reports = []

    # ---------------------------------------------------------------
    log(f"=== Stage 4: rounds {first_round}-{N_ROUNDS} Basic milestone rounds (recovery-assisted DAgger, no PPO) ===")
    # ---------------------------------------------------------------
    for round_idx in range(first_round, N_ROUNDS + 1):
        log(f"--- Round {round_idx}/{N_ROUNDS} ---")
        round_seeds = [round_idx * 100 + i for i in range(3)]

        log(f"Collecting recovery-assisted DAgger data (layouts={len(DAGGER_LAYOUTS)}, seeds={round_seeds})...")
        mining_config = MiningConfig(max_events_per_layout_seed=15, max_events_per_episode=6, max_samples_per_event=1)
        mined = collect_basic_dagger_dataset(
            DAGGER_CURRICULUM, DAGGER_LAYOUTS, seeds=round_seeds, model=model,
            episode_seconds=DAGGER_EPISODE_SECONDS, max_actions=DAGGER_MAX_ACTIONS, config=mining_config,
            progress_every_seconds=20.0,
        )
        dagger_path = EVAL_DIR / f"canonical_basic_dagger_round{round_idx:03d}.npz"
        save_basic_dagger_dataset(mined, str(dagger_path))
        log(f"Mined {mined['observations'].shape[0]} samples. category_counts={mined['category_counts']}")

        intervention_counts = [s["intervention_count"] for s in mined["episode_summaries"]]
        intervention_ticks_fractions = [
            s["intervention_ticks"] / max(1, s["steps"]) for s in mined["episode_summaries"]
        ]
        gave_up = [s["final_state"] == "given_up" for s in mined["episode_summaries"]]
        log(f"  DAgger-collection interventions: total={sum(intervention_counts)} "
            f"mean_ticks_fraction={float(np.mean(intervention_ticks_fractions)):.3f} "
            f"gave_up_episodes={sum(gave_up)}/{len(gave_up)}")

        log("Supervised update on newly mined DAgger data (steering+event: DAgger labels are simulator-teacher-labeled, well-correlated with the geometry representation, unlike human recordings)...")
        dagger_bc_result = bootstrap_policy_from_human_recordings(
            model, dagger_path, train_heads=("steering", "event"), epochs=4, learning_rate=1e-4, batch_size=64,
            validation_fraction=0.2, seed=SEED + round_idx, progress_every_seconds=20.0,
        )
        check_no_nan(model, f"after round {round_idx} DAgger update")

        log("Rehearsing human event data only (guard against forgetting EVA/event basics; steering intentionally excluded, see module docstring)...")
        rehearsal_result = bootstrap_policy_from_human_recordings(
            model, bootstrap_dataset_path, train_heads=("event",), epochs=2, learning_rate=5e-5, batch_size=128,
            validation_fraction=0.15, seed=SEED + round_idx, progress_every_seconds=20.0,
        )
        check_no_nan(model, f"after round {round_idx} human rehearsal")

        milestone_checkpoint = MODELS_DIR / f"{canonical_checkpoint_name('basic', f'milestone_{round_idx:03d}')}.zip"
        save_checkpoint_with_provenance(
            model, milestone_checkpoint, stage="basic", milestone=f"milestone_{round_idx:03d}", seeds=round_seeds,
            config={"round": round_idx, "dagger_epochs": 4, "rehearsal_epochs": 2, "mining_config": mining_config.__dict__},
            curriculum_path=DAGGER_CURRICULUM, dagger_config={"layouts": DAGGER_LAYOUTS, "seeds": round_seeds},
            recovery_config={"enabled": True, "role": "training_wheel_dagger_collection_only"},
            starting_checkpoint=str(Path(previous_checkpoint).resolve()),
        )
        log(f"Saved: {milestone_checkpoint} (+ provenance)")
        previous_checkpoint = milestone_checkpoint

        # --- cheap, synchronous stop condition: DAgger-BC head collapse is
        # known immediately (no extra compute) and is worth stopping for
        # before even dispatching an evaluation of a possibly-broken model.
        alarms = []
        if dagger_bc_result["after"]["gate"]["heads"]["event"]["accuracy"] < 0.3:
            alarms.append(f"event-head accuracy collapsed to {dagger_bc_result['after']['gate']['heads']['event']['accuracy']:.3f}")
        if dagger_bc_result["after"]["gate"]["heads"]["steering"]["accuracy"] < 0.3:
            alarms.append(f"steering-head accuracy collapsed to {dagger_bc_result['after']['gate']['heads']['steering']['accuracy']:.3f} on DAgger validation")

        all_round_reports.append({
            "round": round_idx, "checkpoint": str(milestone_checkpoint),
            "dagger_bc_result_summary": {
                "steering_accuracy_after": dagger_bc_result["after"]["gate"]["heads"]["steering"]["accuracy"],
                "event_accuracy_after": dagger_bc_result["after"]["gate"]["heads"]["event"]["accuracy"],
            },
            "alarms": alarms,
            "eval_report_path": str(EVAL_DIR / f"canonical_basic_milestone_{round_idx:03d}_report.json"),
            "eval_diagnostic_path": str(EVAL_DIR / f"canonical_basic_milestone_{round_idx:03d}_raw_diagnostic.json"),
        })
        summary_path.write_text(json.dumps(all_round_reports, indent=2, default=str), encoding="utf-8")

        if alarms:
            log(f"!!! STOP CONDITIONS TRIGGERED at round {round_idx}: {alarms}")
            log("Stopping the run for diagnosis rather than continuing through a clear break.")
            break

        # The full milestone evaluator + raw diagnostic (the slow part, ~10+
        # minutes) runs out-of-process against this saved checkpoint while
        # the main loop moves straight on to the next round's DAgger
        # collection. This trades the previous automatic full-eval-based
        # stop conditions (near-constant recovery firing, one layout
        # dominating, high give-up fraction) for wall-clock speed -- those
        # are now printed as "!!! ALARM" lines by the worker for manual
        # review/abort rather than blocking round progression.
        dispatch_round_eval(milestone_checkpoint, round_idx)
        log(f"Round {round_idx} core training clean, evaluation dispatched asynchronously. Continuing to next round.")

    # ---------------------------------------------------------------
    log("=== Stage 5: waiting on outstanding async evaluations, final summary ===")
    # ---------------------------------------------------------------
    for proc in eval_processes:
        proc.wait()
    log(f"All {len(eval_processes)} dispatched evaluation worker(s) finished.")
    summary_path.write_text(json.dumps(all_round_reports, indent=2, default=str), encoding="utf-8")
    log(f"Full run summary written to {summary_path}")
    log(f"Final checkpoint: {previous_checkpoint}")
    log("=== RUN COMPLETE ===")


if __name__ == "__main__":
    main()
