from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from farming.actions import FarmingEvent, POLICY_ACTION_NVECS, SteeringAction
from farming.model_contract import ModelContractMetadata

from .factorized_cli import (
    ACTION_CONTRACT_ID,
    INVALID_EVA_CONTRACT_ID,
    TIMING_CONTRACT_ID,
    _balanced_training_vec_env,
    _evaluate_env,
    run_smoke,
)
from .factorized_training import atomic_save_policy, validate_factorized_policy_contract
from .factorized_v193_training import (
    collect_human_demonstration_dataset_v193,
    collect_teacher_dataset_v193,
    rehearse_factorized_policy_v193,
    train_hybrid_factorized_teacher_v193,
)
from .recording_discovery import (
    discover_direct_demonstration_eligible,
    discover_eva_only_supplementary,
)
from .reward_model import REWARD_CONTRACT_ID, SimulatorRewardConfig
from .scripted_policies import scripted_command
from .split_branch_policy import SplitSteeringEventPolicy
from .synthetic import iter_variant_environments

PILOT_CONTRACT_ID = "layout-balanced-calibrated-teacher-rehearsal-v2"


def _mean_probability(layouts: list[dict[str, Any]], branch: str, head: str, value: int) -> float:
    return float(
        np.mean([item[branch][f"{head}_probabilities"][str(value)] for item in layouts])
    )


def _teacher_relative_gate(layouts: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    random_kph = float(np.mean([item["random"]["kills_per_simulated_hour"] for item in layouts]))
    teacher_kph = float(np.mean([item["teacher"]["kills_per_simulated_hour"] for item in layouts]))
    policy_kph = float(np.mean([item["policy"]["kills_per_simulated_hour"] for item in layouts]))
    policy_random_ratio = policy_kph / max(1.0e-9, random_kph)
    policy_teacher_ratio = policy_kph / max(1.0e-9, teacher_kph)
    reasons: list[str] = []
    if policy_kph <= 0.0:
        reasons.append("policy produced no kills")
    if policy_random_ratio < 1.0:
        reasons.append(
            f"policy/random kill-rate ratio {policy_random_ratio:.3f} is below 1.000"
        )
    if policy_teacher_ratio < 0.25:
        reasons.append(
            f"policy/teacher kill-rate ratio {policy_teacher_ratio:.3f} is below 0.250"
        )

    for item in layouts:
        variant = str(item["variant"])
        teacher = item["teacher"]
        policy = item["policy"]
        layout_reasons: list[str] = []
        policy_kills = float(policy["kills_per_simulated_hour"])
        teacher_kills = float(teacher["kills_per_simulated_hour"])
        if policy_kills <= 0.0:
            layout_reasons.append("no kills")
        if float(policy["mean_valid_eva_casts"]) <= 0.0:
            layout_reasons.append("no valid EVA")
        if teacher_kills > 0.0 and policy_kills / teacher_kills < 0.20:
            layout_reasons.append(
                f"kill rate is only {policy_kills / teacher_kills:.3f} of teacher"
            )

        policy_steering = np.asarray(
            [policy["steering_probabilities"][str(i)] for i in range(3)], dtype=float
        )
        teacher_steering = np.asarray(
            [teacher["steering_probabilities"][str(i)] for i in range(3)], dtype=float
        )
        dominant = int(np.argmax(policy_steering))
        if policy_steering[dominant] > 0.98 and teacher_steering[dominant] < 0.90:
            layout_reasons.append(
                f"steering {SteeringAction(dominant).name} is {policy_steering[dominant]:.3f} "
                f"but teacher uses it {teacher_steering[dominant]:.3f}"
            )

        policy_jump = float(policy["event_probabilities"][str(int(FarmingEvent.JUMP))])
        teacher_jump = float(teacher["event_probabilities"][str(int(FarmingEvent.JUMP))])
        if policy_jump > max(0.20, teacher_jump + 0.15):
            layout_reasons.append(
                f"jump fraction {policy_jump:.3f} exceeds teacher {teacher_jump:.3f} by too much"
            )

        policy_invalid = float(policy["mean_invalid_eva_attempts"])
        teacher_invalid = float(teacher["mean_invalid_eva_attempts"])
        policy_valid = float(policy["mean_valid_eva_casts"])
        if policy_invalid > max(25.0, teacher_invalid + 20.0, policy_valid * 8.0):
            layout_reasons.append(
                f"invalid EVA attempts {policy_invalid:.1f} are excessive versus teacher {teacher_invalid:.1f}"
            )

        policy_contacts = float(policy["mean_contacts"])
        teacher_contacts = float(teacher["mean_contacts"])
        if policy_contacts > max(125.0, teacher_contacts * 2.0 + 25.0):
            layout_reasons.append(
                f"contacts {policy_contacts:.1f} are excessive versus teacher {teacher_contacts:.1f}"
            )

        item["gate"] = {"passed": not layout_reasons, "reasons": layout_reasons}
        if layout_reasons:
            reasons.append(f"{variant}: " + "; ".join(layout_reasons))

    aggregate = {
        "random_kills_per_simulated_hour": random_kph,
        "teacher_kills_per_simulated_hour": teacher_kph,
        "policy_kills_per_simulated_hour": policy_kph,
        "policy_to_random_kill_rate_ratio": policy_random_ratio,
        "policy_to_teacher_kill_rate_ratio": policy_teacher_ratio,
        "teacher_steering_probabilities": {
            str(i): _mean_probability(layouts, "teacher", "steering", i) for i in range(3)
        },
        "policy_steering_probabilities": {
            str(i): _mean_probability(layouts, "policy", "steering", i) for i in range(3)
        },
        "teacher_event_probabilities": {
            str(i): _mean_probability(layouts, "teacher", "event", i) for i in range(3)
        },
        "policy_event_probabilities": {
            str(i): _mean_probability(layouts, "policy", "event", i) for i in range(3)
        },
    }
    return reasons, aggregate


def evaluate_checkpoint_v193(
    curriculum: Path,
    checkpoint: Path,
    *,
    stage: str,
    episodes: int,
    episode_seconds: float,
    max_actions: int,
    seed: int,
    device: str,
    output: Path | None,
    require_gate: bool,
) -> tuple[dict[str, Any], bool]:
    try:
        from stable_baselines3 import PPO
    except ImportError as error:
        raise SystemExit("Install requirements/training.txt before evaluation") from error

    policy = PPO.load(str(checkpoint), device=device)
    validate_factorized_policy_contract(policy)
    layouts: list[dict[str, Any]] = []
    for index, (entry, env) in enumerate(
        iter_variant_environments(
            curriculum,
            stage=stage,
            seed=seed,
            episode_steps=max_actions,
            episode_seconds=episode_seconds,
        )
    ):
        rng = np.random.default_rng(seed + index * 10007)

        def random_selector(_obs, _env):
            return np.asarray(
                [rng.integers(0, POLICY_ACTION_NVECS[0]), rng.integers(0, POLICY_ACTION_NVECS[1])],
                dtype=np.int64,
            )

        def teacher_selector(_obs, raw_env):
            return scripted_command("obstacle_aware", raw_env).as_array()

        def policy_selector(obs, _env):
            action, _state = policy.predict(obs, deterministic=True)
            return action

        matched_seed = seed + index * 100
        print(f"Evaluating calibrated factorized layout {index + 1}: {entry.name}", flush=True)
        random_report = _evaluate_env(
            env,
            random_selector,
            episodes=episodes,
            max_actions=max_actions,
            seed=matched_seed,
            label=f"{entry.name}/random",
        )
        teacher_report = _evaluate_env(
            env,
            teacher_selector,
            episodes=episodes,
            max_actions=max_actions,
            seed=matched_seed,
            label=f"{entry.name}/teacher",
        )
        policy_report = _evaluate_env(
            env,
            policy_selector,
            episodes=episodes,
            max_actions=max_actions,
            seed=matched_seed,
            label=f"{entry.name}/policy",
        )
        layouts.append(
            {
                "variant": entry.name,
                "random": random_report,
                "teacher": teacher_report,
                "policy": policy_report,
            }
        )
        env.close()

    reasons, aggregate = _teacher_relative_gate(layouts)
    summary = {
        "checkpoint": str(checkpoint.resolve()),
        "action_contract": ACTION_CONTRACT_ID,
        "action_nvec": list(POLICY_ACTION_NVECS),
        "observation_size": 923,
        "stage": stage,
        "episode_seconds": float(episode_seconds),
        "episodes_per_layout": int(episodes),
        "layouts": layouts,
        "aggregate": aggregate,
        "stage_gate": {"passed": not reasons, "reasons": reasons},
    }
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved: {output}", flush=True)
    return summary, not reasons


def run_pilot(args: argparse.Namespace) -> int:
    try:
        import torch
        from stable_baselines3 import PPO
    except ImportError as error:
        raise SystemExit("Install requirements/training.txt before training") from error

    env, training_layouts = _balanced_training_vec_env(
        args.curriculum,
        stage="early",
        seed=args.seed,
        max_actions=args.max_actions,
        episode_seconds=args.episode_seconds,
    )
    print("Balanced PPO environments: " + ", ".join(training_layouts), flush=True)
    # SplitSteeringEventPolicy architecturally isolates the steering head so
    # it can only see the layout-invariant derived geometry features
    # (simulator.geometry_features), not the full raw observation. The
    # rollout diagnostic proved best_group_relative_angle() is fully
    # recoverable from the 923-value observation, yet a single shared
    # MultiDiscrete([3, 3]) head trained end-to-end learned a per-layout
    # shortcut instead -- steering probabilities barely responded to the
    # true target angle and collapsed to one direction for nearly whole
    # episodes. Removing the raw observation from the steering pathway
    # removes the shortcut's raw material; the event head and value function
    # keep the full observation since nothing implicated them.
    policy = PPO(
        SplitSteeringEventPolicy,
        env,
        verbose=1,
        n_steps=256,
        batch_size=128,
        n_epochs=4,
        learning_rate=5e-5,
        clip_range=0.10,
        target_kl=0.015,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.015,
        seed=args.seed,
        device=args.device,
        tensorboard_log=str(args.tensorboard),
        policy_kwargs={
            "steering_net_arch": [32, 16],
            "event_net_arch": [256, 128],
            "vf_net_arch": [256, 128],
        },
    )

    dataset_report = collect_teacher_dataset_v193(
        args.curriculum,
        stage="early",
        samples=args.teacher_samples,
        episode_seconds=args.teacher_episode_seconds,
        max_actions=args.teacher_max_actions,
        teacher_policy="obstacle_aware",
        seed=args.seed,
        output=args.teacher_dataset,
    )
    args.evaluations.mkdir(parents=True, exist_ok=True)
    dataset_report_path = args.evaluations / "factorized_v193_teacher_dataset.json"
    dataset_report_path.write_text(json.dumps(dataset_report, indent=2) + "\n", encoding="utf-8")

    # Step B: export every currently-classified human recording. The scripted
    # teacher above never used human recordings to derive its behavior and
    # continues to supply layout diversity, group-selection, and obstacle
    # avoidance; human demonstrations regularize the policy toward realistic
    # steering/event combinations on top of it. Falls back to scripted-only
    # automatically (inside train_hybrid_factorized_teacher_v193) when there
    # aren't enough human sessions, so an empty recordings/training/ is safe.
    human_dataset_report: dict[str, Any] | None = None
    human_dataset_path: Path | None = None
    if not args.skip_human_demonstrations:
        demo_eligible = discover_direct_demonstration_eligible([args.human_recordings_dir])
        eva_only_supplementary = discover_eva_only_supplementary(
            [args.eva_only_recordings_dir], exclude=demo_eligible
        )
        human_dataset_report = collect_human_demonstration_dataset_v193(
            demo_eligible,
            eva_only_supplementary,
            output=args.human_dataset,
        )
        if human_dataset_report is None:
            print(
                "No eligible human recordings found under "
                f"{args.human_recordings_dir} / {args.eva_only_recordings_dir}; "
                "continuing scripted-teacher-only.",
                flush=True,
            )
        else:
            human_dataset_path = Path(human_dataset_report["path"])
            human_dataset_report_path = args.evaluations / "factorized_v193_human_dataset.json"
            human_dataset_report_path.write_text(
                json.dumps(human_dataset_report, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"Exported {human_dataset_report['samples']} human demonstration samples from "
                f"{human_dataset_report['sessions']} session(s): {human_dataset_report_path}",
                flush=True,
            )

    teacher_report = train_hybrid_factorized_teacher_v193(
        policy,
        args.teacher_dataset,
        human_dataset_path,
        recognition_epochs=args.recognition_epochs,
        recognition_learning_rate=args.recognition_learning_rate,
        calibration_epochs=args.calibration_epochs,
        calibration_learning_rate=args.calibration_learning_rate,
        batch_size=args.teacher_batch_size,
        human_fraction=args.human_fraction,
        minimum_human_sessions=args.minimum_human_sessions,
        seed=args.seed,
    )
    teacher_clone_report_path = args.evaluations / "factorized_v193_teacher_clone_gate.json"
    teacher_clone_report_path.write_text(
        json.dumps(teacher_report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved calibrated teacher clone report: {teacher_clone_report_path}", flush=True)
    if not bool(teacher_report.get("passed", False)):
        env.close()
        print("Calibrated teacher clone failed; PPO was not started.", flush=True)
        return 3

    metadata = {
        "action_contract": ACTION_CONTRACT_ID,
        "action_nvec": list(POLICY_ACTION_NVECS),
        "forward_latched": True,
        "timing_contract": TIMING_CONTRACT_ID,
        "invalid_eva_contract": INVALID_EVA_CONTRACT_ID,
        "reward_contract": REWARD_CONTRACT_ID,
        "reward_config": SimulatorRewardConfig().as_dict(),
        "pilot_contract": PILOT_CONTRACT_ID,
        "training_layouts": training_layouts,
        "training_episode_seconds": float(args.episode_seconds),
        "teacher_dataset": str(args.teacher_dataset.resolve()),
        "teacher_dataset_report": dataset_report,
        "human_dataset_report": human_dataset_report,
        "teacher_clone": teacher_report,
        "policy_architecture": {
            "class": "SplitSteeringEventPolicy",
            "steering_net_arch": [32, 16],
            "event_net_arch": [256, 128],
            "vf_net_arch": [256, 128],
            "activation": "ReLU",
            "steering_input": "geometry_features (6-value derived, layout-invariant)",
            "event_input": "full 923-value observation",
        },
    }
    setattr(policy, "farming_contract_metadata", ModelContractMetadata.current().as_dict())
    setattr(policy, "synthetic_curriculum_metadata", metadata)
    teacher_path = args.output.with_name(args.output.stem + "_teacher.zip")
    atomic_save_policy(policy, teacher_path)
    print(f"Saved calibrated teacher checkpoint: {teacher_path}", flush=True)

    _, teacher_passed = evaluate_checkpoint_v193(
        args.curriculum,
        teacher_path,
        stage="early",
        episodes=args.gate_episodes,
        episode_seconds=args.gate_episode_seconds,
        max_actions=args.gate_max_actions,
        seed=args.seed,
        device=args.device,
        output=args.evaluations / "factorized_v193_teacher_gate.json",
        require_gate=True,
    )
    if not teacher_passed:
        env.close()
        print("Calibrated teacher rollout gate failed; PPO was not started.", flush=True)
        return 3

    completed = 0
    while completed < args.timesteps:
        chunk = min(args.chunk_size, args.timesteps - completed)
        policy.learn(
            total_timesteps=chunk,
            reset_num_timesteps=False,
            progress_bar=True,
            tb_log_name="factorized_v193_pilot",
        )
        completed += chunk
        rehearsal = rehearse_factorized_policy_v193(
            policy,
            args.teacher_dataset,
            recognition_epochs=args.rehearsal_recognition_epochs,
            calibration_epochs=args.rehearsal_calibration_epochs,
            learning_rate=args.rehearsal_learning_rate,
            batch_size=args.teacher_batch_size,
            seed=args.seed + completed,
        )
        rehearsal_path = args.evaluations / f"factorized_v193_{completed}_rehearsal.json"
        rehearsal_path.write_text(json.dumps(rehearsal, indent=2) + "\n", encoding="utf-8")
        print(f"Saved mixed rehearsal report: {rehearsal_path}", flush=True)
        if not bool(rehearsal.get("passed", False)):
            env.close()
            print(f"Pilot stopped at {completed} steps because mixed rehearsal failed.", flush=True)
            return 3

        setattr(policy, "farming_contract_metadata", ModelContractMetadata.current().as_dict())
        setattr(
            policy,
            "synthetic_curriculum_metadata",
            {**metadata, "pilot_timesteps": completed, "last_rehearsal": rehearsal},
        )
        checkpoint = args.output.with_name(args.output.stem + f"_{completed}.zip")
        atomic_save_policy(policy, checkpoint)
        _, passed = evaluate_checkpoint_v193(
            args.curriculum,
            checkpoint,
            stage="early",
            episodes=args.gate_episodes,
            episode_seconds=args.gate_episode_seconds,
            max_actions=args.gate_max_actions,
            seed=args.seed,
            device=args.device,
            output=args.evaluations / f"factorized_v193_{completed}_gate.json",
            require_gate=True,
        )
        if not passed:
            env.close()
            print(f"Pilot stopped at {completed} steps because the teacher-relative gate failed.", flush=True)
            return 3

    atomic_save_policy(policy, args.output)
    env.close()
    print(f"Pilot complete: {args.output}", flush=True)
    return 0


def resume_ppo_chunk(args: argparse.Namespace) -> int:
    """Continue PPO on an already-gated checkpoint for exactly one bounded
    chunk, then rehearse and gate before returning -- unlike run_pilot's
    loop, this never regenerates or retrains the starting checkpoint (it is
    loaded as-is) and never continues to a further chunk on its own; call it
    again explicitly for another chunk.
    """
    try:
        from stable_baselines3 import PPO
    except ImportError as error:
        raise SystemExit("Install requirements/training.txt before training") from error

    env, training_layouts = _balanced_training_vec_env(
        args.curriculum,
        stage="early",
        seed=args.seed,
        max_actions=args.max_actions,
        episode_seconds=args.episode_seconds,
    )
    print("Balanced PPO environments: " + ", ".join(training_layouts), flush=True)
    print(f"Resuming from checkpoint: {args.checkpoint} (unchanged until this run)", flush=True)
    policy = PPO.load(str(args.checkpoint), env=env, device=args.device, tensorboard_log=str(args.tensorboard))
    validate_factorized_policy_contract(policy)

    policy.learn(
        total_timesteps=args.timesteps,
        reset_num_timesteps=False,
        progress_bar=True,
        tb_log_name="factorized_v193_resume",
    )

    rehearsal = rehearse_factorized_policy_v193(
        policy,
        args.teacher_dataset,
        recognition_epochs=args.rehearsal_recognition_epochs,
        calibration_epochs=args.rehearsal_calibration_epochs,
        learning_rate=args.rehearsal_learning_rate,
        batch_size=args.teacher_batch_size,
        seed=args.seed,
    )
    # Labeled by cumulative progress (args.label), not just this call's chunk
    # size -- resuming the same chunk size again (e.g. 5k -> 10k) would
    # otherwise silently overwrite the previous chunk's reports under an
    # identical filename.
    label = args.label or str(args.timesteps)
    args.evaluations.mkdir(parents=True, exist_ok=True)
    rehearsal_path = args.evaluations / f"factorized_v193_resume_{label}_rehearsal.json"
    rehearsal_path.write_text(json.dumps(rehearsal, indent=2) + "\n", encoding="utf-8")
    print(f"Saved rehearsal report: {rehearsal_path}", flush=True)
    if not bool(rehearsal.get("passed", False)):
        env.close()
        print(f"Resume chunk stopped: mixed rehearsal failed after {args.timesteps} steps.", flush=True)
        return 3

    setattr(policy, "farming_contract_metadata", ModelContractMetadata.current().as_dict())
    atomic_save_policy(policy, args.output)
    env.close()
    print(f"Saved resumed checkpoint: {args.output}", flush=True)

    _, passed = evaluate_checkpoint_v193(
        args.curriculum,
        args.output,
        stage="early",
        episodes=args.gate_episodes,
        episode_seconds=args.gate_episode_seconds,
        max_actions=args.gate_max_actions,
        seed=args.seed,
        device=args.device,
        output=args.evaluations / f"factorized_v193_resume_{label}_gate.json",
        require_gate=True,
    )
    if not passed:
        print("Resume chunk stopped: teacher-relative rollout/collapse gate failed.", flush=True)
        return 3
    print(f"Resume chunk passed all gates after {args.timesteps} steps: {args.output}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FlyFF calibrated factorized pilot v1.9.3")
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("curriculum", type=Path)

    pilot = sub.add_parser("pilot")
    pilot.add_argument("curriculum", type=Path)
    pilot.add_argument("--output", type=Path, default=Path("models/generic_farming_v193_pilot.zip"))
    pilot.add_argument("--evaluations", type=Path, default=Path("simulator/evaluations"))
    pilot.add_argument("--tensorboard", type=Path, default=Path("training_logs/factorized_v193"))
    pilot.add_argument("--teacher-dataset", type=Path, default=Path("simulator/datasets/factorized_v193_teacher.npz"))
    pilot.add_argument("--timesteps", type=int, default=25_000)
    pilot.add_argument("--chunk-size", type=int, default=5_000)
    pilot.add_argument("--teacher-samples", type=int, default=12_000)
    pilot.add_argument("--teacher-episode-seconds", type=float, default=60.0)
    pilot.add_argument("--teacher-max-actions", type=int, default=400)
    pilot.add_argument("--teacher-batch-size", type=int, default=256)
    pilot.add_argument("--recognition-epochs", type=int, default=12)
    pilot.add_argument("--recognition-learning-rate", type=float, default=3.0e-4)
    pilot.add_argument("--calibration-epochs", type=int, default=8)
    pilot.add_argument("--calibration-learning-rate", type=float, default=3.0e-5)
    pilot.add_argument("--episode-seconds", type=float, default=120.0)
    pilot.add_argument("--max-actions", type=int, default=800)
    pilot.add_argument(
        "--gate-episodes",
        type=int,
        default=3,
        help="Seeds per layout for the rollout gate; kills_per_simulated_hour becomes the per-episode median, not a pooled rate, once this is >1.",
    )
    pilot.add_argument("--gate-episode-seconds", type=float, default=120.0)
    pilot.add_argument("--gate-max-actions", type=int, default=800)
    pilot.add_argument("--rehearsal-recognition-epochs", type=int, default=1)
    pilot.add_argument("--rehearsal-calibration-epochs", type=int, default=2)
    pilot.add_argument("--rehearsal-learning-rate", type=float, default=2.0e-5)
    pilot.add_argument(
        "--human-recordings-dir",
        type=Path,
        default=Path("recordings/training"),
        help="Every *.zip here that is direct-demonstration-eligible is used for human BC.",
    )
    pilot.add_argument(
        "--eva-only-recordings-dir",
        type=Path,
        default=Path("recordings/eva_only"),
        help="Every *.zip here with real EVA presses supervises the event head only.",
    )
    pilot.add_argument(
        "--human-dataset", type=Path, default=Path("simulator/datasets/factorized_v193_human_demonstrations.npz")
    )
    pilot.add_argument(
        "--human-fraction",
        type=float,
        default=0.35,
        help="Target round share for human data during hybrid recognition/calibration.",
    )
    pilot.add_argument(
        "--minimum-human-sessions",
        type=int,
        default=2,
        help="Fewer steering-capable human sessions than this falls back to scripted-only.",
    )
    pilot.add_argument(
        "--skip-human-demonstrations",
        action="store_true",
        help="Train scripted-teacher-only even if eligible human recordings exist.",
    )
    pilot.add_argument("--seed", type=int, default=0)
    pilot.add_argument("--device", default="auto")

    resume = sub.add_parser(
        "resume-ppo-chunk",
        help=(
            "Load an existing, already-gated checkpoint unchanged and run exactly "
            "one bounded PPO chunk, then rehearse and gate. Never regenerates or "
            "retrains the starting checkpoint and never chains to a further chunk "
            "automatically -- call again explicitly for another one."
        ),
    )
    resume.add_argument("curriculum", type=Path)
    resume.add_argument("checkpoint", type=Path)
    resume.add_argument("--output", type=Path, required=True)
    resume.add_argument("--evaluations", type=Path, default=Path("simulator/evaluations"))
    resume.add_argument("--tensorboard", type=Path, default=Path("training_logs/factorized_v193_resume"))
    resume.add_argument("--teacher-dataset", type=Path, required=True)
    resume.add_argument("--teacher-batch-size", type=int, default=256)
    resume.add_argument("--timesteps", type=int, default=5_000)
    resume.add_argument(
        "--label",
        default=None,
        help=(
            "Distinguishes this chunk's rehearsal/gate report filenames, e.g. "
            "the cumulative step count (10000). Defaults to --timesteps, which "
            "collides with any earlier chunk that used the same chunk size."
        ),
    )
    resume.add_argument("--max-actions", type=int, default=800)
    resume.add_argument("--episode-seconds", type=float, default=120.0)
    resume.add_argument("--rehearsal-recognition-epochs", type=int, default=1)
    resume.add_argument("--rehearsal-calibration-epochs", type=int, default=2)
    resume.add_argument("--rehearsal-learning-rate", type=float, default=2.0e-5)
    resume.add_argument(
        "--gate-episodes",
        type=int,
        default=3,
        help="Seeds per layout for the post-chunk gate; kills_per_simulated_hour becomes the per-episode median, not a pooled rate, once this is >1.",
    )
    resume.add_argument("--gate-episode-seconds", type=float, default=120.0)
    resume.add_argument("--gate-max-actions", type=int, default=800)
    resume.add_argument("--seed", type=int, default=0)
    resume.add_argument("--device", default="auto")

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("curriculum", type=Path)
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("--stage", default="early", choices=("early", "intermediate", "advanced", "all"))
    evaluate.add_argument("--episodes", type=int, default=1)
    evaluate.add_argument("--episode-seconds", type=float, default=120.0)
    evaluate.add_argument("--max-actions", type=int, default=800)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--require-gate", action="store_true")

    dagger = sub.add_parser(
        "dagger-diagnostic",
        help=(
            "Roll a checkpoint's policy and query the scripted teacher at every "
            "visited state, without changing what the policy does. Reports "
            "teacher agreement, per-layout steering confusion, and EVA-opportunity "
            "recall on states the POLICY actually reaches -- and saves those "
            "states with teacher labels as a dataset in the same schema "
            "collect_teacher_dataset_v193 produces, reusable directly as an "
            "aggregation round. Never starts PPO and never modifies the checkpoint."
        ),
    )
    dagger.add_argument("curriculum", type=Path)
    dagger.add_argument("checkpoint", type=Path)
    dagger.add_argument("--stage", default="early", choices=("early", "intermediate", "advanced", "all"))
    dagger.add_argument("--episodes", type=int, default=1)
    dagger.add_argument("--episode-seconds", type=float, default=120.0)
    dagger.add_argument("--max-actions", type=int, default=800)
    dagger.add_argument("--seed", type=int, default=0)
    dagger.add_argument("--device", default="auto")
    dagger.add_argument("--teacher-policy", default="obstacle_aware")
    dagger.add_argument(
        "--dataset-output", type=Path, default=Path("simulator/datasets/factorized_v193_dagger_round.npz")
    )
    dagger.add_argument(
        "--output", type=Path, default=Path("simulator/evaluations/factorized_v193_dagger_diagnostic.json")
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "smoke":
        return run_smoke(args)
    if args.command == "pilot":
        return run_pilot(args)
    if args.command == "evaluate":
        _summary, passed = evaluate_checkpoint_v193(
            args.curriculum,
            args.checkpoint,
            stage=args.stage,
            episodes=args.episodes,
            episode_seconds=args.episode_seconds,
            max_actions=args.max_actions,
            seed=args.seed,
            device=args.device,
            output=args.output,
            require_gate=args.require_gate,
        )
        return 0 if passed or not args.require_gate else 3
    if args.command == "dagger-diagnostic":
        from .dagger_v193 import collect_dagger_diagnostic_v193

        report = collect_dagger_diagnostic_v193(
            args.curriculum,
            args.checkpoint,
            stage=args.stage,
            episodes=args.episodes,
            episode_seconds=args.episode_seconds,
            max_actions=args.max_actions,
            seed=args.seed,
            device=args.device,
            teacher_policy=args.teacher_policy,
            dataset_output=args.dataset_output,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        aggregate = report["aggregate"]
        print(f"Saved DAgger diagnostic: {args.output}", flush=True)
        print(f"Saved teacher-labeled dataset: {report['dataset']} ({report['dataset_samples']} samples)", flush=True)
        print(
            "Aggregate: steering agreement="
            f"{aggregate['mean_steering_agreement_with_teacher']:.3f}, "
            f"event agreement={aggregate['mean_event_agreement_with_teacher']:.3f}, "
            f"missed EVA opportunity ticks={aggregate['missed_eva_opportunity_ticks']}, "
            f"teacher EVA opportunity ticks={aggregate['teacher_eva_opportunity_ticks']}, "
            f"policy EVA recall on those states="
            f"{aggregate['policy_eva_recall_on_teacher_eva_states']}",
            flush=True,
        )
        return 0
    if args.command == "resume-ppo-chunk":
        return resume_ppo_chunk(args)
    raise SystemExit("Unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
