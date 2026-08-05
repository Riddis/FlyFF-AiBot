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
    collect_teacher_dataset_v193,
    rehearse_factorized_policy_v193,
    train_teacher_clone_v193,
)
from .reward_model import REWARD_CONTRACT_ID, SimulatorRewardConfig
from .scripted_policies import scripted_command
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
        raise SystemExit("Install requirements-training.txt before evaluation") from error

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
        raise SystemExit("Install requirements-training.txt before training") from error

    env, training_layouts = _balanced_training_vec_env(
        args.curriculum,
        stage="early",
        seed=args.seed,
        max_actions=args.max_actions,
        episode_seconds=args.episode_seconds,
    )
    print("Balanced PPO environments: " + ", ".join(training_layouts), flush=True)
    policy = PPO(
        "MlpPolicy",
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
            "net_arch": {"pi": [256, 128], "vf": [256, 128]},
            "activation_fn": torch.nn.ReLU,
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

    teacher_report = train_teacher_clone_v193(
        policy,
        args.teacher_dataset,
        recognition_epochs=args.recognition_epochs,
        recognition_learning_rate=args.recognition_learning_rate,
        calibration_epochs=args.calibration_epochs,
        calibration_learning_rate=args.calibration_learning_rate,
        batch_size=args.teacher_batch_size,
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
        "teacher_clone": teacher_report,
        "policy_architecture": {"pi": [256, 128], "vf": [256, 128], "activation": "ReLU"},
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FlyFF calibrated factorized pilot v1.9.3")
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("curriculum", type=Path)

    pilot = sub.add_parser("pilot")
    pilot.add_argument("curriculum", type=Path)
    pilot.add_argument("--output", type=Path, default=Path("models/generic_farming_v193_pilot.zip"))
    pilot.add_argument("--evaluations", type=Path, default=Path("evaluations"))
    pilot.add_argument("--tensorboard", type=Path, default=Path("training_logs/factorized_v193"))
    pilot.add_argument("--teacher-dataset", type=Path, default=Path("datasets/factorized_v193_teacher.npz"))
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
    pilot.add_argument("--gate-episodes", type=int, default=1)
    pilot.add_argument("--gate-episode-seconds", type=float, default=120.0)
    pilot.add_argument("--gate-max-actions", type=int, default=800)
    pilot.add_argument("--rehearsal-recognition-epochs", type=int, default=1)
    pilot.add_argument("--rehearsal-calibration-epochs", type=int, default=2)
    pilot.add_argument("--rehearsal-learning-rate", type=float, default=2.0e-5)
    pilot.add_argument("--seed", type=int, default=0)
    pilot.add_argument("--device", default="auto")

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
    raise SystemExit("Unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
