from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np

from farming.actions import (
    FarmingCommand,
    FarmingEvent,
    POLICY_ACTION_NVECS,
    SteeringAction,
    coerce_farming_command,
)
from farming.model_contract import ModelContractMetadata

from .factorized_training import (
    atomic_save_policy,
    rehearse_factorized_policy,
    synthetic_teacher_clone_factorized,
    validate_factorized_policy_contract,
)
from .reward_model import REWARD_CONTRACT_ID, SimulatorRewardConfig
from .scripted_policies import obstacle_aware_command
from .synthetic import SyntheticCurriculumEnv, iter_variant_environments

ACTION_CONTRACT_ID = "latched-forward-factorized-steering-event-v1"
TIMING_CONTRACT_ID = "fixed-simulated-time-v1"
INVALID_EVA_CONTRACT_ID = "control-interval-and-restart-cooldown-v1"


def _command_array(value: object) -> np.ndarray:
    command = coerce_farming_command(value)
    return np.asarray(command.as_array(), dtype=np.int64)


def _evaluate_env(
    env,
    selector: Callable[[np.ndarray, object], object],
    *,
    episodes: int,
    max_actions: int,
    seed: int,
    label: str,
) -> dict[str, object]:
    rewards: list[float] = []
    kills: list[int] = []
    elapsed_values: list[float] = []
    valid_casts: list[int] = []
    invalid_casts: list[int] = []
    contacts: list[int] = []
    distances: list[float] = []
    path_efficiencies: list[float] = []
    component_totals: list[dict[str, float]] = []
    steering_counts: Counter[int] = Counter()
    event_counts: Counter[int] = Counter()
    command_counts: Counter[tuple[int, int]] = Counter()
    started = time.perf_counter()

    for episode in range(int(episodes)):
        observation, _ = env.reset(seed=seed + episode)
        reward_total = 0.0
        last_info: dict[str, object] = {}
        used = 0
        for _ in range(int(max_actions)):
            action = _command_array(selector(observation, env))
            steering = int(action[0])
            event = int(action[1])
            steering_counts[steering] += 1
            event_counts[event] += 1
            command_counts[(steering, event)] += 1
            used += 1
            observation, reward, terminated, truncated, last_info = env.step(action)
            reward_total += float(reward)
            if terminated or truncated:
                break
        elapsed = float(last_info.get("elapsed_seconds", 0.0))
        episode_kills = int(last_info.get("total_kills", 0))
        rewards.append(reward_total)
        kills.append(episode_kills)
        elapsed_values.append(elapsed)
        valid_casts.append(int(last_info.get("valid_eva_casts", 0)))
        invalid_casts.append(int(last_info.get("invalid_eva_attempts", 0)))
        contacts.append(int(last_info.get("contacts", 0)))
        distances.append(float(last_info.get("total_distance_cells", 0.0)))
        path_efficiencies.append(float(last_info.get("path_efficiency", 0.0)))
        raw_components = last_info.get("reward_component_totals", {})
        component_totals.append(
            {str(k): float(v) for k, v in raw_components.items()}
            if isinstance(raw_components, dict)
            else {}
        )
        kill_rate = episode_kills * 3600.0 / max(1e-9, elapsed)
        print(
            f"[{label}] episode {episode + 1}/{episodes}: reward={reward_total:.3f}, "
            f"kills={episode_kills}, kills/hour={kill_rate:.1f}, "
            f"valid EVA={valid_casts[-1]}, actions={used}, "
            f"simulated={elapsed:.1f}s, wall={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    total_elapsed = max(1e-9, sum(elapsed_values))
    total_commands = max(1, sum(command_counts.values()))
    component_names = sorted({name for item in component_totals for name in item})
    # Per-episode rates, not the pooled sum-over-total-time rate: pooling lets
    # one strong episode mask a weak one (or vice versa) inside a single
    # number. A single seed/episode run degenerates to exactly the pooled
    # value (median of one value), so this is a no-op for every existing
    # episodes=1 caller -- it only changes semantics once episodes>1, which
    # is exactly where the old pooled figure was misleading gate decisions.
    episode_kill_rates = [
        k * 3600.0 / max(1e-9, e) for k, e in zip(kills, elapsed_values, strict=True)
    ]
    return {
        "episodes": int(episodes),
        "mean_reward": float(np.mean(rewards)),
        "mean_kills": float(np.mean(kills)),
        "kills_per_simulated_hour": float(np.median(episode_kill_rates)),
        "pooled_kills_per_simulated_hour": float(sum(kills) * 3600.0 / total_elapsed),
        "kills_per_simulated_hour_min": float(np.min(episode_kill_rates)),
        "kills_per_simulated_hour_max": float(np.max(episode_kill_rates)),
        "kills_per_simulated_hour_p25": float(np.percentile(episode_kill_rates, 25)),
        "kills_per_simulated_hour_p75": float(np.percentile(episode_kill_rates, 75)),
        "episode_kills_per_simulated_hour": episode_kill_rates,
        "mean_simulated_seconds": float(np.mean(elapsed_values)),
        "mean_valid_eva_casts": float(np.mean(valid_casts)),
        "mean_invalid_eva_attempts": float(np.mean(invalid_casts)),
        "kills_per_valid_eva": float(sum(kills) / max(1, sum(valid_casts))),
        "mean_contacts": float(np.mean(contacts)),
        "mean_total_distance_cells": float(np.mean(distances)),
        "mean_path_efficiency": float(np.mean(path_efficiencies)),
        "steering_counts": {str(i): int(steering_counts[i]) for i in range(3)},
        "steering_probabilities": {
            str(i): float(steering_counts[i] / total_commands) for i in range(3)
        },
        "event_counts": {str(i): int(event_counts[i]) for i in range(3)},
        "event_probabilities": {
            str(i): float(event_counts[i] / total_commands) for i in range(3)
        },
        "command_counts": {
            f"{s},{e}": int(command_counts[(s, e)])
            for s in range(3)
            for e in range(3)
        },
        "mean_reward_components": {
            name: float(np.mean([item.get(name, 0.0) for item in component_totals]))
            for name in component_names
        },
        "episode_rewards": rewards,
        "episode_kills": kills,
    }


def _policy_gate_reasons(
    layouts: list[dict[str, object]],
    *,
    maximum_layout_steering_fraction: float = 0.95,
    maximum_layout_jump_fraction: float = 0.35,
    minimum_aggregate_steering_fraction: float = 0.005,
) -> tuple[list[str], dict[str, float], dict[str, float], float, float, float]:
    """Apply aggregate and per-layout anti-collapse gates."""

    random_kph = float(np.mean([item["random"]["kills_per_simulated_hour"] for item in layouts]))
    policy_kph = float(np.mean([item["policy"]["kills_per_simulated_hour"] for item in layouts]))
    steering_probabilities = {
        str(i): float(np.mean([item["policy"]["steering_probabilities"][str(i)] for item in layouts]))
        for i in range(3)
    }
    event_probabilities = {
        str(i): float(np.mean([item["policy"]["event_probabilities"][str(i)] for item in layouts]))
        for i in range(3)
    }
    reasons: list[str] = []
    ratio = policy_kph / max(1e-9, random_kph)
    if policy_kph <= 0.0:
        reasons.append("policy produced no kills")
    if ratio < 1.0:
        reasons.append(f"policy/random kill-rate ratio {ratio:.3f} is below 1.000")
    if max(steering_probabilities.values()) > 0.90:
        reasons.append("aggregate steering collapsed above 90% to one choice")
    for steering in (SteeringAction.STRAIGHT, SteeringAction.LEFT, SteeringAction.RIGHT):
        probability = steering_probabilities[str(int(steering))]
        if probability < float(minimum_aggregate_steering_fraction):
            reasons.append(
                f"aggregate steering {steering.name} probability {probability:.4f} is below "
                f"{minimum_aggregate_steering_fraction:.4f}"
            )
    if event_probabilities[str(int(FarmingEvent.CAST_EVA))] <= 0.0:
        reasons.append("policy selected no EVA events")

    inactive: list[str] = []
    for item in layouts:
        variant = str(item["variant"])
        policy = item["policy"]
        layout_reasons: list[str] = []
        if float(policy["kills_per_simulated_hour"]) <= 0.0:
            layout_reasons.append("no kills")
        if float(policy["mean_valid_eva_casts"]) <= 0.0:
            layout_reasons.append("no valid EVA")
        layout_steering = {str(k): float(v) for k, v in policy["steering_probabilities"].items()}
        dominant_steering = max(layout_steering.values())
        if dominant_steering > float(maximum_layout_steering_fraction):
            layout_reasons.append(
                f"steering choice reached {dominant_steering:.3f} "
                f"> {maximum_layout_steering_fraction:.3f}"
            )
        jump_fraction = float(policy["event_probabilities"][str(int(FarmingEvent.JUMP))])
        if jump_fraction > float(maximum_layout_jump_fraction):
            layout_reasons.append(
                f"jump fraction {jump_fraction:.3f} > {maximum_layout_jump_fraction:.3f}"
            )
        if float(policy["mean_contacts"]) >= 100.0 and float(policy["mean_kills"]) <= 0.0:
            layout_reasons.append("high-contact zero-kill behavior")
        item["gate"] = {"passed": not layout_reasons, "reasons": layout_reasons}
        if layout_reasons:
            inactive.append(f"{variant} ({'; '.join(layout_reasons)})")
    if inactive:
        reasons.append("per-layout gate failures: " + ", ".join(inactive))
    return reasons, steering_probabilities, event_probabilities, random_kph, policy_kph, ratio


def _balanced_training_vec_env(
    curriculum: Path,
    *,
    stage: str,
    seed: int,
    episode_seconds: float,
    max_actions: int,
):
    """Create one monitored training environment per layout."""

    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    pairs = list(
        iter_variant_environments(
            curriculum,
            stage=stage,
            seed=seed,
            episode_steps=max_actions,
            episode_seconds=episode_seconds,
        )
    )
    if not pairs:
        raise ValueError(f"No synthetic layouts are available for stage {stage!r}")
    names = [entry.name for entry, _env in pairs]
    factories = []
    for entry, env in pairs:
        name = entry.name

        def make_env(raw_env=env, variant_name=name):
            monitored = Monitor(raw_env)
            setattr(monitored, "synthetic_variant", variant_name)
            return monitored

        factories.append(make_env)
    return DummyVecEnv(factories), names


def evaluate_checkpoint(
    curriculum: Path,
    checkpoint: Path,
    *,
    stage: str = "early",
    episodes: int = 1,
    episode_seconds: float = 60.0,
    max_actions: int = 400,
    seed: int = 0,
    device: str = "auto",
    output: Path | None = None,
    require_gate: bool = False,
) -> tuple[dict[str, object], bool]:
    try:
        from stable_baselines3 import PPO
    except ImportError as error:
        raise SystemExit("Install requirements-training.txt before evaluation") from error

    policy = PPO.load(str(checkpoint), device=device)
    validate_factorized_policy_contract(policy)
    layouts: list[dict[str, object]] = []
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

        def policy_selector(obs, _env):
            action, _state = policy.predict(obs, deterministic=True)
            return action

        print(f"Evaluating factorized layout {index + 1}: {entry.name}", flush=True)
        random_report = _evaluate_env(
            env,
            random_selector,
            episodes=episodes,
            max_actions=max_actions,
            seed=seed + index * 100,
            label=f"{entry.name}/random",
        )
        policy_report = _evaluate_env(
            env,
            policy_selector,
            episodes=episodes,
            max_actions=max_actions,
            seed=seed + index * 100,
            label=f"{entry.name}/policy",
        )
        layouts.append({"variant": entry.name, "random": random_report, "policy": policy_report})
        env.close()

    (
        reasons,
        steering_probabilities,
        event_probabilities,
        random_kph,
        policy_kph,
        ratio,
    ) = _policy_gate_reasons(layouts)

    summary = {
        "checkpoint": str(checkpoint.resolve()),
        "action_contract": ACTION_CONTRACT_ID,
        "action_nvec": list(POLICY_ACTION_NVECS),
        "observation_size": 923,
        "stage": stage,
        "episode_seconds": episode_seconds,
        "layouts": layouts,
        "aggregate": {
            "policy_kills_per_simulated_hour": policy_kph,
            "random_kills_per_simulated_hour": random_kph,
            "policy_to_random_kill_rate_ratio": ratio,
            "policy_steering_probabilities": steering_probabilities,
            "policy_event_probabilities": event_probabilities,
        },
        "stage_gate": {"passed": not reasons, "reasons": reasons},
    }
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved: {output}", flush=True)
    if require_gate and reasons:
        return summary, False
    return summary, not reasons


def run_pilot(args: argparse.Namespace) -> int:
    try:
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
    print(
        "Balanced PPO environments: " + ", ".join(training_layouts),
        flush=True,
    )
    policy = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=256,
        batch_size=64,
        n_epochs=4,
        learning_rate=5e-5,
        clip_range=0.10,
        target_kl=0.015,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.02,
        seed=args.seed,
        device=args.device,
        tensorboard_log=str(args.tensorboard),
    )
    teacher_report = synthetic_teacher_clone_factorized(
        policy,
        args.curriculum,
        stage="early",
        samples=args.teacher_samples,
        episode_seconds=30.0,
        max_actions=220,
        teacher_policy="obstacle_aware",
        epochs=10,
        batch_size=256,
        learning_rate=3e-4,
        seed=args.seed,
        raise_on_gate_failure=False,
        dataset_output=args.teacher_dataset,
    )
    args.evaluations.mkdir(parents=True, exist_ok=True)
    teacher_clone_gate = args.evaluations / "factorized_v192_teacher_clone_gate.json"
    teacher_clone_gate.write_text(
        json.dumps(teacher_report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved factorized teacher clone gate: {teacher_clone_gate}", flush=True)
    if not bool(teacher_report.get("validation", {}).get("passed", False)):
        env.close()
        reasons = teacher_report.get("validation", {}).get("reasons", [])
        print(
            "Teacher clone gate failed after balanced-head repair: "
            + "; ".join(str(reason) for reason in reasons),
            flush=True,
        )
        return 3
    metadata = {
        "action_contract": ACTION_CONTRACT_ID,
        "action_nvec": list(POLICY_ACTION_NVECS),
        "forward_latched": True,
        "timing_contract": TIMING_CONTRACT_ID,
        "invalid_eva_contract": INVALID_EVA_CONTRACT_ID,
        "reward_contract": REWARD_CONTRACT_ID,
        "reward_config": SimulatorRewardConfig().as_dict(),
        "pilot_contract": "layout-balanced-teacher-rehearsal-v1",
        "training_layouts": training_layouts,
        "training_episode_seconds": float(args.episode_seconds),
        "teacher_dataset": str(args.teacher_dataset.resolve()),
        "teacher": teacher_report,
    }
    setattr(policy, "farming_contract_metadata", ModelContractMetadata.current().as_dict())
    setattr(policy, "synthetic_curriculum_metadata", metadata)
    teacher_path = args.output.with_name(args.output.stem + "_teacher.zip")
    atomic_save_policy(policy, teacher_path)
    print(f"Saved factorized teacher checkpoint: {teacher_path}", flush=True)
    _, teacher_passed = evaluate_checkpoint(
        args.curriculum,
        teacher_path,
        stage="early",
        episodes=1,
        episode_seconds=60.0,
        max_actions=args.max_actions,
        seed=args.seed,
        device=args.device,
        output=args.evaluations / "factorized_v192_teacher_gate.json",
        require_gate=True,
    )
    if not teacher_passed:
        env.close()
        print("Teacher gate failed; PPO was not started.", flush=True)
        return 3

    completed = 0
    while completed < args.timesteps:
        chunk = min(args.chunk_size, args.timesteps - completed)
        policy.learn(
            total_timesteps=chunk,
            reset_num_timesteps=False,
            progress_bar=True,
            tb_log_name="factorized_v192_pilot",
        )
        completed += chunk
        rehearsal = rehearse_factorized_policy(
            policy,
            args.teacher_dataset,
            epochs=args.rehearsal_epochs,
            batch_size=256,
            learning_rate=args.rehearsal_learning_rate,
            event_loss_scale=args.rehearsal_event_loss_scale,
            seed=args.seed + completed,
        )
        rehearsal_path = args.evaluations / f"factorized_v192_{completed}_rehearsal.json"
        rehearsal_path.write_text(json.dumps(rehearsal, indent=2) + "\n", encoding="utf-8")
        print(f"Saved teacher rehearsal report: {rehearsal_path}", flush=True)
        if not bool(rehearsal.get("validation", {}).get("passed", False)):
            env.close()
            print(
                f"Pilot stopped at {completed} steps because teacher rehearsal failed: "
                + "; ".join(str(reason) for reason in rehearsal["validation"]["reasons"]),
                flush=True,
            )
            return 3
        setattr(policy, "farming_contract_metadata", ModelContractMetadata.current().as_dict())
        setattr(
            policy,
            "synthetic_curriculum_metadata",
            {**metadata, "pilot_timesteps": completed, "last_rehearsal": rehearsal},
        )
        checkpoint = args.output.with_name(args.output.stem + f"_{completed}.zip")
        atomic_save_policy(policy, checkpoint)
        _, passed = evaluate_checkpoint(
            args.curriculum,
            checkpoint,
            stage="early",
            episodes=1,
            episode_seconds=60.0,
            max_actions=args.max_actions,
            seed=args.seed,
            device=args.device,
            output=args.evaluations / f"factorized_v192_{completed}_gate.json",
            require_gate=True,
        )
        if not passed:
            env.close()
            print(f"Pilot stopped at {completed} steps because the gate failed.", flush=True)
            return 3
    atomic_save_policy(policy, args.output)
    env.close()
    print(f"Pilot complete: {args.output}", flush=True)
    return 0


def run_smoke(args: argparse.Namespace) -> int:
    from .synthetic import SyntheticCurriculumEnv

    env = SyntheticCurriculumEnv(
        args.curriculum,
        stage="early",
        seed=0,
        episode_steps=20,
        episode_seconds=2.0,
    )
    observation, info = env.reset(seed=0)
    assert observation.shape == (923,)
    assert tuple(int(v) for v in env.action_space.nvec.tolist()) == POLICY_ACTION_NVECS
    commands = (
        FarmingCommand(SteeringAction.STRAIGHT, FarmingEvent.NONE),
        FarmingCommand(SteeringAction.LEFT, FarmingEvent.CAST_EVA),
        FarmingCommand(SteeringAction.RIGHT, FarmingEvent.JUMP),
    )
    for command in commands:
        observation, _reward, terminated, truncated, info = env.step(command.as_array())
        assert observation.shape == (923,)
        assert np.all(np.isfinite(observation))
        if terminated or truncated:
            break
    env.close()
    print(
        json.dumps(
            {
                "passed": True,
                "observation_shape": [923],
                "action_space": "MultiDiscrete([3, 3])",
                "forward_latched": True,
                "last_info": info,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FlyFF factorized steering/event simulator v1.9.2")
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("curriculum", type=Path)

    pilot = sub.add_parser("pilot")
    pilot.add_argument("curriculum", type=Path)
    pilot.add_argument("--output", type=Path, default=Path("models/generic_farming_v192_pilot.zip"))
    pilot.add_argument("--evaluations", type=Path, default=Path("evaluations"))
    pilot.add_argument("--tensorboard", type=Path, default=Path("training_logs/factorized_v192"))
    pilot.add_argument("--timesteps", type=int, default=25_000)
    pilot.add_argument("--chunk-size", type=int, default=5_000)
    pilot.add_argument("--teacher-samples", type=int, default=6_000)
    pilot.add_argument("--episode-seconds", type=float, default=90.0)
    pilot.add_argument("--max-actions", type=int, default=600)
    pilot.add_argument(
        "--teacher-dataset",
        type=Path,
        default=Path("datasets/factorized_v192_teacher.npz"),
    )
    pilot.add_argument("--rehearsal-epochs", type=int, default=2)
    pilot.add_argument("--rehearsal-learning-rate", type=float, default=2.5e-5)
    pilot.add_argument("--rehearsal-event-loss-scale", type=float, default=1.75)
    pilot.add_argument("--seed", type=int, default=0)
    pilot.add_argument("--device", default="auto")

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("curriculum", type=Path)
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("--stage", default="early", choices=("early", "intermediate", "advanced", "all"))
    evaluate.add_argument("--episodes", type=int, default=1)
    evaluate.add_argument("--episode-seconds", type=float, default=60.0)
    evaluate.add_argument("--max-actions", type=int, default=400)
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
        _summary, passed = evaluate_checkpoint(
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
