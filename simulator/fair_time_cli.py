from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

from farming.actions import FarmingAction
from farming.model_contract import ModelContractMetadata

from .environment import RecordedFarmingEnv
from .reward_model import REWARD_CONTRACT_ID, SimulatorRewardConfig
from .synthetic import SyntheticCurriculumEnv, iter_variant_environments
from .scripted_policies import (
    nearest_group_action,
    nearest_reachable_action,
    obstacle_aware_action,
)
from .training import (
    atomic_save_policy,
    synthetic_teacher_clone,
    validate_policy_contract,
)
from .world_model import RecordedWorldModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FlyFF fair-time, reward-audited farming simulator commands (v1.8)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser(
        "train-synthetic",
        help="Train PPO across synthetic layouts with fixed simulated-time episodes",
    )
    train.add_argument("curriculum", type=Path)
    train.add_argument(
        "--stage", choices=("early", "intermediate", "advanced", "all"), default="all"
    )
    train.add_argument("--timesteps", type=int, default=100_000)
    train.add_argument("--episode-seconds", type=float, default=300.0)
    train.add_argument("--max-actions", type=int, default=2_000)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument(
        "--tensorboard", type=Path, default=Path("training_logs/synthetic_generic_v16")
    )
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="auto")
    train.add_argument("--learning-rate", type=float, default=5.0e-5)
    train.add_argument("--n-steps", type=int, default=256)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--n-epochs", type=int, default=4)
    train.add_argument("--clip-range", type=float, default=0.10)
    train.add_argument("--target-kl", type=float, default=0.015)
    train.add_argument("--gamma", type=float, default=0.995)
    train.add_argument("--gae-lambda", type=float, default=0.95)
    train.add_argument("--ent-coef", type=float, default=0.02)
    train.add_argument("--checkpoint-freq", type=int, default=25_000)
    train.add_argument("--checkpoint-dir", type=Path)
    train.add_argument("--resume", type=Path)
    train.add_argument(
        "--teacher-bootstrap-samples",
        type=int,
        default=0,
        help="Pretrain the actor on feasible scripted synthetic actions before fresh PPO",
    )
    train.add_argument("--teacher-bootstrap-epochs", type=int, default=8)
    train.add_argument("--teacher-bootstrap-batch-size", type=int, default=256)
    train.add_argument("--teacher-bootstrap-learning-rate", type=float, default=3.0e-4)
    train.add_argument("--teacher-bootstrap-episode-seconds", type=float, default=30.0)
    train.add_argument("--teacher-bootstrap-max-actions", type=int, default=220)
    train.add_argument(
        "--teacher-bootstrap-policy",
        choices=("nearest_group", "obstacle_aware"),
        default="obstacle_aware",
    )
    train.add_argument(
        "--teacher-output",
        type=Path,
        help="Optional checkpoint path saved immediately after teacher bootstrap",
    )

    evaluate = sub.add_parser(
        "evaluate-synthetic",
        help="Run matched fixed-time random and policy evaluation on synthetic layouts",
    )
    evaluate.add_argument("curriculum", type=Path)
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument(
        "--stage", choices=("early", "intermediate", "advanced", "all"), default="all"
    )
    evaluate.add_argument("--variant")
    evaluate.add_argument("--episodes-per-layout", type=int, default=1)
    evaluate.add_argument("--episode-seconds", type=float, default=60.0)
    evaluate.add_argument("--max-actions", type=int, default=400)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--torch-threads", type=int, default=1)
    evaluate.add_argument("--progress-every", type=int, default=1)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--minimum-random-ratio", type=float, default=1.0)
    evaluate.add_argument("--maximum-action-probability", type=float, default=0.90)
    evaluate.add_argument("--require-gate", action="store_true")

    compare = sub.add_parser(
        "compare-policies",
        help="Compare random and saved PPO checkpoints over matched simulated time",
    )
    compare.add_argument("model", type=Path)
    compare.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Checkpoint to evaluate. Repeat for multiple checkpoints.",
    )
    compare.add_argument("--episodes", type=int, default=1)
    compare.add_argument("--episode-seconds", type=float, default=120.0)
    compare.add_argument("--max-actions", type=int, default=800)
    compare.add_argument("--seed", type=int, default=0)
    compare.add_argument("--device", default="auto")
    compare.add_argument("--torch-threads", type=int, default=1)
    compare.add_argument("--progress-every", type=int, default=1)
    compare.add_argument("--output", type=Path)

    audit = sub.add_parser(
        "audit-rewards",
        help="Compare scripted baselines before spending time on PPO training",
    )
    audit.add_argument("curriculum", type=Path)
    audit.add_argument(
        "--stage", choices=("early", "intermediate", "advanced", "all"), default="all"
    )
    audit.add_argument("--variant")
    audit.add_argument("--episodes-per-layout", type=int, default=3)
    audit.add_argument("--episode-seconds", type=float, default=10.0)
    audit.add_argument("--max-actions", type=int, default=80)
    audit.add_argument(
        "--layout-limit",
        type=int,
        default=0,
        help="Evaluate only the first N matching layouts; use 0 for all",
    )
    audit.add_argument("--seed", type=int, default=0)
    audit.add_argument("--progress-every", type=int, default=1)
    audit.add_argument("--output", type=Path)
    audit.add_argument("--require-sanity", action="store_true")
    return parser


def _validate_common_limits(*, episodes: int, episode_seconds: float, max_actions: int) -> None:
    if int(episodes) < 1:
        raise SystemExit("episodes must be at least 1")
    if not np.isfinite(float(episode_seconds)) or float(episode_seconds) <= 0.0:
        raise SystemExit("episode-seconds must be finite and positive")
    if int(max_actions) < 1:
        raise SystemExit("max-actions must be positive")


def _validate_ppo_settings(args: argparse.Namespace) -> None:
    if args.timesteps < 0:
        raise SystemExit("timesteps cannot be negative")
    if args.n_steps < 1 or args.batch_size < 1 or args.n_epochs < 1:
        raise SystemExit("n-steps, batch-size, and n-epochs must be positive")
    if args.n_steps % args.batch_size != 0:
        raise SystemExit("--batch-size must divide --n-steps exactly")
    if args.learning_rate <= 0.0:
        raise SystemExit("--learning-rate must be positive")
    if not 0.0 < args.clip_range <= 1.0:
        raise SystemExit("--clip-range must be within (0, 1]")
    if args.target_kl is not None and args.target_kl <= 0.0:
        raise SystemExit("--target-kl must be positive")
    if args.teacher_bootstrap_samples < 0:
        raise SystemExit("--teacher-bootstrap-samples cannot be negative")
    if args.resume is not None and args.teacher_bootstrap_samples:
        raise SystemExit("Teacher bootstrap is only supported for a fresh PPO policy")
    if args.teacher_bootstrap_samples:
        if args.teacher_bootstrap_samples < 500:
            raise SystemExit("--teacher-bootstrap-samples must be at least 500 when enabled")
        if args.teacher_bootstrap_epochs < 1:
            raise SystemExit("--teacher-bootstrap-epochs must be positive")
        if args.teacher_bootstrap_batch_size < 1:
            raise SystemExit("--teacher-bootstrap-batch-size must be positive")
        if args.teacher_bootstrap_learning_rate <= 0.0:
            raise SystemExit("--teacher-bootstrap-learning-rate must be positive")
        _validate_common_limits(
            episodes=1,
            episode_seconds=args.teacher_bootstrap_episode_seconds,
            max_actions=args.teacher_bootstrap_max_actions,
        )
    _validate_common_limits(
        episodes=1,
        episode_seconds=args.episode_seconds,
        max_actions=args.max_actions,
    )


def _resolve_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    candidate = path if path.suffix.lower() == ".zip" else Path(f"{path}.zip")
    if candidate.is_file():
        return candidate
    raise SystemExit(f"Checkpoint does not exist: {path}")


def _parse_checkpoint_specs(values: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    labels: set[str] = set()
    for value in values:
        if "=" not in value:
            raise SystemExit(f"Invalid --checkpoint value {value!r}; expected LABEL=PATH")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        raw_path = raw_path.strip().strip('"')
        if not label or not raw_path:
            raise SystemExit(f"Invalid --checkpoint value {value!r}; expected LABEL=PATH")
        if label in labels:
            raise SystemExit(f"Duplicate checkpoint label: {label}")
        labels.add(label)
        parsed.append((label, _resolve_checkpoint(Path(raw_path))))
    return parsed


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def _set_torch_threads(count: int) -> None:
    if int(count) <= 0:
        return
    import torch

    torch.set_num_threads(int(count))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


class _AtomicCheckpointCallback:
    @staticmethod
    def build(save_freq: int, save_path: Path, name_prefix: str):
        from stable_baselines3.common.callbacks import BaseCallback

        frequency = int(save_freq)
        if frequency <= 0:
            return None
        save_path.mkdir(parents=True, exist_ok=True)

        class Callback(BaseCallback):
            def _on_step(self) -> bool:
                if self.num_timesteps > 0 and self.num_timesteps % frequency == 0:
                    target = save_path / f"{name_prefix}_{self.num_timesteps}_steps"
                    atomic_save_policy(self.model, target)
                return True

        return Callback(verbose=0)


def _run_training(args: argparse.Namespace) -> int:
    try:
        from stable_baselines3 import PPO
    except ImportError as error:
        raise SystemExit(
            "Training requires gymnasium, stable-baselines3, torch, and tensorboard. "
            "Install requirements-training.txt first."
        ) from error

    _validate_ppo_settings(args)
    env = SyntheticCurriculumEnv(
        args.curriculum,
        stage=args.stage,
        seed=args.seed,
        episode_steps=args.max_actions,
        episode_seconds=args.episode_seconds,
    )
    args.tensorboard.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reset_num_timesteps = args.resume is None
    teacher_report: dict[str, Any] | None = None

    if args.resume is not None:
        checkpoint = _resolve_checkpoint(args.resume)
        policy = PPO.load(
            str(checkpoint),
            env=env,
            device=args.device,
            tensorboard_log=str(args.tensorboard),
        )
        validate_policy_contract(policy)
        resume_metadata = getattr(policy, "synthetic_curriculum_metadata", None)
        if not isinstance(resume_metadata, dict) or (
            resume_metadata.get("timing_contract") != "fixed-simulated-time-v1"
            or resume_metadata.get("invalid_eva_contract")
            != "control-interval-and-restart-cooldown-v1"
            or resume_metadata.get("reward_contract") != REWARD_CONTRACT_ID
        ):
            raise ValueError(
                "Resume checkpoint was not produced by the v1.8 EVA-bootstrap "
                "trainer. Start fresh or resume a v1.8 stage checkpoint."
            )
        policy.learning_rate = float(args.learning_rate)
        policy._setup_lr_schedule()
        for group in policy.policy.optimizer.param_groups:
            group["lr"] = float(args.learning_rate)
        policy.n_epochs = int(args.n_epochs)
        policy.gamma = float(args.gamma)
        policy.gae_lambda = float(args.gae_lambda)
        policy.clip_range = lambda _progress: float(args.clip_range)
        policy.target_kl = float(args.target_kl)
        policy.ent_coef = float(args.ent_coef)
        prior_teacher = resume_metadata.get("synthetic_teacher_bootstrap")
        if isinstance(prior_teacher, dict):
            teacher_report = prior_teacher
        print(f"Resuming generic PPO checkpoint: {checkpoint}", flush=True)
    else:
        policy = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(args.tensorboard),
            n_steps=int(args.n_steps),
            batch_size=int(args.batch_size),
            n_epochs=int(args.n_epochs),
            learning_rate=float(args.learning_rate),
            clip_range=float(args.clip_range),
            target_kl=float(args.target_kl),
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            ent_coef=float(args.ent_coef),
            seed=int(args.seed),
            device=args.device,
        )

    if args.teacher_bootstrap_samples:
        print(
            f"Bootstrapping actor from {args.teacher_bootstrap_policy} teacher "
            f"({args.teacher_bootstrap_samples} samples)...",
            flush=True,
        )
        teacher_report = synthetic_teacher_clone(
            policy,
            args.curriculum,
            stage=args.stage,
            samples=int(args.teacher_bootstrap_samples),
            episode_seconds=float(args.teacher_bootstrap_episode_seconds),
            max_actions=int(args.teacher_bootstrap_max_actions),
            teacher_policy=str(args.teacher_bootstrap_policy),
            epochs=int(args.teacher_bootstrap_epochs),
            batch_size=int(args.teacher_bootstrap_batch_size),
            learning_rate=float(args.teacher_bootstrap_learning_rate),
            seed=int(args.seed),
        )
        print(json.dumps({"synthetic_teacher_bootstrap": teacher_report}, indent=2), flush=True)

    setattr(policy, "farming_contract_metadata", ModelContractMetadata.current().as_dict())
    setattr(
        policy,
        "synthetic_curriculum_metadata",
        {
            "curriculum": str(Path(args.curriculum).resolve()),
            "stage": str(args.stage),
            "variants": [item.name for item in env.entries],
            "episode_seconds": float(args.episode_seconds),
            "max_episode_actions": int(args.max_actions),
            "timing_contract": "fixed-simulated-time-v1",
            "invalid_eva_contract": "control-interval-and-restart-cooldown-v1",
            "reward_contract": REWARD_CONTRACT_ID,
            "reward_config": SimulatorRewardConfig().as_dict(),
            "synthetic_teacher_bootstrap": teacher_report,
            "training": {
                "learning_rate": float(args.learning_rate),
                "n_steps": int(args.n_steps),
                "batch_size": int(args.batch_size),
                "n_epochs": int(args.n_epochs),
                "clip_range": float(args.clip_range),
                "target_kl": float(args.target_kl),
                "gamma": float(args.gamma),
                "gae_lambda": float(args.gae_lambda),
                "ent_coef": float(args.ent_coef),
            },
        },
    )

    if args.teacher_output is not None:
        if teacher_report is None or args.resume is not None:
            raise ValueError(
                "--teacher-output requires a fresh run with teacher bootstrap enabled"
            )
        saved_teacher = atomic_save_policy(policy, args.teacher_output)
        print(f"Saved teacher-bootstrap checkpoint: {saved_teacher}", flush=True)

    checkpoint_dir = args.checkpoint_dir or (
        args.output.parent / f"{args.output.name}_checkpoints"
    )
    callback = _AtomicCheckpointCallback.build(
        int(args.checkpoint_freq), checkpoint_dir, args.output.name
    )

    interrupted = False
    try:
        if int(args.timesteps) > 0:
            policy.learn(
                total_timesteps=int(args.timesteps),
                callback=callback,
                progress_bar=True,
                reset_num_timesteps=reset_num_timesteps,
                tb_log_name=f"fair_{args.stage}",
            )
    except KeyboardInterrupt:
        interrupted = True
        print("Training interrupted. Saving current policy atomically...", flush=True)
    finally:
        saved = atomic_save_policy(policy, args.output)
        env.close()

    status = "interrupted" if interrupted else "complete"
    print(f"Saved {status} fair-time generic PPO checkpoint: {saved}", flush=True)
    return 0


def _evaluate_env(
    env: RecordedFarmingEnv,
    selector: Callable[[np.ndarray, RecordedFarmingEnv], int],
    *,
    episodes: int,
    max_actions: int,
    seed: int,
    label: str,
    progress_every: int,
) -> dict[str, object]:
    rewards: list[float] = []
    kills: list[int] = []
    elapsed_values: list[float] = []
    actions_used: list[int] = []
    attempts: list[int] = []
    valid_casts: list[int] = []
    invalid_attempts: list[int] = []
    missed_opportunities: list[int] = []
    distances: list[float] = []
    net_displacements: list[float] = []
    path_efficiencies: list[float] = []
    repeated_rates: list[float] = []
    section_transitions_values: list[int] = []
    contacts: list[int] = []
    episode_reward_components: list[dict[str, float]] = []
    horizon_reached: list[bool] = []
    action_counts: Counter[int] = Counter()
    started = time.perf_counter()

    for episode in range(int(episodes)):
        observation, reset_info = env.reset(seed=seed + episode)
        reward_total = 0.0
        seen: set[tuple[int, int]] = set()
        repeats = 0
        sampled = 0
        transitions = 0
        last_info: dict[str, object] = dict(reset_info)
        used = 0
        initial_cell = env.map.native_to_layout_cell(
            float(reset_info.get("player_x", 0.0)),
            float(reset_info.get("player_z", 0.0)),
        )
        if initial_cell is not None:
            seen.add(initial_cell)
        previous_section = env.map.section(
            float(reset_info.get("player_x", 0.0)),
            float(reset_info.get("player_z", 0.0)),
            section_count=env.model.section_count,
        )

        for _ in range(int(max_actions)):
            action = int(selector(observation, env))
            if not 0 <= action < 5:
                raise ValueError(f"{label} returned invalid action {action}")
            action_counts[action] += 1
            used += 1
            observation, reward, terminated, truncated, last_info = env.step(action)
            reward_total += float(reward)
            cell = env.map.native_to_layout_cell(
                float(last_info.get("player_x", 0.0)),
                float(last_info.get("player_z", 0.0)),
            )
            if cell is not None:
                sampled += 1
                repeats += int(cell in seen)
                seen.add(cell)
            section = env.map.section(
                float(last_info.get("player_x", 0.0)),
                float(last_info.get("player_z", 0.0)),
                section_count=env.model.section_count,
            )
            if section != previous_section:
                transitions += 1
                previous_section = section
            if terminated or truncated:
                break

        episode_elapsed = float(last_info.get("elapsed_seconds", 0.0))
        target_seconds = env.episode_seconds
        horizon_reached.append(
            target_seconds is None
            or math.isclose(episode_elapsed, target_seconds, rel_tol=0.0, abs_tol=1.0e-8)
        )
        episode_kills = int(last_info.get("total_kills", 0))
        rewards.append(reward_total)
        kills.append(episode_kills)
        elapsed_values.append(episode_elapsed)
        actions_used.append(used)
        attempts.append(int(last_info.get("eva_attempts", 0)))
        valid_casts.append(int(last_info.get("valid_eva_casts", 0)))
        invalid_attempts.append(int(last_info.get("invalid_eva_attempts", 0)))
        missed_opportunities.append(
            int(last_info.get("missed_eva_opportunities", 0))
        )
        distances.append(float(last_info.get("total_distance_cells", 0.0)))
        net_displacements.append(float(last_info.get("net_displacement_cells", 0.0)))
        path_efficiencies.append(float(last_info.get("path_efficiency", 0.0)))
        repeated_rates.append(float(repeats / max(1, sampled)))
        section_transitions_values.append(transitions)
        contacts.append(int(last_info.get("contacts", 0)))
        raw_totals = last_info.get("reward_component_totals", {})
        component_totals = (
            {str(name): float(value) for name, value in raw_totals.items()}
            if isinstance(raw_totals, dict)
            else {}
        )
        episode_reward_components.append(component_totals)

        if progress_every and (
            (episode + 1) % int(progress_every) == 0 or episode + 1 == episodes
        ):
            kill_rate = episode_kills * 3600.0 / max(1.0e-9, episode_elapsed)
            print(
                f"[{label}] Episode {episode + 1}/{episodes}: "
                f"reward={reward_total:.3f}, kills={episode_kills}, "
                f"kills/hour={kill_rate:.1f}, valid EVA={valid_casts[-1]}, "
                f"invalid EVA={invalid_attempts[-1]}, "
                f"missed EVA={missed_opportunities[-1]}, actions={used}, "
                f"simulated={_format_duration(episode_elapsed)}, "
                f"wall={_format_duration(time.perf_counter() - started)}.",
                flush=True,
            )

    total_actions = max(1, sum(action_counts.values()))
    total_elapsed = max(1.0e-9, float(sum(elapsed_values)))
    total_attempts = sum(attempts)
    total_valid_casts = sum(valid_casts)
    total_invalid_attempts = sum(invalid_attempts)
    component_names = sorted(
        {name for totals in episode_reward_components for name in totals}
    )
    mean_reward_components = {
        name: float(
            np.mean([totals.get(name, 0.0) for totals in episode_reward_components])
        )
        for name in component_names
    }
    reward_components_per_simulated_hour = {
        name: float(
            sum(totals.get(name, 0.0) for totals in episode_reward_components)
            * 3600.0
            / total_elapsed
        )
        for name in component_names
    }
    return {
        "episodes": int(episodes),
        "mean_reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "mean_kills": float(np.mean(kills)),
        "kills_per_simulated_hour": float(sum(kills) * 3600.0 / total_elapsed),
        "mean_simulated_seconds": float(np.mean(elapsed_values)),
        "mean_actions": float(np.mean(actions_used)),
        "time_horizon_reached_rate": float(np.mean(horizon_reached)),
        "episodes_short_of_time_horizon": int(len(horizon_reached) - sum(horizon_reached)),
        "mean_eva_attempts": float(np.mean(attempts)),
        "mean_valid_eva_casts": float(np.mean(valid_casts)),
        "mean_invalid_eva_attempts": float(np.mean(invalid_attempts)),
        "mean_missed_eva_opportunities": float(np.mean(missed_opportunities)),
        "valid_eva_rate": float(total_valid_casts / max(1, total_attempts)),
        "invalid_eva_rate": float(total_invalid_attempts / max(1, total_attempts)),
        "kills_per_valid_eva": float(sum(kills) / max(1, total_valid_casts)),
        "mean_total_distance_cells": float(np.mean(distances)),
        "mean_net_displacement_cells": float(np.mean(net_displacements)),
        "mean_path_efficiency": float(np.mean(path_efficiencies)),
        "mean_repeated_cell_rate": float(np.mean(repeated_rates)),
        "mean_section_transitions": float(np.mean(section_transitions_values)),
        "mean_contacts": float(np.mean(contacts)),
        "reward_contract": REWARD_CONTRACT_ID,
        "mean_reward_components": mean_reward_components,
        "reward_components_per_simulated_hour": reward_components_per_simulated_hour,
        "action_counts": {str(action): int(action_counts[action]) for action in range(5)},
        "action_probabilities": {
            str(action): float(action_counts[action] / total_actions) for action in range(5)
        },
        "episode_rewards": rewards,
        "episode_kills": kills,
        "episode_simulated_seconds": elapsed_values,
        "episode_valid_eva_casts": valid_casts,
        "episode_invalid_eva_attempts": invalid_attempts,
        "episode_missed_eva_opportunities": missed_opportunities,
        "episode_reward_components": episode_reward_components,
    }


def _load_policy(path: Path, *, device: str, env: Any | None = None):
    try:
        from stable_baselines3 import PPO
    except ImportError as error:
        raise SystemExit(
            "Evaluation requires stable-baselines3 and torch. "
            "Install requirements-training.txt first."
        ) from error
    checkpoint = _resolve_checkpoint(path)
    policy = PPO.load(str(checkpoint), env=env, device=device)
    validate_policy_contract(policy)
    return policy, checkpoint


def _run_synthetic_evaluation(args: argparse.Namespace) -> int:
    _validate_common_limits(
        episodes=args.episodes_per_layout,
        episode_seconds=args.episode_seconds,
        max_actions=args.max_actions,
    )
    if args.progress_every < 0:
        raise SystemExit("progress-every cannot be negative")
    if args.minimum_random_ratio < 0.0:
        raise SystemExit("minimum-random-ratio cannot be negative")
    if not 0.0 < args.maximum_action_probability <= 1.0:
        raise SystemExit("maximum-action-probability must be within (0, 1]")

    _set_torch_threads(args.torch_threads)
    policy, checkpoint = _load_policy(args.checkpoint, device=args.device)
    policy_metadata = getattr(policy, "synthetic_curriculum_metadata", None)
    if not isinstance(policy_metadata, dict) or (
        policy_metadata.get("timing_contract") != "fixed-simulated-time-v1"
        or policy_metadata.get("invalid_eva_contract")
        != "control-interval-and-restart-cooldown-v1"
        or policy_metadata.get("reward_contract") != REWARD_CONTRACT_ID
    ):
        raise ValueError(
            "Synthetic evaluation requires a v1.8 EVA-bootstrap checkpoint. "
            "Use compare-policies only for diagnostic evaluation of older models."
        )
    reports: list[dict[str, object]] = []
    environments = iter_variant_environments(
        args.curriculum,
        stage=args.stage,
        seed=args.seed,
        episode_steps=args.max_actions,
        episode_seconds=args.episode_seconds,
        variant_name=args.variant,
    )
    for index, (entry, env) in enumerate(environments):
        random_rng = np.random.default_rng(args.seed + index * 10007)

        def random_selector(_observation: np.ndarray, _env: RecordedFarmingEnv) -> int:
            return int(random_rng.integers(0, 5))

        def policy_selector(observation: np.ndarray, _env: RecordedFarmingEnv) -> int:
            action, _state = policy.predict(observation, deterministic=True)
            return int(np.asarray(action).item())

        print(f"Evaluating synthetic layout {index + 1}: {entry.name}", flush=True)
        random_report = _evaluate_env(
            env,
            random_selector,
            episodes=args.episodes_per_layout,
            max_actions=args.max_actions,
            seed=args.seed + index * 100,
            label=f"{entry.name}/random",
            progress_every=args.progress_every,
        )
        policy_report = _evaluate_env(
            env,
            policy_selector,
            episodes=args.episodes_per_layout,
            max_actions=args.max_actions,
            seed=args.seed + index * 100,
            label=f"{entry.name}/policy",
            progress_every=args.progress_every,
        )
        reports.append(
            {
                "variant": entry.name,
                "stage": entry.stage,
                "template": entry.template,
                "density_profile": entry.density_profile,
                "respawn_profile": entry.respawn_profile,
                "random": random_report,
                "policy": policy_report,
            }
        )
        env.close()

    if not reports:
        raise SystemExit("No synthetic layouts matched the requested selection")

    random_kph = float(
        np.mean([item["random"]["kills_per_simulated_hour"] for item in reports])
    )
    policy_kph = float(
        np.mean([item["policy"]["kills_per_simulated_hour"] for item in reports])
    )
    policy_probabilities = {
        str(action): float(
            np.mean(
                [item["policy"]["action_probabilities"][str(action)] for item in reports]
            )
        )
        for action in range(5)
    }
    maximum_action_probability = max(policy_probabilities.values())
    ratio = float(policy_kph / max(1.0e-9, random_kph))
    policy_valid_casts = float(
        np.mean([item["policy"]["mean_valid_eva_casts"] for item in reports])
    )
    reasons: list[str] = []
    inactive_layouts = [
        str(item["variant"])
        for item in reports
        if float(item["policy"]["kills_per_simulated_hour"]) <= 0.0
        or float(item["policy"]["mean_valid_eva_casts"]) <= 0.0
    ]
    if policy_kph <= 0.0:
        reasons.append("policy produced no kills")
    if ratio < args.minimum_random_ratio:
        reasons.append(
            f"policy/random kill-rate ratio {ratio:.3f} is below "
            f"{args.minimum_random_ratio:.3f}"
        )
    if maximum_action_probability > args.maximum_action_probability:
        reasons.append(
            f"one action probability {maximum_action_probability:.3f} exceeds "
            f"{args.maximum_action_probability:.3f}"
        )
    if policy_valid_casts <= 0.0:
        reasons.append("policy completed no valid EVA casts")
    if inactive_layouts:
        reasons.append(
            "policy produced no kills or no valid EVA cast on layouts: "
            + ", ".join(inactive_layouts)
        )

    summary = {
        "checkpoint": str(checkpoint.resolve()),
        "curriculum": str(Path(args.curriculum).resolve()),
        "stage": args.stage,
        "variant": args.variant,
        "episode_seconds": float(args.episode_seconds),
        "max_actions": int(args.max_actions),
        "timing_contract": "fixed-simulated-time-v1",
        "invalid_eva_contract": "control-interval-and-restart-cooldown-v1",
        "reward_contract": REWARD_CONTRACT_ID,
        "reward_config": SimulatorRewardConfig().as_dict(),
        "layouts": reports,
        "aggregate": {
            "policy_kills_per_simulated_hour": policy_kph,
            "random_kills_per_simulated_hour": random_kph,
            "policy_to_random_kill_rate_ratio": ratio,
            "policy_mean_reward": float(
                np.mean([item["policy"]["mean_reward"] for item in reports])
            ),
            "random_mean_reward": float(
                np.mean([item["random"]["mean_reward"] for item in reports])
            ),
            "policy_mean_valid_eva_casts": policy_valid_casts,
            "policy_mean_invalid_eva_attempts": float(
                np.mean(
                    [item["policy"]["mean_invalid_eva_attempts"] for item in reports]
                )
            ),
            "policy_mean_missed_eva_opportunities": float(
                np.mean(
                    [
                        item["policy"]["mean_missed_eva_opportunities"]
                        for item in reports
                    ]
                )
            ),
            "policy_kills_per_valid_eva": float(
                np.mean([item["policy"]["kills_per_valid_eva"] for item in reports])
            ),
            "policy_action_probabilities": policy_probabilities,
            "maximum_policy_action_probability": maximum_action_probability,
        },
        "stage_gate": {
            "passed": not reasons,
            "criteria": {
                "minimum_policy_to_random_kill_rate_ratio": args.minimum_random_ratio,
                "maximum_single_action_probability": args.maximum_action_probability,
                "requires_nonzero_kills": True,
                "requires_valid_eva_cast": True,
                "requires_activity_on_every_layout": True,
            },
            "reasons": reasons,
        },
    }
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved fair-time synthetic evaluation: {args.output}", flush=True)
    return 3 if args.require_gate and reasons else 0



def _run_reward_audit(args: argparse.Namespace) -> int:
    _validate_common_limits(
        episodes=args.episodes_per_layout,
        episode_seconds=args.episode_seconds,
        max_actions=args.max_actions,
    )
    if args.progress_every < 0:
        raise SystemExit("progress-every cannot be negative")
    if args.layout_limit < 0:
        raise SystemExit("layout-limit cannot be negative")

    policy_order = (
        "forward_only",
        "circle_left",
        "circle_right",
        "forward_jump",
        "random",
        "eva_spam",
        "eva_on_cooldown",
        "nearest_reachable_monster",
        "nearest_group",
        "obstacle_aware_farming",
    )
    layout_reports: list[dict[str, object]] = []
    environments = iter_variant_environments(
        args.curriculum,
        stage=args.stage,
        seed=args.seed,
        episode_steps=args.max_actions,
        episode_seconds=args.episode_seconds,
        variant_name=args.variant,
    )
    for index, (entry, env) in enumerate(environments):
        if args.layout_limit and index >= args.layout_limit:
            env.close()
            break
        random_rng = np.random.default_rng(args.seed + index * 10007 + 911)

        def forward_only(_observation: np.ndarray, _env: RecordedFarmingEnv) -> int:
            return int(FarmingAction.RUN_FORWARD)

        def circle_left(_observation: np.ndarray, _env: RecordedFarmingEnv) -> int:
            return int(FarmingAction.RUN_FORWARD_LEFT)

        def circle_right(_observation: np.ndarray, _env: RecordedFarmingEnv) -> int:
            return int(FarmingAction.RUN_FORWARD_RIGHT)

        def forward_jump(_observation: np.ndarray, _env: RecordedFarmingEnv) -> int:
            return int(FarmingAction.RUN_FORWARD_JUMP)

        def random_selector(_observation: np.ndarray, _env: RecordedFarmingEnv) -> int:
            return int(random_rng.integers(0, 5))

        def eva_spam(_observation: np.ndarray, _env: RecordedFarmingEnv) -> int:
            return int(FarmingAction.CAST_EVA)

        def eva_on_cooldown(
            _observation: np.ndarray, active_env: RecordedFarmingEnv
        ) -> int:
            if active_env.eva_available():
                return int(FarmingAction.CAST_EVA)
            return int(FarmingAction.RUN_FORWARD)

        def nearest_group(
            _observation: np.ndarray, active_env: RecordedFarmingEnv
        ) -> int:
            return nearest_group_action(active_env)

        def nearest_reachable_monster(
            _observation: np.ndarray, active_env: RecordedFarmingEnv
        ) -> int:
            return nearest_reachable_action(active_env)

        def obstacle_aware_farming(
            _observation: np.ndarray, active_env: RecordedFarmingEnv
        ) -> int:
            return obstacle_aware_action(active_env)

        selectors: dict[str, Callable[[np.ndarray, RecordedFarmingEnv], int]] = {
            "forward_only": forward_only,
            "circle_left": circle_left,
            "circle_right": circle_right,
            "forward_jump": forward_jump,
            "random": random_selector,
            "eva_spam": eva_spam,
            "eva_on_cooldown": eva_on_cooldown,
            "nearest_reachable_monster": nearest_reachable_monster,
            "nearest_group": nearest_group,
            "obstacle_aware_farming": obstacle_aware_farming,
        }
        print(f"Auditing reward on synthetic layout {index + 1}: {entry.name}", flush=True)
        policies: dict[str, dict[str, object]] = {}
        for policy_name in policy_order:
            policies[policy_name] = _evaluate_env(
                env,
                selectors[policy_name],
                episodes=args.episodes_per_layout,
                max_actions=args.max_actions,
                seed=args.seed + index * 100,
                label=f"{entry.name}/{policy_name}",
                progress_every=args.progress_every,
            )
        layout_reports.append(
            {
                "variant": entry.name,
                "stage": entry.stage,
                "template": entry.template,
                "density_profile": entry.density_profile,
                "respawn_profile": entry.respawn_profile,
                "policies": policies,
            }
        )
        env.close()

    if not layout_reports:
        raise SystemExit("No synthetic layouts matched the requested reward audit")

    aggregate: dict[str, dict[str, object]] = {}
    for policy_name in policy_order:
        reports = [item["policies"][policy_name] for item in layout_reports]
        component_names = sorted(
            {
                name
                for report in reports
                for name in report.get("mean_reward_components", {})
            }
        )
        aggregate[policy_name] = {
            "mean_reward": float(np.mean([item["mean_reward"] for item in reports])),
            "mean_kills_per_simulated_hour": float(
                np.mean([item["kills_per_simulated_hour"] for item in reports])
            ),
            "mean_valid_eva_casts": float(
                np.mean([item["mean_valid_eva_casts"] for item in reports])
            ),
            "mean_invalid_eva_attempts": float(
                np.mean([item["mean_invalid_eva_attempts"] for item in reports])
            ),
            "mean_repeated_cell_rate": float(
                np.mean([item["mean_repeated_cell_rate"] for item in reports])
            ),
            "mean_reward_components": {
                name: float(
                    np.mean(
                        [
                            item.get("mean_reward_components", {}).get(name, 0.0)
                            for item in reports
                        ]
                    )
                )
                for name in component_names
            },
            "minimum_time_horizon_reached_rate": float(
                min(item["time_horizon_reached_rate"] for item in reports)
            ),
        }

    collapsed = ("forward_only", "circle_left", "circle_right")
    best_collapsed_reward = max(
        float(aggregate[name]["mean_reward"]) for name in collapsed
    )
    best_collapsed_kph = max(
        float(aggregate[name]["mean_kills_per_simulated_hour"])
        for name in collapsed
    )
    heuristic_reward = float(aggregate["nearest_group"]["mean_reward"])
    heuristic_kph = float(
        aggregate["nearest_group"]["mean_kills_per_simulated_hour"]
    )
    obstacle_reward = float(aggregate["obstacle_aware_farming"]["mean_reward"])
    obstacle_kph = float(
        aggregate["obstacle_aware_farming"]["mean_kills_per_simulated_hour"]
    )
    random_reward = float(aggregate["random"]["mean_reward"])
    random_kph = float(aggregate["random"]["mean_kills_per_simulated_hour"])
    cooldown_reward = float(aggregate["eva_on_cooldown"]["mean_reward"])
    cooldown_kph = float(
        aggregate["eva_on_cooldown"]["mean_kills_per_simulated_hour"]
    )

    required_failures: list[str] = []
    advisory_warnings: list[str] = []
    if heuristic_reward <= best_collapsed_reward:
        required_failures.append(
            "nearest-group heuristic reward does not beat all collapsed movement baselines"
        )
    if heuristic_kph <= best_collapsed_kph:
        required_failures.append(
            "nearest-group heuristic kill rate does not beat all collapsed movement baselines"
        )
    if obstacle_reward <= best_collapsed_reward or obstacle_kph <= best_collapsed_kph:
        required_failures.append(
            "obstacle-aware farming does not beat collapsed baselines in both reward and kills"
        )
    if any(
        float(report["minimum_time_horizon_reached_rate"]) < 1.0
        for report in aggregate.values()
    ):
        required_failures.append(
            "at least one policy was truncated before the fixed simulated-time horizon"
        )
    if heuristic_reward <= random_reward:
        advisory_warnings.append(
            "random reward is at least as high as nearest-group heuristic reward"
        )
    if heuristic_kph <= random_kph:
        advisory_warnings.append(
            "random kill rate is at least as high as nearest-group heuristic kill rate"
        )
    if heuristic_reward <= cooldown_reward:
        advisory_warnings.append(
            "EVA-on-cooldown reward is at least as high as nearest-group heuristic reward"
        )
    if heuristic_kph <= cooldown_kph:
        advisory_warnings.append(
            "EVA-on-cooldown kill rate is at least as high as nearest-group heuristic kill rate"
        )
    if obstacle_reward < heuristic_reward or obstacle_kph < heuristic_kph:
        advisory_warnings.append(
            "obstacle-aware heuristic does not exceed nearest-group in both reward and kill rate"
        )

    summary = {
        "curriculum": str(Path(args.curriculum).resolve()),
        "stage": args.stage,
        "variant": args.variant,
        "episode_seconds": float(args.episode_seconds),
        "max_actions": int(args.max_actions),
        "layout_limit": int(args.layout_limit),
        "timing_contract": "fixed-simulated-time-v1",
        "invalid_eva_contract": "control-interval-and-restart-cooldown-v1",
        "reward_contract": REWARD_CONTRACT_ID,
        "reward_config": SimulatorRewardConfig().as_dict(),
        "layouts": layout_reports,
        "aggregate": aggregate,
        "sanity": {
            "passed_required_checks": not required_failures,
            "required_checks": {
                "nearest_group_reward_above_collapsed": heuristic_reward
                > best_collapsed_reward,
                "nearest_group_kill_rate_above_collapsed": heuristic_kph
                > best_collapsed_kph,
            },
            "required_failures": required_failures,
            "advisory_warnings": advisory_warnings,
        },
    }
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved reward audit: {args.output}", flush=True)
    return 4 if args.require_sanity and required_failures else 0

def _run_policy_comparison(args: argparse.Namespace) -> int:
    _validate_common_limits(
        episodes=args.episodes,
        episode_seconds=args.episode_seconds,
        max_actions=args.max_actions,
    )
    if args.progress_every < 0:
        raise SystemExit("progress-every cannot be negative")
    _set_torch_threads(args.torch_threads)
    model = RecordedWorldModel.load(args.model)
    specs = _parse_checkpoint_specs(args.checkpoint)
    if not specs:
        raise SystemExit("Provide at least one --checkpoint LABEL=PATH value")

    reports: list[dict[str, object]] = []

    def make_env() -> RecordedFarmingEnv:
        return RecordedFarmingEnv(
            model,
            seed=args.seed,
            episode_steps=args.max_actions,
            episode_seconds=args.episode_seconds,
        )

    random_env = make_env()
    random_rng = np.random.default_rng(args.seed + 991)
    reports.append(
        {
            "policy": "random",
            **_evaluate_env(
                random_env,
                lambda _obs, _env: int(random_rng.integers(0, 5)),
                episodes=args.episodes,
                max_actions=args.max_actions,
                seed=args.seed,
                label="random",
                progress_every=args.progress_every,
            ),
        }
    )
    random_env.close()

    for label, checkpoint_path in specs:
        env = make_env()
        policy, checkpoint = _load_policy(
            checkpoint_path, device=args.device, env=env
        )

        def selector(
            observation: np.ndarray,
            _env: RecordedFarmingEnv,
            *,
            loaded=policy,
        ) -> int:
            action, _state = loaded.predict(observation, deterministic=True)
            return int(np.asarray(action).item())

        report = _evaluate_env(
            env,
            selector,
            episodes=args.episodes,
            max_actions=args.max_actions,
            seed=args.seed,
            label=label,
            progress_every=args.progress_every,
        )
        reports.append(
            {"policy": label, "checkpoint": str(checkpoint.resolve()), **report}
        )
        env.close()

    output = {
        "model": str(Path(args.model).resolve()),
        "episode_seconds": float(args.episode_seconds),
        "max_actions": int(args.max_actions),
        "timing_contract": "fixed-simulated-time-v1",
        "invalid_eva_contract": "control-interval-and-restart-cooldown-v1",
        "reward_contract": REWARD_CONTRACT_ID,
        "reward_config": SimulatorRewardConfig().as_dict(),
        "policies": reports,
    }
    rendered = json.dumps(output, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved fair-time policy comparison: {args.output}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train-synthetic":
        return _run_training(args)
    if args.command == "evaluate-synthetic":
        return _run_synthetic_evaluation(args)
    if args.command == "compare-policies":
        return _run_policy_comparison(args)
    if args.command == "audit-rewards":
        return _run_reward_audit(args)
    raise AssertionError(args.command)
