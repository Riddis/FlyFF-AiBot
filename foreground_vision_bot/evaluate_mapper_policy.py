from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from project_paths import MAPPING_MODEL_RELATIVE, resolve_app_path


@dataclass(frozen=True)
class EvaluationSummary:
    episodes: int
    completed: int
    mean_coverage: float
    mean_reward: float
    mean_steps: float
    median_coverage: float
    p90_coverage: float
    action_counts: dict[str, int]
    contact_rate: float
    invalid_observation_rate: float
    masked_fallback_rate: float
    forward_rate: float
    turn_rate: float
    recovery_rate: float
    wait_rate: float
    reacquire_rate: float
    recovery_success_rate: float
    stagnation_truncation_rate: float
    mean_max_wait_streak: float
    frontier_escape_rate: float
    frontier_progress_rate: float
    frontier_escape_success_rate: float
    mean_max_no_progress_streak: float


def evaluate_policy(
    model_path: Path,
    *,
    episodes: int = 100,
    seed: int = 90_000,
) -> EvaluationSummary:
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as error:
        raise RuntimeError(
            "Evaluation requires sb3-contrib, stable-baselines3 and gymnasium. "
            "Install with: pip install -r requirements_mapper_rl.txt"
        ) from error
    from mapper.rl.GymEnv import MapperSimEnv
    from mapper.rl.PolicyTypes import MapperAction

    model_path = resolve_app_path(model_path)
    model = MaskablePPO.load(str(model_path))
    env = MapperSimEnv()
    coverages: list[float] = []
    rewards: list[float] = []
    steps: list[int] = []
    maximum_wait_streaks: list[int] = []
    maximum_no_progress_streaks: list[int] = []
    completed = 0
    action_counts: Counter[str] = Counter()
    contact_steps = 0
    invalid_steps = 0
    masked_fallbacks = 0
    recovery_actions = 0
    successful_recoveries = 0
    stagnation_truncations = 0
    total_steps = 0
    frontier_escape_steps = 0
    frontier_progress_steps = 0
    frontier_escape_successes = 0
    try:
        for episode in range(episodes):
            observation, _info = env.reset(seed=seed + episode)
            total_reward = 0.0
            episode_steps = 0
            terminated = False
            truncated = False
            info: dict[str, object] = {}
            episode_max_wait_streak = 0
            episode_max_no_progress_streak = 0
            while not (terminated or truncated):
                action_mask = env.action_masks()
                action, _state = model.predict(
                    observation,
                    deterministic=True,
                    action_masks=action_mask,
                )
                action_value = int(action.item() if hasattr(action, "item") else action)
                observation, reward, terminated, truncated, info = env.step(
                    action_value
                )
                executed_name = str(
                    info.get("executed_action", MapperAction(action_value).name)
                )
                action_counts[executed_name] += 1
                total_steps += 1
                contact_steps += int(info.get("quality") == "CONTACT")
                invalid_steps += int(
                    info.get("quality")
                    in {"CAMERA_OBSCURED", "HEADING_UNAVAILABLE", "UNRESOLVED"}
                )
                masked_fallbacks += int(bool(info.get("action_was_masked", False)))
                is_recovery_action = executed_name in {
                    MapperAction.WAIT.name,
                    MapperAction.REACQUIRE_HEADING.name,
                }
                recovery_actions += int(is_recovery_action)
                successful_recoveries += int(bool(info.get("recovery_succeeded", False)))
                episode_max_wait_streak = max(
                    episode_max_wait_streak,
                    int(info.get("maximum_wait_streak_seen", 0)),
                )
                frontier_escape_steps += int(bool(info.get("frontier_escape_step", False)))
                frontier_progress_steps += int(bool(info.get("frontier_progress", False)))
                frontier_escape_successes += int(
                    bool(info.get("frontier_escape_succeeded", False))
                )
                episode_max_no_progress_streak = max(
                    episode_max_no_progress_streak,
                    int(info.get("maximum_no_progress_streak_seen", 0)),
                )
                total_reward += float(reward)
                episode_steps += 1
            coverage = float(info.get("coverage", 0.0))
            coverages.append(coverage)
            rewards.append(total_reward)
            steps.append(episode_steps)
            maximum_wait_streaks.append(episode_max_wait_streak)
            maximum_no_progress_streaks.append(episode_max_no_progress_streak)
            completed += int(bool(info.get("completed", terminated)))
            stagnation_truncations += int(
                bool(info.get("stagnation_truncated", False))
            )
    finally:
        env.close()

    forward = action_counts[MapperAction.FORWARD.name]
    turns = (
        action_counts[MapperAction.TURN_LEFT.name]
        + action_counts[MapperAction.TURN_RIGHT.name]
    )
    waits = action_counts[MapperAction.WAIT.name]
    reacquires = action_counts[MapperAction.REACQUIRE_HEADING.name]
    recoveries = waits + reacquires
    denominator = max(1, total_steps)
    return EvaluationSummary(
        episodes=episodes,
        completed=completed,
        mean_coverage=float(np.mean(coverages)),
        mean_reward=float(np.mean(rewards)),
        mean_steps=float(np.mean(steps)),
        median_coverage=float(np.median(coverages)),
        p90_coverage=float(np.percentile(coverages, 90)),
        action_counts=dict(sorted(action_counts.items())),
        contact_rate=(contact_steps / denominator),
        invalid_observation_rate=(invalid_steps / denominator),
        masked_fallback_rate=(masked_fallbacks / denominator),
        forward_rate=(forward / denominator),
        turn_rate=(turns / denominator),
        recovery_rate=(recoveries / denominator),
        wait_rate=(waits / denominator),
        reacquire_rate=(reacquires / denominator),
        recovery_success_rate=(successful_recoveries / max(1, recovery_actions)),
        stagnation_truncation_rate=(stagnation_truncations / max(1, episodes)),
        mean_max_wait_streak=float(np.mean(maximum_wait_streaks)),
        frontier_escape_rate=(frontier_escape_steps / denominator),
        frontier_progress_rate=(frontier_progress_steps / denominator),
        frontier_escape_success_rate=(
            frontier_escape_successes / max(1, frontier_escape_steps)
        ),
        mean_max_no_progress_streak=float(np.mean(maximum_no_progress_streaks)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate mapper RL policy.")
    parser.add_argument(
        "--model",
        type=Path,
        default=MAPPING_MODEL_RELATIVE.with_suffix(".zip"),
    )
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    summary = evaluate_policy(args.model, episodes=args.episodes)
    print(
        f"episodes={summary.episodes} completed={summary.completed} "
        f"mean_coverage={summary.mean_coverage:.3f} "
        f"mean_reward={summary.mean_reward:.2f} "
        f"mean_steps={summary.mean_steps:.1f} "
        f"median_coverage={summary.median_coverage:.3f} "
        f"p90_coverage={summary.p90_coverage:.3f} "
        f"contact_rate={summary.contact_rate:.3f} "
        f"invalid_rate={summary.invalid_observation_rate:.3f} "
        f"masked_fallback_rate={summary.masked_fallback_rate:.5f} "
        f"forward_rate={summary.forward_rate:.3f} "
        f"turn_rate={summary.turn_rate:.3f} "
        f"recovery_rate={summary.recovery_rate:.3f} "
        f"wait_rate={summary.wait_rate:.3f} "
        f"reacquire_rate={summary.reacquire_rate:.3f} "
        f"recovery_success_rate={summary.recovery_success_rate:.3f} "
        f"stagnation_truncation_rate={summary.stagnation_truncation_rate:.3f} "
        f"mean_max_wait_streak={summary.mean_max_wait_streak:.2f} "
        f"frontier_escape_rate={summary.frontier_escape_rate:.3f} "
        f"frontier_progress_rate={summary.frontier_progress_rate:.3f} "
        f"frontier_escape_success_rate={summary.frontier_escape_success_rate:.3f} "
        f"mean_max_no_progress_streak={summary.mean_max_no_progress_streak:.1f} "
        f"actions={summary.action_counts}"
    )


if __name__ == "__main__":
    main()
