from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EvaluationSummary:
    episodes: int
    completed: int
    mean_coverage: float
    mean_reward: float
    mean_steps: float


def evaluate_policy(
    model_path: Path,
    *,
    episodes: int = 100,
    seed: int = 90_000,
) -> EvaluationSummary:
    try:
        from stable_baselines3 import PPO
    except ImportError as error:
        raise RuntimeError(
            "Evaluation requires stable-baselines3 and gymnasium."
        ) from error
    from mapper.rl.GymEnv import MapperSimEnv

    model = PPO.load(str(model_path))
    env = MapperSimEnv()
    coverages: list[float] = []
    rewards: list[float] = []
    steps: list[int] = []
    completed = 0
    try:
        for episode in range(episodes):
            observation, _info = env.reset(seed=seed + episode)
            total_reward = 0.0
            episode_steps = 0
            terminated = False
            truncated = False
            info: dict[str, object] = {}
            while not (terminated or truncated):
                action, _state = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(
                    int(action.item() if hasattr(action, "item") else action)
                )
                total_reward += float(reward)
                episode_steps += 1
            coverage = float(info.get("coverage", 0.0))
            coverages.append(coverage)
            rewards.append(total_reward)
            steps.append(episode_steps)
            completed += int(terminated)
    finally:
        env.close()

    return EvaluationSummary(
        episodes=episodes,
        completed=completed,
        mean_coverage=float(np.mean(coverages)),
        mean_reward=float(np.mean(rewards)),
        mean_steps=float(np.mean(steps)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate mapper RL policy.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/mapper_explorer_ppo.zip"),
    )
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    summary = evaluate_policy(args.model, episodes=args.episodes)
    print(
        f"episodes={summary.episodes} completed={summary.completed} "
        f"mean_coverage={summary.mean_coverage:.3f} "
        f"mean_reward={summary.mean_reward:.2f} "
        f"mean_steps={summary.mean_steps:.1f}"
    )


if __name__ == "__main__":
    main()
