"""2026-08-11: single-obstacle fixed-waypoint GoalNav training, run after
scratchpad_single_obstacle_symmetry_check.py passed all mandatory checks
(array-level mirror symmetry, reachability, clear-direct-path for the
"none" case).

Tests whether PPO learns genuine mirrored local detours around a single
obstacle: LEFT detour when the right side is blocked, RIGHT detour when
the left side is blocked, ~STRAIGHT when the obstacle doesn't block the
path at all -- without becoming a general router (fixed heading=0.0,
single waypoint, single obstacle; see simulator/single_obstacle_env.py's
module docstring for the explicit architecture scoping).

Current/reference movement model only (terminology corrected 2026-08-11;
no more paired symmetric-vs-reference ablation -- that question was
answered by the open-map stage). Geodesic reward (not euclidean). Success
within the waypoint radius (geodesic proximity, not just euclidean --
see single_obstacle_env.py's fix for why). Physical contact = failure.

Held-out evaluation set generated from a seed disjoint from every
training RNG range, per-gap_side breakdown reported separately (mirrored-
case comparison) alongside the aggregate.

Usage: python scratchpad_single_obstacle_train.py [total_timesteps=60000] [checkpoint_interval=15000]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from farming.actions import SteeringAction
from simulator.navigation_history import NavigationHistoryWrapper
from simulator.single_obstacle_env import (
    GAP_SIDES, SingleObstacleWrapper, held_out_obstacle_specs_for_side, make_single_obstacle_env, sample_obstacle_spec,
)
from simulator.split_branch_policy import SplitSteeringNavigationPolicy
from simulator.static_waypoint_env import SUCCESS_RADIUS_CELLS
from simulator.synthetic import iter_variant_environments
from simulator.world_model import MovementModel

SEEDS = [0, 2, 8]
EPISODE_STEPS = 120
N_ENVS = 4
N_STEPS = 256
BATCH_SIZE = 256
N_EPOCHS = 10
LEARNING_RATE = 3.0e-4
CLIP_RANGE = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENT_COEF = 0.02
POLICY_KWARGS = {"steering_net_arch": [64, 32], "event_net_arch": [64, 32], "vf_net_arch": [64, 32]}
STEERING_NAMES = {0: "STRAIGHT", 1: "LEFT", 2: "RIGHT"}

N_EVAL_SPECS_PER_SIDE = 15
EVAL_SPEC_SEED = 778_000_000  # disjoint from training RNG ranges (seed*1000+idx below)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_reference_movement() -> tuple[MovementModel, ...]:
    entry, env = next(iter(iter_variant_environments(
        "simulator/curricula/synthetic_curriculum/curriculum.json", stage="early", seed=0, episode_steps=10, episode_seconds=30.0,
    )))
    movement = env.model.movement
    env.close()
    return movement


def _random_spec(rng: np.random.Generator):
    side = GAP_SIDES[int(rng.integers(0, len(GAP_SIDES)))]
    return sample_obstacle_spec(rng, gap_side=side)


def make_env_factory(seed_offset: int, movement: tuple[MovementModel, ...]):
    def _make():
        rng = np.random.default_rng(seed_offset)
        # Placeholder spec for the throwaway initial construction only --
        # SingleObstacleWrapper.reset() rebuilds env.map from a freshly
        # sampled spec on every reset, including the very first one.
        placeholder_spec = _random_spec(rng)
        raw_env = make_single_obstacle_env(placeholder_spec, episode_steps=EPISODE_STEPS, seed=seed_offset, movement=movement)
        wrapped = SingleObstacleWrapper(
            NavigationHistoryWrapper(raw_env), movement=movement,
            spec_source=lambda rng: _random_spec(rng), rng=rng,
        )
        return Monitor(wrapped)
    return _make


def eval_gap_side(model: PPO, gap_side: str, movement: tuple[MovementModel, ...], *, deterministic: bool) -> dict:
    specs = held_out_obstacle_specs_for_side(N_EVAL_SPECS_PER_SIDE, gap_side=gap_side, seed=EVAL_SPEC_SEED)
    steering_counts = {0: 0, 1: 0, 2: 0}
    successes = 0
    collisions = 0
    steps_to_success: list[int] = []
    path_efficiencies: list[float] = []
    oscillation_rates: list[float] = []
    for i, spec in enumerate(specs):
        raw_env = make_single_obstacle_env(spec, episode_steps=EPISODE_STEPS, seed=910_000_000 + i, movement=movement)
        env = SingleObstacleWrapper(NavigationHistoryWrapper(raw_env), movement=movement, spec_source=lambda rng, spec=spec: spec)
        obs, _info = env.reset(seed=910_000_000 + i)
        initial_geodesic = env.initial_geodesic_cells
        steering_sequence: list[int] = []
        for tick in range(EPISODE_STEPS):
            action, _state = model.predict(obs, deterministic=deterministic)
            steering_counts[int(action[0])] += 1
            steering_sequence.append(int(action[0]))
            obs, reward, term, trunc, info = env.step(action)
            if term:
                if reward >= 1.0:
                    successes += 1
                    steps_to_success.append(tick + 1)
                    traveled = float(info.get("total_distance_cells", 0.0))
                    required_progress = max(0.0, initial_geodesic - SUCCESS_RADIUS_CELLS)
                    if traveled > 0 and required_progress > 0:
                        path_efficiencies.append(required_progress / traveled)
                elif reward <= -1.0:
                    collisions += 1
                break
            if trunc:
                break
        env.close()
        if len(steering_sequence) > 1:
            switches = sum(1 for a, b in zip(steering_sequence, steering_sequence[1:]) if a != b)
            oscillation_rates.append(switches / (len(steering_sequence) - 1))
    n = len(specs)
    total_ticks = sum(steering_counts.values())
    return {
        "gap_side": gap_side,
        "deterministic": deterministic,
        "n_specs": n,
        "success_rate": successes / n if n else None,
        "collision_rate": collisions / n if n else None,
        "mean_steps_to_success": float(np.mean(steps_to_success)) if steps_to_success else None,
        "mean_path_efficiency": float(np.mean(path_efficiencies)) if path_efficiencies else None,
        "mean_oscillation_rate": float(np.mean(oscillation_rates)) if oscillation_rates else None,
        "steering_fractions": {STEERING_NAMES[k]: v / max(1, total_ticks) for k, v in steering_counts.items()},
    }


def run_seed(seed: int, movement: tuple[MovementModel, ...], total_timesteps: int, checkpoint_interval: int) -> None:
    log(f"=== single-obstacle GoalNav: seed={seed} total_timesteps={total_timesteps} ===")
    torch.manual_seed(seed)
    factories = [make_env_factory(seed * 1000 + idx, movement) for idx in range(N_ENVS)]
    vec_env = SubprocVecEnv(factories)
    model = PPO(
        SplitSteeringNavigationPolicy, vec_env, policy_kwargs=POLICY_KWARGS,
        seed=seed, device="cpu", n_steps=N_STEPS, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE, clip_range=CLIP_RANGE, gamma=GAMMA, gae_lambda=GAE_LAMBDA,
        ent_coef=ENT_COEF, verbose=0,
    )

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    eval_log_path = ROOT / "simulator" / "evaluations" / f"single_obstacle_seed{seed}_checkpoint_evals.json"
    all_evals = []

    done = 0
    while done < total_timesteps:
        chunk = min(checkpoint_interval, total_timesteps - done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
        done = int(model.num_timesteps)
        ckpt_path = models_dir / f"single_obstacle_seed{seed}_{done:07d}.zip"
        model.save(str(ckpt_path))

        results = []
        for side in GAP_SIDES:
            for deterministic in (True, False):
                results.append(eval_gap_side(model, side, movement, deterministic=deterministic))
        all_evals.append({"num_timesteps": done, "results": results})
        eval_log_path.write_text(json.dumps(all_evals, indent=2), encoding="utf-8")

        log(f"[seed={seed} checkpoint @ {done}] (held-out, {N_EVAL_SPECS_PER_SIDE}/side)")
        for r in results:
            mode = "det" if r["deterministic"] else "stoch"
            log(f"    {r['gap_side']:6s} {mode:5s}: success={r['success_rate']} collision={r['collision_rate']} "
                f"steps_to_success={r['mean_steps_to_success']} path_eff={r['mean_path_efficiency']} "
                f"oscillation={r['mean_oscillation_rate']} steering={r['steering_fractions']}")

    vec_env.close()
    log(f"=== seed={seed} COMPLETE ===")


def main() -> None:
    total_timesteps = int(sys.argv[1]) if len(sys.argv) > 1 else 60_000
    checkpoint_interval = int(sys.argv[2]) if len(sys.argv) > 2 else 15_000
    movement = get_reference_movement()
    log(f"reference movement model: LEFT turn_mean={movement[1].turn_mean_radians:.4f} | "
        f"RIGHT turn_mean={movement[2].turn_mean_radians:.4f}")
    for seed in SEEDS:
        run_seed(seed, movement, total_timesteps, checkpoint_interval)


if __name__ == "__main__":
    main()
