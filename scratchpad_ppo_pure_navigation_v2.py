"""2026-08-10 corrected PPO pure-navigation ablation.

SUPERSEDED 2026-08-10 (correction #3): pure_navigation_env.py's target
selection was rebuilt around a self-contained single-source-of-truth
selector (`select_target`); this script's `current_target_position` import
predates that rebuild and no longer exists. The runs launched from this
script (`goal_stable`/`goal_normal`/`safety`, all *_0200704.zip
checkpoints) remain useful as historical diagnostic artifacts (see
run_logs/OVERNIGHT_20260809_PIPELINE.md's 2026-08-10 entries) but should
not be retrained from this file as-is. See scratchpad_ppo_pure_navigation_v4.py
for the corrected driver.

Three runs needed (run this script once per row):
  name          reward_mode  target_mode
  safety        safety       normal_target   (locomotion-only baseline;
                                               target features may be
                                               ignored -- do not use this
                                               run to conclude anything
                                               about target selection)
  goal_stable   goal         stable_waypoint
  goal_normal   goal         normal_target

Steering-only (event forced to NONE by PureNavigationWrapper). Checkpoints
+ a small fixed unseen-navigation evaluation every CHECKPOINT_INTERVAL
timesteps, so training can be stopped as soon as the question is answered
rather than always running to TOTAL_TIMESTEPS.

Usage: python scratchpad_ppo_pure_navigation_v2.py <name> <reward_mode> <target_mode> [total_timesteps] [checkpoint_interval]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from farming.actions import SteeringAction
from simulator.curriculum_manifests import load_heldout_manifest
from simulator.navigation_history import NavigationHistoryWrapper
from simulator.pure_navigation_env import PureNavigationWrapper, current_target_position
from simulator.split_branch_policy import SplitSteeringNavigationPolicy
from simulator.synthetic import iter_variant_environments

TRAINING_CURRICULUM = str(ROOT / "curricula" / "synthetic_curriculum" / "curriculum.json")
STAGE = "early"
SEED = 0
EPISODE_SECONDS = 150.0
MAX_ACTIONS = 1000

N_STEPS = 256
BATCH_SIZE = 256
N_EPOCHS = 10
LEARNING_RATE = 3.0e-4
CLIP_RANGE = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENT_COEF = 0.02

# Small, fixed, UNSEEN evaluation set (disjoint from training -- reuses the
# oracle_fresh_confirmation pool built 2026-08-09, seed base 23M, confirmed
# disjoint from every training/tuning seed range). Kept small (3 layouts x
# 2 seeds = 6 episodes) and short (EVAL_MAX_ACTIONS) so per-checkpoint eval
# stays cheap relative to the training chunk itself.
EVAL_LAYOUTS = [
    "01_early_open_field_typical_fast", "05_early_broad_lobes_typical_fast", "09_early_wide_neck_typical_fast",
]
EVAL_SEEDS = [0, 1]
EVAL_MAX_ACTIONS = 400


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_env_factory(entry, target_mode: str, reward_mode: str, seed_offset: int):
    def _make():
        _entry2, raw_env = next(iter(iter_variant_environments(
            TRAINING_CURRICULUM, stage=STAGE, seed=SEED + seed_offset, episode_steps=MAX_ACTIONS,
            episode_seconds=EPISODE_SECONDS, variant_name=entry.name,
        )))
        wrapped = PureNavigationWrapper(
            NavigationHistoryWrapper(raw_env), target_mode=target_mode, reward_mode=reward_mode,
        )
        return Monitor(wrapped)
    return _make


def quick_eval(model, *, target_mode: str, reward_mode: str) -> dict:
    manifest = load_heldout_manifest("evaluations/manifests/oracle_fresh_confirmation.json")
    episodes = []
    for layout_name in EVAL_LAYOUTS:
        for seed in EVAL_SEEDS:
            entry, raw_env = next(iter(iter_variant_environments(
                manifest.curriculum_path, stage=manifest.stage, seed=seed, episode_steps=EVAL_MAX_ACTIONS,
                episode_seconds=EPISODE_SECONDS, variant_name=layout_name,
            )))
            env = PureNavigationWrapper(
                NavigationHistoryWrapper(raw_env), target_mode=target_mode, reward_mode=reward_mode,
            )
            obs, _info = env.reset(seed=seed)
            base_env = env.unwrapped
            steering_choices = []
            collided = False
            steps = 0
            start_distance_cells = float(base_env.total_distance_cells)
            # Sum of the wrapper's own (target-switch-discontinuity-free,
            # see pure_navigation_env.py's 2026-08-10 correction) per-tick
            # goal reward -- NOT a start-vs-end distance delta, which would
            # reintroduce the same "different target at each end" flaw if
            # the target switched anywhere during the episode.
            cumulative_goal_reward = 0.0
            for _tick in range(EVAL_MAX_ACTIONS):
                action, _state = model.predict(obs, deterministic=True)
                steering_choices.append(int(action[0]))
                obs, reward, term, trunc, info = env.step(action)
                steps += 1
                if term and reward <= -1.0:
                    collided = True
                    break
                if reward_mode == "goal":
                    cumulative_goal_reward += float(reward)
                if term or trunc:
                    break
            end_distance_cells = float(base_env.total_distance_cells)
            env.close()

            switches = sum(1 for a, b in zip(steering_choices, steering_choices[1:]) if a != b)
            target_progress = cumulative_goal_reward if reward_mode == "goal" else None

            episodes.append({
                "layout": layout_name, "seed": seed, "collided": collided, "steps": steps,
                "displacement_cells": end_distance_cells - start_distance_cells,
                "target_progress_cells": target_progress,
                "oscillation_rate": switches / max(1, len(steering_choices) - 1) if len(steering_choices) > 1 else 0.0,
                "steering_counts": {
                    SteeringAction(a).name: steering_choices.count(a) for a in (0, 1, 2)
                },
            })

    n = len(episodes)
    collided_count = sum(1 for e in episodes if e["collided"])
    summary = {
        "n_episodes": n,
        "pct_episodes_with_any_collision": 100.0 * collided_count / n,
        "mean_episode_length": float(np.mean([e["steps"] for e in episodes])),
        "mean_displacement_cells": float(np.mean([e["displacement_cells"] for e in episodes])),
        "mean_oscillation_rate": float(np.mean([e["oscillation_rate"] for e in episodes])),
        "episodes": episodes,
    }
    progress_vals = [e["target_progress_cells"] for e in episodes if e["target_progress_cells"] is not None]
    if progress_vals:
        summary["mean_target_progress_cells"] = float(np.mean(progress_vals))
    return summary


def main() -> None:
    name = sys.argv[1]
    reward_mode = sys.argv[2]
    target_mode = sys.argv[3]
    total_timesteps = int(sys.argv[4]) if len(sys.argv) > 4 else 200_000
    checkpoint_interval = int(sys.argv[5]) if len(sys.argv) > 5 else 25_000
    assert reward_mode in ("safety", "goal")
    assert target_mode in ("stable_waypoint", "normal_target")

    log(f"=== PPO pure-navigation v2: name={name} reward_mode={reward_mode} target_mode={target_mode} "
        f"total_timesteps={total_timesteps} checkpoint_interval={checkpoint_interval} ===")

    pairs = list(iter_variant_environments(
        TRAINING_CURRICULUM, stage=STAGE, seed=SEED, episode_steps=MAX_ACTIONS, episode_seconds=EPISODE_SECONDS,
    ))
    for _entry, env in pairs:
        env.close()
    log(f"Training layouts: {[e.name for e, _ in pairs]}")

    factories = [make_env_factory(entry, target_mode, reward_mode, idx) for idx, (entry, _env) in enumerate(pairs)]
    vec_env = SubprocVecEnv(factories)

    model = PPO(
        SplitSteeringNavigationPolicy, vec_env,
        policy_kwargs={"steering_net_arch": [64, 32], "event_net_arch": [64, 32], "vf_net_arch": [64, 32]},
        seed=SEED, device="cpu",
        n_steps=N_STEPS, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, learning_rate=LEARNING_RATE,
        clip_range=CLIP_RANGE, gamma=GAMMA, gae_lambda=GAE_LAMBDA, ent_coef=ENT_COEF, verbose=0,
    )
    log(f"Fresh policy built. observation_space={model.observation_space}")

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    eval_log_path = ROOT / "evaluations" / f"pure_nav_v2_{name}_checkpoint_evals.json"
    all_evals = []

    done = 0
    while done < total_timesteps:
        chunk = min(checkpoint_interval, total_timesteps - done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
        done = int(model.num_timesteps)
        ckpt_path = models_dir / f"pure_nav_v2_{name}_{done:07d}.zip"
        model.save(str(ckpt_path))
        log(f"Checkpoint saved: {ckpt_path} (num_timesteps={done})")

        eval_result = quick_eval(model, target_mode=target_mode, reward_mode=reward_mode)
        eval_result["num_timesteps"] = done
        all_evals.append(eval_result)
        eval_log_path.write_text(json.dumps(all_evals, indent=2, default=str), encoding="utf-8")
        log(f"[eval @ {done}] collision%={eval_result['pct_episodes_with_any_collision']:.1f} "
            f"mean_len={eval_result['mean_episode_length']:.1f} "
            f"mean_displacement={eval_result['mean_displacement_cells']:.1f} "
            f"mean_oscillation={eval_result['mean_oscillation_rate']:.3f} "
            + (f"mean_target_progress={eval_result.get('mean_target_progress_cells', float('nan')):.1f}"
               if "mean_target_progress_cells" in eval_result else ""))

    vec_env.close()
    log("=== RUN COMPLETE ===")


if __name__ == "__main__":
    main()
