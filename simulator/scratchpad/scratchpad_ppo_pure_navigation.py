"""2026-08-09 PPO ablation: does direct RL under an unambiguous "collision
ends the episode" incentive learn clean navigation on its own, without any
of the hand-engineered oracle machinery (terminal-continuation gate, robust
escape-BFS, target-selection hysteresis)?

Two conditions (run this script once per condition, same everything else):
  A. stable_waypoint -- target-selection hysteresis margin set effectively
     infinite, so the initially-selected target holds for the whole episode.
  B. normal_target -- the env's actual current default target-selection
     (hysteresis enabled, margin=3.0 cells).

Same training curriculum, same movement noise (unchanged env physics), same
policy architecture (SplitSteeringNavigationPolicy, same net_arch as
build_fresh_basic_policy), same training budget (timesteps, hyperparameters)
between conditions -- ONLY target_mode differs.

Usage: python scratchpad_ppo_pure_navigation.py <stable_waypoint|normal_target> [timesteps]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from simulator.navigation_history import NavigationHistoryWrapper
from simulator.pure_navigation_env import PureNavigationWrapper
from simulator.split_branch_policy import SplitSteeringNavigationPolicy
from simulator.synthetic import iter_variant_environments

TRAINING_CURRICULUM = str(ROOT / "simulator" / "curricula" / "synthetic_curriculum" / "curriculum.json")
STAGE = "early"
SEED = 0
EPISODE_SECONDS = 150.0
MAX_ACTIONS = 1000

# From-scratch PPO hyperparameters -- deliberately NOT the conservative
# fine-tuning defaults in basic_training.build_fresh_basic_policy (lr=5e-5,
# ent_coef=0.015) which are tuned for refining an already-good policy; a
# random-init policy needs a real learning rate and more exploration.
N_STEPS = 256
BATCH_SIZE = 256
N_EPOCHS = 10
LEARNING_RATE = 3.0e-4
CLIP_RANGE = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENT_COEF = 0.02


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_env_factory(entry, mode: str, seed_offset: int):
    def _make():
        _entry2, raw_env = next(iter(iter_variant_environments(
            TRAINING_CURRICULUM, stage=STAGE, seed=SEED + seed_offset, episode_steps=MAX_ACTIONS,
            episode_seconds=EPISODE_SECONDS, variant_name=entry.name,
        )))
        wrapped = PureNavigationWrapper(NavigationHistoryWrapper(raw_env), target_mode=mode)
        monitored = Monitor(wrapped)
        return monitored
    return _make


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "normal_target"
    timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else 300_000
    assert mode in ("stable_waypoint", "normal_target"), mode

    log(f"=== Pure-navigation PPO ablation: mode={mode} timesteps={timesteps} ===")

    pairs = list(iter_variant_environments(
        TRAINING_CURRICULUM, stage=STAGE, seed=SEED, episode_steps=MAX_ACTIONS, episode_seconds=EPISODE_SECONDS,
    ))
    for _entry, env in pairs:
        env.close()
    log(f"Training layouts: {[e.name for e, _ in pairs]}")

    factories = [make_env_factory(entry, mode, idx) for idx, (entry, _env) in enumerate(pairs)]
    vec_env = SubprocVecEnv(factories)

    model = PPO(
        SplitSteeringNavigationPolicy,
        vec_env,
        policy_kwargs={
            "steering_net_arch": [64, 32],
            "event_net_arch": [64, 32],
            "vf_net_arch": [64, 32],
        },
        seed=SEED,
        device="cpu",
        n_steps=N_STEPS, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, learning_rate=LEARNING_RATE,
        clip_range=CLIP_RANGE, gamma=GAMMA, gae_lambda=GAE_LAMBDA, ent_coef=ENT_COEF,
        verbose=0,
    )
    log(f"Fresh policy built. observation_space={model.observation_space}")

    from simulator.progress_reporting import SB3ProgressCallback
    callback = SB3ProgressCallback(total_timesteps=timesteps, label=f"pure_nav_{mode}", min_interval_seconds=30.0)

    model.learn(total_timesteps=timesteps, progress_bar=False, callback=callback)
    log(f"Training complete. num_timesteps={model.num_timesteps}")

    out_path = ROOT / "models" / f"pure_navigation_ppo_{mode}.zip"
    out_path.parent.mkdir(exist_ok=True)
    model.save(str(out_path))
    log(f"Saved: {out_path}")
    vec_env.close()


if __name__ == "__main__":
    main()
