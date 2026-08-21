"""2026-08-14: two ISOLATED reward-objective ablations for the
deterministic waypoint orbit pathology, per explicit user instruction,
following the software-only audit in scratchpad_orbit_reward_discount_
audit.py (which found BOTH hypothesized mechanisms real but neither
universal: some orbit cycles have positive periodic-discounted value from
some/all phases under pure gamma=0.99 discounting; the GAE advantage near
truncation is inflated toward positive in a subset of cases by the flat
~12.0-12.45 bootstrap value; together they leave too little per-state
signal to reliably suppress a closed zero-progress loop).

Same task family, seeds (0/2/8), PPO hyperparameters, and 40,960-step
budget as scratchpad_generalized_waypoint_train_calibrated_arc.py (the
frozen baseline, PRESERVED UNCHANGED -- this script writes to
differently-namespaced checkpoints/eval logs, never overwrites it). Fresh
initialization each seed, no transplant.

Two mutually exclusive, independently testable reward modes (select via
CLI arg), implemented as a thin wrapper AROUND the existing, unmodified
StaticWaypointWrapper -- intercepting only the (reward, terminated,
truncated) tuple it already produces, never reimplementing its distance/
success/collision logic:

  timeout_failure_only: reward stream is IDENTICAL to baseline (no living
    cost). The only change: when the horizon is reached without success
    or collision, the environment now reports `terminated=True,
    truncated=False` (a genuine terminal failure) instead of
    `truncated=True` -- so SB3 no longer bootstraps that episode's final
    value (see the verified on_policy_algorithm.py code path in the
    2026-08-14b diagnostic); the ordinary per-tick reward on that last
    tick is untouched.

  living_cost_only: timeout/truncation semantics are UNCHANGED from
    baseline (still bootstrapped). Every tick's reward (ordinary
    progress, and the terminal success/collision payouts alike) has a
    constant LIVING_COST subtracted. LIVING_COST=0.0441/tick is NOT a
    guessed value -- it is scratchpad_orbit_reward_discount_audit.py's
    own derived candidate: the smallest constant that provably makes
    EVERY one of the 84 observed orbit-cycle phase-returns negative
    (0.0294/tick), with an explicit 1.5x safety margin. Measured impact
    on real successful trajectories: 1.5-2.2% of SUCCESS_TERMINAL_
    REWARD=20 over a typical 7-10 tick run.

A third ("both") mode is defined for completeness but, per instruction,
is only run if the two isolated results warrant it.

Usage: python scratchpad_generalized_waypoint_train_reward_ablation.py <timeout_failure_only|living_cost_only|both> [total_timesteps=40000] [checkpoint_interval=10000]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from simulator.navigation_history import NavigationHistoryWrapper
from simulator.split_branch_policy import SplitSteeringNavigationPolicy
from simulator.static_waypoint_env import (
    COLLISION_TERMINAL_REWARD, SUCCESS_RADIUS_CELLS, SUCCESS_TERMINAL_REWARD, StaticWaypointWrapper,
    held_out_eval_specs, make_static_waypoint_env, sample_generalized_spec,
)

SEEDS = [0, 2, 8]
EPISODE_STEPS = 100
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

DISTANCE_RANGE = (8.0, 25.0)
POSITION_OFFSET_RADIUS_CELLS = 10.0
N_EVAL_SPECS = 40
EVAL_SPEC_SEED = 777_000_000  # SAME dev seed as the baseline calibrated-arc run, for direct comparability

# Data-derived (see module docstring) -- NOT a guess.
LIVING_COST = 0.0441

# Discount-consistent potential-based progress shaping, added 2026-08-14
# per explicit user instruction after the 6-seed replication experiment
# showed "both" (true-terminal + living cost) reaches 593/600 on the
# 850M pool but still leaves a low-frequency residual of the same
# near_target_overshoot_limit_cycle in most seeds. Mathematical audit
# (scratchpad_potential_shaping_audit.py) confirmed to floating-point
# precision that replacing the plain progress delta (d_before - d_after)
# with the potential-shaped form (d_before - GAMMA*d_after) -- i.e.
# F(s,s') = gamma*Phi(s') - Phi(s) with Phi(s) = -distance(s) -- makes
# the discounted return of looping ANY cycle forever equal EXACTLY the
# current distance-to-waypoint, independent of phase: this removes the
# phase-dependent POSITIVE-return exploit the plain delta created under
# gamma=0.99 (some orbit phases were profitable under the old formula;
# none can be under this one), without materially changing the reward
# scale on successful trajectories (delta is exactly explained by the
# bounded, well-understood sum(d_after)*(1-gamma) term, verified in the
# audit, not an unbounded change). Uses the SAME GAMMA constant PPO
# trains with (defined below) -- the shaping identity is only exact at
# that same gamma.

VALID_MODES = (
    "timeout_failure_only", "living_cost_only", "both",
    "combined_discount_consistent_progress",
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class RewardAblationWrapper(StaticWaypointWrapper):
    """Wraps (not reimplements) StaticWaypointWrapper's step() output.
    See module docstring for exactly what each mode changes."""

    def __init__(self, env: Any, *, reward_mode: str, **kwargs: Any) -> None:
        super().__init__(env, **kwargs)
        if reward_mode not in VALID_MODES:
            raise ValueError(f"reward_mode must be one of {VALID_MODES}, got {reward_mode!r}")
        self.reward_mode = reward_mode

    def step(self, action):
        # Captured BEFORE super().step() overwrites self._prev_distance with
        # this tick's post-move distance -- this IS the exact distance_before
        # value StaticWaypointWrapper.step() itself uses internally to compute
        # its own `progress = self._prev_distance - distance`.
        distance_before = self._prev_distance
        obs, reward, terminated, truncated, info = super().step(action)
        distance_after = self._prev_distance  # parent already updated this to the new distance

        if self.reward_mode in ("timeout_failure_only", "both", "combined_discount_consistent_progress"):
            if truncated and not terminated:
                terminated, truncated = True, False
        if self.reward_mode == "combined_discount_consistent_progress":
            # Replace the plain progress delta with the potential-shaped form
            # ONLY on ordinary (non-terminal-payout) ticks -- collision/success
            # fixed payouts are left exactly as StaticWaypointWrapper computed
            # them, matching scratchpad_potential_shaping_audit.py's Part B
            # methodology (terminal payout held constant, only progress ticks
            # reformulated).
            is_terminal_payout = reward in (COLLISION_TERMINAL_REWARD, SUCCESS_TERMINAL_REWARD)
            if not is_terminal_payout:
                reward = distance_before - GAMMA * distance_after
        if self.reward_mode in ("living_cost_only", "both", "combined_discount_consistent_progress"):
            reward = reward - LIVING_COST
        return obs, reward, terminated, truncated, info


def make_env_factory(seed_offset: int, reward_mode: str):
    def _make():
        raw_env = make_static_waypoint_env(episode_steps=EPISODE_STEPS, seed=seed_offset)
        rng = np.random.default_rng(seed_offset)
        wrapped = RewardAblationWrapper(
            NavigationHistoryWrapper(raw_env), reward_mode=reward_mode,
            spec_source=lambda: sample_generalized_spec(
                rng, distance_range=DISTANCE_RANGE, position_offset_radius_cells=POSITION_OFFSET_RADIUS_CELLS,
            ),
        )
        return Monitor(wrapped)
    return _make


def eval_held_out(model: PPO, reward_mode: str, *, deterministic: bool) -> dict:
    """Evaluated on the BASELINE reward (StaticWaypointWrapper, unmodified)
    -- outcome/success/collision/timeout are defined identically to the
    frozen baseline's own eval, regardless of which reward the policy was
    TRAINED under, so results are directly comparable across all three
    (baseline, A, B) runs on the same terms."""
    specs = held_out_eval_specs(
        N_EVAL_SPECS, seed=EVAL_SPEC_SEED, distance_range=DISTANCE_RANGE,
        position_offset_radius_cells=POSITION_OFFSET_RADIUS_CELLS,
    )
    steering_counts = {0: 0, 1: 0, 2: 0}
    successes = 0
    collisions = 0
    timeouts = 0
    any_contact_episodes = 0
    steps_to_success: list[int] = []
    final_distances: list[float] = []
    path_efficiencies: list[float] = []
    oscillation_rates: list[float] = []
    reversal_rates: list[float] = []

    for i, spec in enumerate(specs):
        raw_env = make_static_waypoint_env(episode_steps=EPISODE_STEPS, seed=900_000_000 + i)
        env = StaticWaypointWrapper(NavigationHistoryWrapper(raw_env), spec_source=lambda spec=spec: spec)
        obs, _info = env.reset(seed=900_000_000 + i)
        initial_distance = env.initial_distance_cells
        steering_sequence: list[int] = []
        episode_contact = False
        final_distance = initial_distance
        for tick in range(EPISODE_STEPS):
            action, _state = model.predict(obs, deterministic=deterministic)
            steering_counts[int(action[0])] += 1
            steering_sequence.append(int(action[0]))
            obs, reward, term, trunc, info = env.step(action)
            final_distance = env._distance_to_waypoint_cells()
            if int(info.get("contacts", 0)) > 0:
                episode_contact = True
            if term:
                if reward >= 1.0:
                    successes += 1
                    steps_to_success.append(tick + 1)
                    traveled = float(info.get("total_distance_cells", 0.0))
                    required_progress = max(0.0, initial_distance - SUCCESS_RADIUS_CELLS)
                    if traveled > 0 and required_progress > 0:
                        path_efficiencies.append(required_progress / traveled)
                elif reward <= -1.0:
                    collisions += 1
                break
            if trunc:
                timeouts += 1
                break
        else:
            timeouts += 1
        env.close()
        final_distances.append(final_distance)
        if episode_contact:
            any_contact_episodes += 1
        if len(steering_sequence) > 1:
            switches = sum(1 for a, b in zip(steering_sequence, steering_sequence[1:]) if a != b)
            oscillation_rates.append(switches / (len(steering_sequence) - 1))
            reversals = sum(
                1 for a, b in zip(steering_sequence, steering_sequence[1:])
                if (a == 1 and b == 2) or (a == 2 and b == 1)
            )
            reversal_rates.append(reversals / (len(steering_sequence) - 1))

    total_ticks = sum(steering_counts.values())
    return {
        "deterministic": deterministic,
        "n_specs": N_EVAL_SPECS,
        "success_rate": successes / N_EVAL_SPECS,
        "collision_rate": collisions / N_EVAL_SPECS,
        "timeout_rate": timeouts / N_EVAL_SPECS,
        "any_contact_rate": any_contact_episodes / N_EVAL_SPECS,
        "mean_final_distance_cells": float(np.mean(final_distances)) if final_distances else None,
        "mean_steps_to_success": float(np.mean(steps_to_success)) if steps_to_success else None,
        "mean_path_efficiency": float(np.mean(path_efficiencies)) if path_efficiencies else None,
        "mean_oscillation_rate": float(np.mean(oscillation_rates)) if oscillation_rates else None,
        "mean_left_right_reversal_rate": float(np.mean(reversal_rates)) if reversal_rates else None,
        "steering_fractions": {STEERING_NAMES[k]: v / max(1, total_ticks) for k, v in steering_counts.items()},
    }


def run_seed(seed: int, reward_mode: str, total_timesteps: int, checkpoint_interval: int) -> None:
    log(f"=== generalized waypoint GoalNav [{reward_mode}]: seed={seed} total_timesteps={total_timesteps} ===")
    torch.manual_seed(seed)
    factories = [make_env_factory(seed * 1000 + idx, reward_mode) for idx in range(N_ENVS)]
    vec_env = SubprocVecEnv(factories)
    model = PPO(
        SplitSteeringNavigationPolicy, vec_env, policy_kwargs=POLICY_KWARGS,
        seed=seed, device="cpu", n_steps=N_STEPS, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE, clip_range=CLIP_RANGE, gamma=GAMMA, gae_lambda=GAE_LAMBDA,
        ent_coef=ENT_COEF, verbose=0,
    )

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    eval_log_path = ROOT / "evaluations" / f"generalized_waypoint_{reward_mode}_seed{seed}_checkpoint_evals.json"
    all_evals = []

    done = 0
    while done < total_timesteps:
        chunk = min(checkpoint_interval, total_timesteps - done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
        done = int(model.num_timesteps)
        ckpt_path = models_dir / f"generalized_waypoint_{reward_mode}_seed{seed}_{done:07d}.zip"
        model.save(str(ckpt_path))

        results = [eval_held_out(model, reward_mode, deterministic=d) for d in (True, False)]
        all_evals.append({"num_timesteps": done, "results": results})
        eval_log_path.write_text(json.dumps(all_evals, indent=2), encoding="utf-8")

        log(f"[{reward_mode} seed={seed} checkpoint @ {done}] (held-out, n={N_EVAL_SPECS})")
        for r in results:
            mode = "det" if r["deterministic"] else "stoch"
            log(f"    {mode:5s}: success={r['success_rate']:.2f} collision={r['collision_rate']:.2f} "
                f"timeout={r['timeout_rate']:.2f} final_dist={r['mean_final_distance_cells']:.2f} "
                f"steps={r['mean_steps_to_success']} path_eff={r['mean_path_efficiency']} "
                f"osc={r['mean_oscillation_rate']} reversal={r['mean_left_right_reversal_rate']} "
                f"steering={r['steering_fractions']}")

    vec_env.close()
    log(f"=== [{reward_mode}] seed={seed} COMPLETE ===")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in VALID_MODES:
        print(f"Usage: python {Path(__file__).name} <{'|'.join(VALID_MODES)}> [total_timesteps=40000] [checkpoint_interval=10000]")
        sys.exit(1)
    reward_mode = sys.argv[1]
    total_timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else 40_000
    checkpoint_interval = int(sys.argv[3]) if len(sys.argv) > 3 else 10_000
    uses_living_cost = reward_mode in ("living_cost_only", "both", "combined_discount_consistent_progress")
    uses_shaping = reward_mode == "combined_discount_consistent_progress"
    log(f"reward_mode={reward_mode} living_cost={LIVING_COST if uses_living_cost else 'n/a'} "
        f"discount_consistent_progress={uses_shaping} gamma={GAMMA if uses_shaping else 'n/a'}")
    for seed in SEEDS:
        run_seed(seed, reward_mode, total_timesteps, checkpoint_interval)


if __name__ == "__main__":
    main()
