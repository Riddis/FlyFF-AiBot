"""2026-08-14: Beginner Navigation Training Mix, Part 2 -- continuation
training of the frozen, qualified open-waypoint checkpoint
(generalized_waypoint_both_seed2_0051200.zip) against a mixed episode
distribution: open-waypoint rehearsal (preserving the already-qualified
capability) + router-driven single-wall/two-wall obstacle episodes (the
frozen plan_route/select_persistent_waypoint/TargetPersistenceController
select a MOVING waypoint each tick -- see simulator/router_waypoint_env.py).
Router/planner/controller are completely frozen -- this trains ONLY the
policy's ability to physically execute the waypoint sequences that
composition already produces.

Usage: `python scratchpad_beginner_navigation_mix_train.py [extend_by_steps=20480] [checkpoint_interval=5120]`

Per the approved plan:
  - Mixture is EPISODE-weighted (EPISODE_MODE_PROBS), not transition-share
    -- realized per-mode transition-tick fractions are logged for
    transparency only, never used to adjust the probabilities.
  - 3 independent CONTINUATION replicates (distinct RNG streams from the
    SAME frozen starting weights) -- not fresh-init seeds, hence a
    separate numeric namespace (100/102/108) from the original lineage's
    SEEDS=[0,2,8].
  - Every training-worker's mode/spec RNG streams are derived from
    numpy.random.SeedSequence, keyed on (TRAIN_SEED_BASE, continuation_
    seed, worker_rank, stream_name) -- deterministic but distinct per
    worker AND per stream within one worker, so a same-mode resample
    inside the wrapper never perturbs a different stream's later draws.
  - PPO's own RNG is reseeded via model.set_random_seed() once per
    replicate (right after loading the shared frozen checkpoint) --
    verified via a policy-parameter checksum after the first chunk,
    asserted pairwise distinct across all 3 replicates before any
    replicate is allowed to continue further.
  - Checkpoint step accounting is asserted exactly after every chunk.
  - The stop/extend decision past EXTEND_BY_STEPS is NOT automatic --
    see the numerically-defined rule in main()'s closing summary.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from scratchpad_beginner_routing_randomized_walls import sample_randomized_wall_spec
from scratchpad_beginner_routing_two_wall_s_route import S_ROUTE_DIRECTIONS, sample_two_wall_spec
from scratchpad_generalized_waypoint_train_reward_ablation import eval_held_out
from simulator.environment import RecordedFarmingEnv
from simulator.navigation_history import NavigationHistoryWrapper
from simulator.router_waypoint_env import ObstacleEpisodeSpec, RouterMixedWaypointWrapper
from simulator.single_obstacle_env import GAP_SIDES
from simulator.static_waypoint_env import sample_generalized_spec
from tests.helpers.beginner_navigation_mix_harness import DEV_POOL_SPEC_SEED, eval_obstacle_manifest, load_manifest
from tests.helpers.router_qualification_harness import build_multi_wall_world

EpisodeMode = Literal["open", "single_wall", "two_wall"]

STARTING_CHECKPOINT = ROOT / "models" / "generalized_waypoint_both_seed2_0051200.zip"
RESUME_FROM_STEPS = 51_200
DEFAULT_CHECKPOINT_INTERVAL = 5_120        # 5 x (N_STEPS*N_ENVS=256*4=1024)
DEFAULT_EXTEND_BY_STEPS = 20_480           # 20 x 1024 -- first controlled tranche
N_ENVS = 4

EPISODE_MODE_PROBS: dict[EpisodeMode, float] = {"open": 0.50, "single_wall": 0.25, "two_wall": 0.25}
EPISODE_STEPS_BY_MODE: dict[EpisodeMode, int] = {"open": 100, "single_wall": 200, "two_wall": 200}
DISTANCE_RANGE = (8.0, 25.0)
POSITION_OFFSET_RADIUS_CELLS = 10.0

# Continuation replicate RNG streams from the SAME frozen weights -- NOT
# fresh-init SEEDS=[0,2,8], hence a distinct numeric namespace.
CONTINUATION_SEEDS = [100, 102, 108]
TRAIN_SEED_BASE = 20_260_814  # arbitrary but fixed entropy root for the SeedSequence hierarchy

MODEL_TAG = "router_mix"
MODELS_DIR = ROOT / "models"
EVALUATIONS_DIR = ROOT / "evaluations"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# -- per-worker, per-stream RNG separation ----------------------------------

def make_stream_rngs(train_seed_base: int, continuation_seed: int, worker_rank: int) -> dict[str, np.random.Generator]:
    """Deterministic but distinct child RNG streams for one training
    worker: replicate_A/worker_0 != replicate_A/worker_1 != replicate_B/
    worker_0, and within one worker mode_rng != open_rng != single_wall_rng
    != two_wall_rng -- so a same-mode resample inside the wrapper (Part 1)
    never perturbs a different stream's later draws."""
    parent = np.random.SeedSequence([train_seed_base, continuation_seed, worker_rank])
    names = ("mode", "open", "single_wall", "two_wall")
    children = parent.spawn(len(names))
    return {name: np.random.default_rng(seq) for name, seq in zip(names, children)}


# -- env factory --------------------------------------------------------

def make_mixed_env_factory(continuation_seed: int, worker_rank: int):
    def _make():
        streams = make_stream_rngs(TRAIN_SEED_BASE, continuation_seed, worker_rank)
        mode_names: list[EpisodeMode] = list(EPISODE_MODE_PROBS.keys())
        mode_probs = list(EPISODE_MODE_PROBS.values())

        def mode_source() -> EpisodeMode:
            return streams["mode"].choice(mode_names, p=mode_probs)

        def open_spec_source():
            return sample_generalized_spec(
                streams["open"], distance_range=DISTANCE_RANGE, position_offset_radius_cells=POSITION_OFFSET_RADIUS_CELLS,
            )

        def obstacle_spec_source(mode: EpisodeMode) -> ObstacleEpisodeSpec:
            if mode == "single_wall":
                side = streams["single_wall"].choice(GAP_SIDES)
                wall_spec = sample_randomized_wall_spec(streams["single_wall"], gap_side=str(side))
                return ObstacleEpisodeSpec(
                    wall_specs=[wall_spec.obstacle], approach_heading_offset_radians=wall_spec.approach_heading_offset_radians,
                    distance_cells=wall_spec.obstacle.distance_cells,
                )
            direction = streams["two_wall"].choice(S_ROUTE_DIRECTIONS)
            two_wall_spec = sample_two_wall_spec(streams["two_wall"], direction=str(direction))
            return ObstacleEpisodeSpec(
                wall_specs=[two_wall_spec.wall1_obstacle_spec(), two_wall_spec.wall2_obstacle_spec()],
                approach_heading_offset_radians=two_wall_spec.approach_heading_offset_radians,
                distance_cells=two_wall_spec.distance_cells,
            )

        placeholder_map, placeholder_world = build_multi_wall_world([])
        raw_env = RecordedFarmingEnv(placeholder_world, map_model=placeholder_map, episode_steps=200)
        nav_env = NavigationHistoryWrapper(raw_env)
        wrapped = RouterMixedWaypointWrapper(
            nav_env, mode_source=mode_source, open_spec_source=open_spec_source,
            obstacle_spec_source=obstacle_spec_source, world_builder=build_multi_wall_world,
            episode_steps_by_mode=EPISODE_STEPS_BY_MODE,
        )
        return Monitor(wrapped)
    return _make


# -- realized transition-fraction instrumentation (transparency only) ------

class TransitionFractionCallback(BaseCallback):
    """Tallies REALIZED per-mode transition-tick counts across every
    parallel worker, every tick, via the `episode_mode` field
    RouterMixedWaypointWrapper.step() adds to info. Pure logging -- the
    declared EPISODE_MODE_PROBS contract stays episode-weighted; this
    never feeds back into anything."""

    def __init__(self) -> None:
        super().__init__()
        self.counts: dict[str, int] = {"open": 0, "single_wall": 0, "two_wall": 0}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            mode = info.get("episode_mode")
            if mode in self.counts:
                self.counts[mode] += 1
        return True

    def fractions(self) -> dict[str, float]:
        total = sum(self.counts.values())
        if total == 0:
            return {mode: None for mode in self.counts}
        return {mode: count / total for mode, count in self.counts.items()}


# -- checksum / step-accounting helpers ----------------------------------

def checksum_policy_params(model: PPO) -> str:
    hasher = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        hasher.update(name.encode("utf-8"))
        hasher.update(tensor.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


# -- evaluation -----------------------------------------------------------

def eval_checkpoint(model: PPO, dev_manifest: dict) -> dict:
    obstacle_result = eval_obstacle_manifest(model, dev_manifest, episode_seed_base=812_600_000)
    open_regression = eval_held_out(model, MODEL_TAG, deterministic=True)
    return {"obstacle_dev_pool": obstacle_result, "open_regression_777M": open_regression}


def log_eval_summary(seed: int, done: int, eval_result: dict) -> None:
    obs = eval_result["obstacle_dev_pool"]["combined_summary"]
    reg = eval_result["open_regression_777M"]
    log(
        f"[seed={seed} @ {done}] obstacle_dev: success={obs['success_rate']:.3f} "
        f"collision={obs['collision_rate']:.3f} timeout={obs['timeout_rate']:.3f} "
        f"n={eval_result['obstacle_dev_pool']['n_total']} path_eff={obs['mean_path_efficiency']} | "
        f"open_regression: success={reg['success_rate']:.3f} collision={reg['collision_rate']:.3f}"
    )


def save_checkpoint(model: PPO, seed: int, done: int) -> Path:
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / f"generalized_waypoint_{MODEL_TAG}_seed{seed}_{done:07d}.zip"
    model.save(str(path))
    return path


def append_eval_log(seed: int, done: int, eval_result: dict, transition_fractions: dict | None) -> None:
    EVALUATIONS_DIR.mkdir(exist_ok=True)
    log_path = EVALUATIONS_DIR / f"generalized_waypoint_{MODEL_TAG}_seed{seed}_checkpoint_evals.json"
    all_evals = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    all_evals.append({
        "num_timesteps": done, "eval": eval_result,
        "realized_transition_fractions_last_chunk": transition_fractions,
    })
    log_path.write_text(json.dumps(all_evals, indent=2, default=str), encoding="utf-8")


# -- training orchestration ------------------------------------------------
#
# 2026-08-14 MISTAKES.md fix: the original version of this orchestration
# trained each replicate's first 5,120-step chunk, saved, destroyed the
# VecEnv, then later reloaded the checkpoint for each subsequent chunk --
# with a comment claiming the replicate's RNG had "organically diverged"
# and didn't need reseeding again. That was WRONG: PPO.load() restores
# self.seed from the checkpoint file (the ORIGINAL frozen checkpoint's
# seed, 2) and then _setup_model() unconditionally calls
# set_random_seed(self.seed) -- silently resetting the global RNGs back to
# seed=2 on every reload, discarding the earlier set_random_seed(replicate_
# seed) call's effect. The custom mode/spec SeedSequence-derived streams
# had the same problem: rebuilding the env factories on every reload
# restarted them from position 0 instead of continuing. Proven empirically
# before this fix (see MISTAKES.md), not just reasoned about. Fixed by
# keeping ONE model + ONE VecEnv alive for a replicate's ENTIRE training
# span (51,200 -> target_steps), never destroying/reloading mid-replicate.
# The 3 replicates still run sequentially (one at a time, matching this
# investigation's established resource-usage convention), and each one's
# first-chunk checksum is still recorded for the cross-replicate
# divergence check -- just asserted once, after all 3 are fully done,
# since there's no longer a reload boundary that requires pausing.

def run_replicate_continuous(seed: int, checkpoint_interval: int, target_steps: int, dev_manifest: dict) -> str:
    log(f"=== replicate seed={seed}: continuous run, single model+VecEnv, {RESUME_FROM_STEPS} -> {target_steps} ===")
    factories = [make_mixed_env_factory(seed, rank) for rank in range(N_ENVS)]
    vec_env = SubprocVecEnv(factories)
    model = PPO.load(str(STARTING_CHECKPOINT), env=vec_env, device="cpu")
    assert model.num_timesteps == RESUME_FROM_STEPS, (
        f"expected loaded checkpoint at {RESUME_FROM_STEPS} steps, got {model.num_timesteps}"
    )
    model.set_random_seed(seed)  # reseed ONCE, right after loading the SHARED frozen checkpoint -- never again

    first_checksum: str | None = None
    while model.num_timesteps < target_steps:
        prev = model.num_timesteps
        tf_callback = TransitionFractionCallback()
        model.learn(total_timesteps=checkpoint_interval, reset_num_timesteps=False, progress_bar=False, callback=tf_callback)
        done = int(model.num_timesteps)
        assert done == prev + checkpoint_interval, f"step accounting violated: expected {prev + checkpoint_interval}, got {done}"

        if first_checksum is None:
            first_checksum = checksum_policy_params(model)
        save_checkpoint(model, seed, done)
        eval_result = eval_checkpoint(model, dev_manifest)
        transition_fractions = tf_callback.fractions()
        append_eval_log(seed, done, eval_result, transition_fractions=transition_fractions)
        log_eval_summary(seed, done, eval_result)
        log(f"[seed={seed} @ {done}] realized transition fractions (this chunk): {transition_fractions} "
            f"(declared episode-draw contract: {EPISODE_MODE_PROBS})")

    vec_env.close()
    log(f"=== replicate seed={seed}: COMPLETE @ {model.num_timesteps} (continuous, single process/env, "
        f"first-chunk checksum={first_checksum[:12]}...) ===")
    assert first_checksum is not None
    return first_checksum


def main() -> None:
    extend_by_steps = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_EXTEND_BY_STEPS
    checkpoint_interval = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CHECKPOINT_INTERVAL
    target_steps = RESUME_FROM_STEPS + extend_by_steps
    assert extend_by_steps % checkpoint_interval == 0, "extend_by_steps must be an exact multiple of checkpoint_interval"
    assert checkpoint_interval % (256 * N_ENVS) == 0, (
        f"checkpoint_interval must be a multiple of N_STEPS*N_ENVS=1024 -- SB3 collects full "
        f"rollout chunks regardless of the requested total, so a non-multiple would make the "
        f"exact step-accounting assertion spuriously fail"
    )
    assert STARTING_CHECKPOINT.exists(), f"starting checkpoint missing: {STARTING_CHECKPOINT}"

    dev_manifest = load_manifest(EVALUATIONS_DIR / f"router_mix_dev_pool_{DEV_POOL_SPEC_SEED}_manifest.json")

    log(f"=== Beginner Navigation Training Mix (rerun, continuous-process fix): resume_from={RESUME_FROM_STEPS} "
        f"extend_by={extend_by_steps} target={target_steps} checkpoint_interval={checkpoint_interval} "
        f"replicates={CONTINUATION_SEEDS} mode_probs={EPISODE_MODE_PROBS} ===")

    checksums: dict[int, str] = {}
    for seed in CONTINUATION_SEEDS:
        checksums[seed] = run_replicate_continuous(seed, checkpoint_interval, target_steps, dev_manifest)

    distinct = len(set(checksums.values()))
    log(f"replicate checksum divergence check (first-chunk checksums): {distinct}/{len(checksums)} distinct -- {checksums}")
    assert distinct == len(checksums), (
        f"replicate policy-parameter checksums are NOT pairwise distinct after the first chunk "
        f"({checksums}) -- PPO reseeding failed to produce independent streams; stopping rather "
        f"than treating 3 identical runs as 3 independent replicates"
    )

    log(f"=== All {len(CONTINUATION_SEEDS)} replicates trained to {target_steps}, each a single continuous "
        f"process/env/RNG stream. Predeclared stop/extend rule: do NOT continue automatically -- apply it "
        f"against the logged per-checkpoint evals (evaluations/generalized_waypoint_{MODEL_TAG}_seed*_checkpoint_evals.json) "
        f"and Part 4's mechanical selection separately, per the approved plan. ===")


if __name__ == "__main__":
    main()
