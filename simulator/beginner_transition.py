"""Beginner/Intermediate/Advanced PPO continuation + rehearsal, and Basic's
zero-shot raw diagnostic.

Under the completed frozen-navigation-sub-policy + learned-target-selection
architecture (docs/architecture/CURRICULUM_TRAINING_PIPELINE.md section 4/6),
Basic, Beginner, Intermediate, and Advanced all share ONE trainable policy
architecture (`simulator.split_branch_policy.SplitFarmingTargetEventPolicy`:
`MultiDiscrete([TARGET_ACTION_SIZE, len(FarmingEvent)])` over the plain
`Box(RAW_OBSERVATION_SIZE,)` observation). There is therefore no cross-
architecture checkpoint bridge at the Basic -> Beginner boundary (unlike the
retired event-only-checkpoint design this module previously supported):
Beginner continues Basic's own graduated checkpoint directly via `PPO.load`,
exactly the same way Intermediate continues Beginner's and Advanced
continues Intermediate's -- one shared `continue_farming_policy_ppo_chunk`.

Beginner/Intermediate/Advanced never use recovery -- `simulator.navigation_
ppo.balanced_training_vec_env_farming_policy` never wraps its training envs
with `RecoveryController`, so the PPO rollout buffer is always faithful (the
policy's own sampled target+event action is always what gets executed) and
its on-policy assumption is never at risk. See `simulator.basic_environment`'s
module docstring for the full recovery/PPO design rationale this depends on.

A zero-shot raw (recovery-off) rollout right after Basic graduation is a
DIAGNOSTIC starting point for Beginner, never a Basic graduation
requirement -- see `simulator.basic_milestone_evaluator`'s module docstring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def continue_farming_policy_ppo_chunk(
    checkpoint: str | Path,
    output: str | Path,
    *,
    curriculum: str | Path,
    timesteps: int,
    stage: str = "early",
    seed: int = 0,
    episode_seconds: float = 150.0,
    max_actions: int = 1000,
    device: str = "cpu",
    progress_every_seconds: float = 15.0,
    canonical_stage: str = "beginner",
) -> dict[str, Any]:
    """One bounded PPO chunk on an already-`SplitFarmingTargetEventPolicy`-
    shaped checkpoint (Basic's own graduated checkpoint, or a prior round's
    own output) -- recovery off throughout (structural, see module
    docstring), steering externally driven by `simulator.navigation_ppo.
    balanced_training_vec_env_farming_policy`'s `FarmingPolicyWrapper`
    composition, never sampled by or logged from this policy. Used
    identically by Beginner, Intermediate, and Advanced -- only the
    curriculum/checkpoint lineage and `canonical_stage` differ."""

    from .navigation_ppo import resume_ppo_chunk_farming_policy
    from .navigation_subpolicy import farming_policy_architecture_contract
    from .progress_reporting import SB3ProgressCallback
    from .run_provenance import build_run_manifest, write_run_manifest

    conservative_ppo_hparams = {
        "n_steps": 256, "batch_size": 128, "n_epochs": 4, "learning_rate": 5e-5,
        "clip_range": 0.10, "target_kl": 0.015, "gamma": 0.995, "gae_lambda": 0.95, "ent_coef": 0.015,
    }
    result = resume_ppo_chunk_farming_policy(
        checkpoint=checkpoint, curriculum=curriculum, output=output, timesteps=timesteps,
        stage=stage, seed=seed, episode_seconds=episode_seconds, max_actions=max_actions, device=device,
        callback=SB3ProgressCallback(int(timesteps), label=f"{canonical_stage}_ppo", min_interval_seconds=progress_every_seconds),
    )
    manifest = build_run_manifest(
        stage=canonical_stage, milestone="ppo_chunk", seeds=seed,
        config={"timesteps": timesteps, "stage": stage, "episode_seconds": episode_seconds,
                "max_actions": max_actions, **conservative_ppo_hparams},
        curriculum_path=str(curriculum), recovery_config={"enabled": False, "reason": "structural -- see module docstring"},
        architecture_contract=farming_policy_architecture_contract(),
        starting_checkpoint=str(Path(checkpoint).resolve()), output_checkpoint=result["checkpoint_out"],
    )
    write_run_manifest(result["checkpoint_out"], manifest)
    return result


def rehearse_farming_policy_on_basic_data(
    checkpoint: str | Path,
    output: str | Path,
    *,
    basic_dataset_paths: list[str | Path],
    max_epochs: int = 20,
    learning_rate: float = 2.0e-3,
    batch_size: int = 128,
    seed: int = 0,
    canonical_stage: str = "beginner",
) -> dict[str, Any]:
    """Periodic BC rehearsal on Basic-stage data (human bootstrap +
    recovery-assisted DAgger, concatenated) to guard against PPO forgetting
    Basic-stage event/EVA competence -- `simulator.basic_training.
    bootstrap_farming_event_head`, the canonical event-training function for
    `SplitFarmingTargetEventPolicy` (used identically for Basic's own DAgger
    rounds and this rehearsal pass).

    Deliberately event-only, same as the retired event-only-checkpoint
    design's own rehearsal was: the combined dataset mixes human data
    (whose target/steering columns carry no trustworthy signal at all --
    see `basic_training.build_human_bootstrap_dataset`'s docstring) with
    DAgger data (whose target column IS trustworthy) in one pool, with no
    per-sample source tag threaded through to separate them for target
    training -- excluding target selection from rehearsal entirely is the
    conservative choice. PPO itself is expected to (re)learn target-
    selection competence during its own chunks; only event/EVA historically
    needed extra rehearsal support (human-data class imbalance)."""

    from stable_baselines3 import PPO

    from .basic_training import bootstrap_farming_event_head
    from .navigation_subpolicy import farming_policy_architecture_contract
    from .run_provenance import build_run_manifest, write_run_manifest

    combined_path = Path(output).with_suffix(".rehearsal_combined.npz")
    _concatenate_basic_datasets(basic_dataset_paths, combined_path)

    model = PPO.load(str(checkpoint), device="cpu")
    result = bootstrap_farming_event_head(
        model, combined_path, max_epochs=max_epochs, learning_rate=learning_rate, batch_size=batch_size, seed=seed,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))

    manifest = build_run_manifest(
        stage=canonical_stage, milestone="rehearsal", seeds=seed,
        config={"max_epochs": max_epochs, "learning_rate": learning_rate, "batch_size": batch_size,
                "basic_dataset_paths": [str(p) for p in basic_dataset_paths]},
        architecture_contract=farming_policy_architecture_contract(),
        starting_checkpoint=str(Path(checkpoint).resolve()), output_checkpoint=str(output_path.resolve()),
    )
    write_run_manifest(output_path, manifest)
    return result


def _concatenate_basic_datasets(dataset_paths: list[str | Path], output_path: Path) -> Path:
    import numpy as np

    observations, actions, steering_valid, session_index = [], [], [], []
    session_offset = 0
    for path in dataset_paths:
        with np.load(Path(path), allow_pickle=False) as data:
            observations.append(np.asarray(data["observations"], dtype=np.float32))
            actions.append(np.asarray(data["actions"], dtype=np.int64))
            steering_valid.append(np.asarray(data["steering_label_valid"], dtype=np.bool_))
            sessions = np.asarray(data["session_index"], dtype=np.int64)
            session_index.append(sessions + session_offset)
            session_offset += int(sessions.max()) + 1 if len(sessions) else 0
    np.savez_compressed(
        output_path,
        observations=np.concatenate(observations, axis=0),
        actions=np.concatenate(actions, axis=0),
        steering_label_valid=np.concatenate(steering_valid, axis=0),
        session_index=np.concatenate(session_index, axis=0),
    )
    return output_path


def _aggregate_raw_diagnostic(checkpoint: str | Path, per_layout_results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    per_layout: dict[str, Any] = {}
    for layout_name, results in per_layout_results.items():
        per_layout[layout_name] = {
            "n_episodes": len(results),
            "physical_stagnation_episodes": sum(1 for r in results if r["physical_stagnation"]),
            "mean_contacts_per_100_distance": float(
                sum(r["contacts_per_100_distance"] for r in results) / max(1, len(results))
            ),
        }
    return {
        "role": "beginner_starting_point_diagnostic",
        "checkpoint": str(Path(checkpoint).resolve()),
        "per_layout": per_layout,
        "notes": "Diagnostic only -- not a Basic graduation requirement, not a Beginner graduation result.",
    }


def zero_shot_raw_diagnostic(
    checkpoint: str | Path,
    *,
    heldout_manifest_path: str,
    seeds: list[int],
    episode_seconds: float = 150.0,
    max_actions: int = 1000,
) -> dict[str, Any]:
    """Raw-policy (recovery disabled) evaluation of a just-graduated Basic
    checkpoint against the existing immutable held-out pool -- purely
    informational: how much of Beginner's job is already done vs. still to
    learn. Never a pass/fail gate on Basic. Steering comes from
    `FrozenNavigationSteering`, target selection and event come from
    `checkpoint`'s own forward pass (`navigation_subpolicy.
    run_composed_episode` -- the SAME composition Basic's own assisted
    evaluator and every PPO stage's evaluator use, since all four stages
    share one policy architecture). Sequential/in-process -- see
    `zero_shot_raw_diagnostic_parallel` for a multi-process equivalent."""

    from stable_baselines3 import PPO

    from .curriculum_manifests import load_heldout_manifest, resolve_manifest_curriculum_path
    from .navigation_subpolicy import FrozenNavigationSteering, run_composed_episode

    model = PPO.load(str(checkpoint), device="cpu")
    net = model.policy
    navigation_steering = FrozenNavigationSteering.load_frozen(device="cpu")
    manifest = load_heldout_manifest(heldout_manifest_path)
    curriculum_path = str(resolve_manifest_curriculum_path(manifest.curriculum_path))

    per_layout_results: dict[str, list[dict[str, Any]]] = {}
    for layout_name in manifest.layouts:
        per_layout_results[layout_name] = [
            run_composed_episode(
                curriculum_path, layout_name, farming_policy=net, navigation_steering=navigation_steering,
                seed=seed, episode_seconds=episode_seconds, max_actions=max_actions, stage=manifest.stage,
            )
            for seed in seeds
        ]
    return _aggregate_raw_diagnostic(checkpoint, per_layout_results)


_PARALLEL_WORKER_NET: Any = None
_PARALLEL_WORKER_NAVIGATION_STEERING: Any = None


def _init_raw_diagnostic_worker(checkpoint_path: str) -> None:
    global _PARALLEL_WORKER_NET, _PARALLEL_WORKER_NAVIGATION_STEERING
    from stable_baselines3 import PPO
    from .navigation_subpolicy import FrozenNavigationSteering
    _PARALLEL_WORKER_NET = PPO.load(checkpoint_path, device="cpu").policy
    _PARALLEL_WORKER_NAVIGATION_STEERING = FrozenNavigationSteering.load_frozen(device="cpu")


def _raw_diagnostic_worker_task(curriculum_path: str, layout_name: str, seed: int, episode_seconds: float, max_actions: int, stage: str = "early") -> dict[str, Any]:
    from .navigation_subpolicy import run_composed_episode
    return run_composed_episode(
        curriculum_path, layout_name, farming_policy=_PARALLEL_WORKER_NET,
        navigation_steering=_PARALLEL_WORKER_NAVIGATION_STEERING,
        seed=seed, episode_seconds=episode_seconds, max_actions=max_actions, stage=stage,
    )


def zero_shot_raw_diagnostic_parallel(
    checkpoint: str | Path,
    *,
    heldout_manifest_path: str,
    seeds: list[int],
    episode_seconds: float = 150.0,
    max_actions: int = 1000,
    n_workers: int = 4,
) -> dict[str, Any]:
    """Same report as `zero_shot_raw_diagnostic`, computed by farming
    episodes out across `n_workers` OS processes instead of one sequential
    loop -- see `evaluate_basic_milestone_parallel`'s docstring for the
    same compute-for-wallclock tradeoff rationale."""

    from concurrent.futures import ProcessPoolExecutor

    from .curriculum_manifests import load_heldout_manifest, resolve_manifest_curriculum_path

    manifest = load_heldout_manifest(heldout_manifest_path)
    curriculum_path = str(resolve_manifest_curriculum_path(manifest.curriculum_path))
    tasks = [(curriculum_path, layout_name, seed, episode_seconds, max_actions, manifest.stage)
              for layout_name in manifest.layouts for seed in seeds]
    with ProcessPoolExecutor(
        max_workers=max(1, n_workers), initializer=_init_raw_diagnostic_worker, initargs=(str(checkpoint),),
    ) as pool:
        flat_results = list(pool.map(_raw_diagnostic_worker_task, *zip(*tasks)))

    per_layout_results: dict[str, list[dict[str, Any]]] = {name: [] for name in manifest.layouts}
    for task, result in zip(tasks, flat_results):
        per_layout_results[task[1]].append(result)
    return _aggregate_raw_diagnostic(checkpoint, per_layout_results)
