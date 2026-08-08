"""Basic -> Beginner transition: disable recovery, switch to PPO.

Beginner never uses recovery -- balanced_training_vec_env_phase2 (imported
from simulator.navigation_ppo, unchanged) never wraps its training envs with
RecoveryController, so resume_ppo_chunk_phase2's rollout buffer is always
faithful (the policy's own sampled action is always what gets executed) and
its on-policy assumption is never at risk. See simulator.basic_environment's
module docstring for the full recovery/PPO design rationale this depends on.

A zero-shot raw (recovery-off) rollout right after Basic graduation is a
DIAGNOSTIC starting point for Beginner, never a Basic graduation
requirement -- see simulator.basic_milestone_evaluator's module docstring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _raw_policy_forward(net: Any, observation_925: Any) -> tuple[int, int, Any]:
    import numpy as np
    import torch

    with torch.no_grad():
        obs_t = torch.as_tensor(observation_925[None, :], dtype=torch.float32, device=net.device)
        dist = net.get_distribution(obs_t).distribution
        s = dist[0].probs[0].cpu().numpy()
    return int(s.argmax()), int(dist[1].probs[0].cpu().numpy().argmax()), s


def _run_925_episode_raw(curriculum_path: str, layout_name: str, *, net: Any, seed: int, episode_seconds: float, max_actions: int) -> dict[str, Any]:
    """Raw (recovery-off) rollout of a 925-input SplitSteeringNavigationPolicy.

    milestone_evaluator.run_episode cannot be reused directly here: it drives
    iter_variant_environments' raw 923-value env unwrapped, which is correct
    for the standard production contract but would feed this policy the
    wrong input width (see the shape-mismatch this replaced during the
    smoke test). NavigationHistoryWrapper supplies the same 925-value input
    Basic/Beginner training already use, so this is an apples-to-apples
    evaluation of the actual policy being graduated, not a proxy for it.
    """
    import numpy as np

    from .movement_classification import classify_episode_movement
    from .navigation_history import NavigationHistoryWrapper
    from .synthetic import iter_variant_environments

    entry, base_env = next(iter(iter_variant_environments(
        curriculum_path, stage="early", seed=seed, episode_steps=max_actions,
        episode_seconds=episode_seconds, variant_name=layout_name,
    )))
    env = NavigationHistoryWrapper(base_env)
    observation, _ = env.reset(seed=seed)

    steering_choices, unique_cells_trace, total_distance_trace = [], [], []
    info: dict[str, Any] = {}
    for _ in range(int(max_actions)):
        steering, event, _probs = _raw_policy_forward(net, np.asarray(observation, dtype=np.float32))
        steering_choices.append(steering)
        observation, _r, terminated, truncated, info = env.step(np.asarray([steering, event], dtype=np.int64))
        unique_cells_trace.append(int(info["unique_cells"]))
        total_distance_trace.append(float(info["total_distance_cells"]))
        if terminated or truncated:
            break
    env.close()

    movement = classify_episode_movement(
        steering_choices=steering_choices, unique_cells_trace=unique_cells_trace, total_distance_trace=total_distance_trace,
    )
    return {
        "layout": layout_name, "seed": int(seed), "steps": len(steering_choices),
        "total_kills": int(info.get("total_kills", 0)),
        "contacts": int(info.get("contacts", 0)),
        "contacts_per_100_distance": float(info.get("contacts", 0)) * 100.0 / max(1e-9, float(info.get("total_distance_cells", 0.0))),
        "unique_cells": int(info.get("unique_cells", 0)),
        **movement,
    }


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
    """Raw-policy (recovery disabled), Beginner-style evaluation of a
    just-graduated Basic checkpoint, against the existing immutable
    held-out pool. Purely informational: how much of Beginner's job is
    already done vs. still to learn. Never a pass/fail gate on Basic.
    Sequential/in-process -- see `zero_shot_raw_diagnostic_parallel` for a
    multi-process equivalent."""

    from stable_baselines3 import PPO

    from .curriculum_manifests import load_heldout_manifest

    model = PPO.load(str(checkpoint), device="cpu")
    net = model.policy
    manifest = load_heldout_manifest(heldout_manifest_path)

    per_layout_results: dict[str, list[dict[str, Any]]] = {}
    for layout_name in manifest.layouts:
        per_layout_results[layout_name] = [
            _run_925_episode_raw(
                manifest.curriculum_path, layout_name, net=net, seed=seed,
                episode_seconds=episode_seconds, max_actions=max_actions,
            )
            for seed in seeds
        ]
    return _aggregate_raw_diagnostic(checkpoint, per_layout_results)


_PARALLEL_WORKER_NET: Any = None


def _init_raw_diagnostic_worker(checkpoint_path: str) -> None:
    global _PARALLEL_WORKER_NET
    from stable_baselines3 import PPO
    _PARALLEL_WORKER_NET = PPO.load(checkpoint_path, device="cpu").policy


def _raw_diagnostic_worker_task(curriculum_path: str, layout_name: str, seed: int, episode_seconds: float, max_actions: int) -> dict[str, Any]:
    return _run_925_episode_raw(
        curriculum_path, layout_name, net=_PARALLEL_WORKER_NET, seed=seed,
        episode_seconds=episode_seconds, max_actions=max_actions,
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

    from .curriculum_manifests import load_heldout_manifest

    manifest = load_heldout_manifest(heldout_manifest_path)
    tasks = [(manifest.curriculum_path, layout_name, seed, episode_seconds, max_actions)
              for layout_name in manifest.layouts for seed in seeds]
    with ProcessPoolExecutor(
        max_workers=max(1, n_workers), initializer=_init_raw_diagnostic_worker, initargs=(str(checkpoint),),
    ) as pool:
        flat_results = list(pool.map(_raw_diagnostic_worker_task, *zip(*tasks)))

    per_layout_results: dict[str, list[dict[str, Any]]] = {name: [] for name in manifest.layouts}
    for task, result in zip(tasks, flat_results):
        per_layout_results[task[1]].append(result)
    return _aggregate_raw_diagnostic(checkpoint, per_layout_results)


def graduate_basic_to_beginner(
    basic_checkpoint: str | Path,
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
) -> dict[str, Any]:
    """One bounded PPO chunk on the Basic-graduated checkpoint, recovery off
    throughout (structural, not configurable -- see module docstring),
    conservative project-standard hyperparameters (navigation_ppo.
    resume_ppo_chunk_phase2's own defaults, passed through unchanged).
    Mirrors resume_ppo_chunk_phase2's own shape: one bounded chunk, save,
    never loops on its own, no rehearsal folded in -- call
    rehearse_beginner_on_basic_data separately/periodically between chunks
    if forgetting is observed."""

    from .navigation_ppo import resume_ppo_chunk_phase2
    from .progress_reporting import SB3ProgressCallback
    from .run_provenance import build_run_manifest, write_run_manifest

    conservative_ppo_hparams = {
        "n_steps": 256, "batch_size": 128, "n_epochs": 4, "learning_rate": 5e-5,
        "clip_range": 0.10, "target_kl": 0.015, "gamma": 0.995, "gae_lambda": 0.95, "ent_coef": 0.015,
    }
    result = resume_ppo_chunk_phase2(
        checkpoint=basic_checkpoint, curriculum=curriculum, output=output, timesteps=timesteps,
        stage=stage, seed=seed, episode_seconds=episode_seconds, max_actions=max_actions, device=device,
        callback=SB3ProgressCallback(int(timesteps), label="beginner_ppo", min_interval_seconds=progress_every_seconds),
    )
    manifest = build_run_manifest(
        stage="beginner", milestone="ppo_chunk", seeds=seed,
        config={"timesteps": timesteps, "stage": stage, "episode_seconds": episode_seconds,
                "max_actions": max_actions, **conservative_ppo_hparams},
        curriculum_path=str(curriculum), recovery_config={"enabled": False, "reason": "structural -- see module docstring"},
        starting_checkpoint=str(Path(basic_checkpoint).resolve()), output_checkpoint=result["checkpoint_out"],
    )
    write_run_manifest(result["checkpoint_out"], manifest)
    return result


def rehearse_beginner_on_basic_data(
    checkpoint: str | Path,
    output: str | Path,
    *,
    basic_dataset_paths: list[str | Path],
    epochs: int = 2,
    learning_rate: float = 1e-5,
    batch_size: int = 128,
    seed: int = 0,
) -> dict[str, Any]:
    """Periodic BC rehearsal on Basic-stage data (human bootstrap +
    recovery-assisted DAgger, concatenated) to guard against Beginner's PPO
    phase forgetting Basic-stage event/EVA competence. Reuses
    basic_training.bootstrap_policy_from_human_recordings' training loop
    (session-stratified split, masked steering loss) unchanged -- it does
    not care whether "session" boundaries came from real recordings or
    simulator episodes, only that adjacent indices within one session are
    temporally correlated and should not straddle train/val.

    Deliberately event-only (bootstrap_policy_from_human_recordings'
    default train_heads): the combined dataset mixes human data (steering
    not well-correlated with the current representation, see that
    function's module docstring) with DAgger data (steering IS
    well-correlated) in one pool, with no per-sample source tag this
    function threads through -- excluding steering from rehearsal entirely
    is the conservative choice until that's examined properly. The user has
    flagged comparing pre/post-rehearsal navigation metrics before the
    first real Beginner continuation specifically because of this; not
    resolved here, intentionally deferred.
    """

    from stable_baselines3 import PPO

    from .basic_training import bootstrap_policy_from_human_recordings
    from .run_provenance import build_run_manifest, write_run_manifest

    combined_path = Path(output).with_suffix(".rehearsal_combined.npz")
    _concatenate_basic_datasets(basic_dataset_paths, combined_path)

    model = PPO.load(str(checkpoint), device="cpu")
    result = bootstrap_policy_from_human_recordings(
        model, combined_path, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, seed=seed,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))

    manifest = build_run_manifest(
        stage="beginner", milestone="rehearsal", seeds=seed,
        config={"epochs": epochs, "learning_rate": learning_rate, "batch_size": batch_size,
                "basic_dataset_paths": [str(p) for p in basic_dataset_paths]},
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
