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


def run_episode_925(
    curriculum_path: str | Path,
    layout_name: str,
    *,
    net: Any,
    seed: int,
    episode_seconds: float,
    max_actions: int,
    recovery: Any = None,
) -> dict[str, Any]:
    """925-dim-aware counterpart to milestone_evaluator.run_episode.

    milestone_evaluator.run_episode drives iter_variant_environments' raw
    923-value env unwrapped -- correct for the standard production
    contract, but the wrong input width for a SplitSteeringNavigationPolicy
    (which expects the 925-value NavigationHistoryWrapper-augmented
    observation Basic/Beginner training already use; see
    _run_925_episode_raw's docstring for the identical reasoning applied to
    the simpler zero-shot diagnostic). Mirrors run_episode's exact
    tick-by-tick logic (angle tracking, geodesic-vs-euclidean disagreement,
    recovery integration, movement classification) verbatim, changing only
    where the observation comes from: env.reset()/env.step() on the WRAPPED
    env for the 925-value observation fed to the policy, base_env for every
    raw-simulator-API call (player_x/z, heading, map, _visible_candidates,
    _geodesic_field, eva_target_count) -- the same keep-both-references
    pattern basic_environment._roll_basic_episode already established for
    exactly this reason.
    """
    import numpy as np

    from .milestone_evaluator import _is_durable_recovery, _policy_forward
    from .movement_classification import classify_episode_movement
    from .navigation_history import NavigationHistoryWrapper
    from .scripted_policies import scripted_command
    from .synthetic import iter_variant_environments

    entry, base_env = next(iter(iter_variant_environments(
        str(curriculum_path), stage="early", seed=seed, episode_steps=max_actions,
        episode_seconds=episode_seconds, variant_name=layout_name,
    )))
    env = NavigationHistoryWrapper(base_env)
    observation, _ = env.reset(seed=seed)

    steering_choices: list[int] = []
    unique_cells_trace: list[int] = []
    total_distance_trace: list[float] = []
    contacts_trace: list[int] = []
    steering_matches: list[bool] = []
    event_matches: list[bool] = []
    angles: list[float] = []
    left_probs: list[float] = []
    right_probs: list[float] = []
    eva_target_counts: list[int] = []
    teacher_events: list[int] = []
    policy_events: list[int] = []
    geodesic_euclidean_disagreements = 0
    geodesic_euclidean_total = 0
    recovery_kills_during_intervention = 0
    info: dict[str, Any] = {}
    previous_distance = 0.0
    previous_contacts = 0

    for _ in range(int(max_actions)):
        angle = base_env.best_group_relative_angle()
        teacher_command = scripted_command("obstacle_aware", base_env)
        candidates = base_env._visible_candidates()
        if candidates:
            player_cell = base_env.map.native_to_layout_cell(base_env.player_x, base_env.player_z)
            geodesic_field = base_env._geodesic_field(player_cell)
            _potential, best_actor_id = base_env._group_approach_potential(candidates, geodesic_field)
            if best_actor_id is not None:
                geodesic_euclidean_total += 1
                if candidates[0][1].actor_id != best_actor_id:
                    geodesic_euclidean_disagreements += 1

        steering, event, steering_probs = _policy_forward(net, np.asarray(observation, dtype=np.float32))

        was_recovering = recovery is not None and recovery.state.value == "recovering"
        if recovery is not None:
            steering, event = recovery.step(
                tick=len(steering_choices),
                player_x=base_env.player_x, player_z=base_env.player_z, heading=base_env.heading,
                displacement_this_tick=info.get("total_distance_cells", 0.0) - previous_distance if info else 0.0,
                contact_this_tick=bool(info.get("contacts", 0) - previous_contacts) if info else False,
                map_model=base_env.map, policy_steering=steering, policy_event=event,
            )

        steering_choices.append(steering)
        steering_matches.append(steering == int(teacher_command.steering))
        event_matches.append(event == int(teacher_command.event))
        eva_target_counts.append(int(base_env.eva_target_count()))
        teacher_events.append(int(teacher_command.event))
        policy_events.append(event)
        if angle is not None:
            angles.append(angle)
            left_probs.append(float(steering_probs[1]))
            right_probs.append(float(steering_probs[2]))

        kills_before = int(info.get("total_kills", 0)) if info else 0
        previous_distance = float(info.get("total_distance_cells", 0.0)) if info else 0.0
        previous_contacts = int(info.get("contacts", 0)) if info else 0
        observation, _reward, terminated, truncated, info = env.step(np.asarray([steering, event], dtype=np.int64))
        if was_recovering:
            recovery_kills_during_intervention += int(info["total_kills"]) - kills_before
        unique_cells_trace.append(int(info["unique_cells"]))
        total_distance_trace.append(float(info["total_distance_cells"]))
        contacts_trace.append(int(info["contacts"]))
        if terminated or truncated:
            break

    env.close()

    movement = classify_episode_movement(
        steering_choices=steering_choices, unique_cells_trace=unique_cells_trace, total_distance_trace=total_distance_trace,
    )

    recovery_summary: dict[str, Any] | None = None
    if recovery is not None:
        durable_count = sum(
            1 for r in recovery.interventions
            if r.outcome == "recovered" and _is_durable_recovery(
                r.end_tick, unique_cells_trace=unique_cells_trace, contacts_trace=contacts_trace,
                total_distance_trace=total_distance_trace,
            )
        )
        recovery_summary = {
            "final_state": recovery.state.value,
            "intervention_count": len(recovery.interventions),
            "recovered_count": sum(1 for r in recovery.interventions if r.outcome == "recovered"),
            "durably_recovered_count": durable_count,
            "gave_up_count": sum(1 for r in recovery.interventions if r.outcome == "gave_up"),
            "intervention_durations": [
                (r.end_tick - r.trigger_tick) for r in recovery.interventions if r.end_tick is not None
            ],
            "kills_during_intervention": int(recovery_kills_during_intervention),
        }

    return {
        "layout": layout_name, "seed": int(seed), "steps": len(steering_choices),
        "kills_per_simulated_hour": float(info.get("total_kills", 0)) * 3600.0 / max(1e-9, float(info.get("elapsed_seconds", 0.0))),
        "total_kills": int(info.get("total_kills", 0)),
        "valid_eva_casts": int(info.get("valid_eva_casts", 0)),
        "invalid_eva_attempts": int(info.get("invalid_eva_attempts", 0)),
        "missed_eva_opportunities": int(info.get("missed_eva_opportunities", 0)),
        "contacts": int(info.get("contacts", 0)),
        "contacts_per_100_distance": float(info.get("contacts", 0)) * 100.0 / max(1e-9, float(info.get("total_distance_cells", 0.0))),
        "total_distance_cells": float(info.get("total_distance_cells", 0.0)),
        "unique_cells": int(info.get("unique_cells", 0)),
        "path_efficiency": float(info.get("path_efficiency", 0.0)),
        "steering_agreement": float(np.mean(steering_matches)) if steering_matches else None,
        "event_agreement": float(np.mean(event_matches)) if event_matches else None,
        "corr_angle_p_left": float(np.corrcoef(angles, left_probs)[0, 1]) if len(angles) > 5 else None,
        "corr_angle_p_right": float(np.corrcoef(angles, right_probs)[0, 1]) if len(angles) > 5 else None,
        "geodesic_euclidean_disagreement_rate": (
            geodesic_euclidean_disagreements / geodesic_euclidean_total if geodesic_euclidean_total else None
        ),
        "zero_kill": bool(info.get("total_kills", 0) == 0),
        "recovery": recovery_summary,
        **movement,
        "_eva_target_counts": eva_target_counts,
        "_teacher_events": teacher_events,
        "_policy_events": policy_events,
    }


def evaluate_heldout_925(
    model: Any, manifest: Any, *, seeds: list[int], episode_seconds: float, max_actions: int, use_recovery: bool = False,
) -> dict[str, Any]:
    """925-dim-aware counterpart to milestone_evaluator.evaluate_heldout --
    see run_episode_925's docstring for why the original cannot be reused
    directly against a SplitSteeringNavigationPolicy checkpoint. Reuses
    milestone_evaluator's own teacher-episode and aggregation logic
    unchanged (neither touches observation width)."""
    import numpy as np

    from .milestone_evaluator import _new_recovery_controller, _run_teacher_episode, _summarize_episodes

    net = model.policy
    per_layout: dict[str, Any] = {}
    for layout_name in manifest.layouts:
        teacher_results = [
            _run_teacher_episode(manifest.curriculum_path, layout_name, seed=seed, episode_seconds=episode_seconds, max_actions=max_actions)
            for seed in seeds
        ]
        teacher_median_kph = float(np.median([r["kills_per_simulated_hour"] for r in teacher_results])) if teacher_results else None
        policy_results = [
            run_episode_925(
                manifest.curriculum_path, layout_name, net=net, seed=seed, episode_seconds=episode_seconds,
                max_actions=max_actions, recovery=(_new_recovery_controller() if use_recovery else None),
            )
            for seed in seeds
        ]
        per_layout[layout_name] = _summarize_episodes(layout_name, policy_results, teacher_median_kph)
    return {"role": "heldout", "stage": manifest.stage, "assisted": use_recovery, "layouts": per_layout}


def evaluate_challenge_925(
    model: Any, manifest: Any, *, family_seeds: list[int], episode_seconds: float, max_actions: int, use_recovery: bool = False,
) -> dict[str, Any]:
    """925-dim-aware counterpart to milestone_evaluator.evaluate_challenge."""
    import numpy as np

    from .milestone_evaluator import _new_recovery_controller, _run_teacher_episode, _summarize_episodes

    net = model.policy
    fixed_results: dict[str, Any] = {}
    for scenario in manifest.fixed_regression_scenarios:
        teacher = _run_teacher_episode(
            scenario.curriculum_path, scenario.layout, seed=scenario.seed,
            episode_seconds=scenario.episode_seconds, max_actions=scenario.max_actions,
        )
        policy = run_episode_925(
            scenario.curriculum_path, scenario.layout, net=net, seed=scenario.seed,
            episode_seconds=scenario.episode_seconds, max_actions=scenario.max_actions,
            recovery=(_new_recovery_controller() if use_recovery else None),
        )
        policy["teacher_ratio"] = (
            policy["kills_per_simulated_hour"] / teacher["kills_per_simulated_hour"] if teacher["kills_per_simulated_hour"] else None
        )
        policy["expected_failure_signature"] = scenario.expected_failure_signature
        fixed_results[scenario.id] = {k: v for k, v in policy.items() if not k.startswith("_")}

    per_layout: dict[str, Any] = {}
    for layout_name in manifest.challenge_family_layouts:
        teacher_results = [
            _run_teacher_episode(manifest.challenge_family_curriculum_path, layout_name, seed=seed, episode_seconds=episode_seconds, max_actions=max_actions)
            for seed in family_seeds
        ]
        teacher_median_kph = float(np.median([r["kills_per_simulated_hour"] for r in teacher_results])) if teacher_results else None
        policy_results = [
            run_episode_925(
                manifest.challenge_family_curriculum_path, layout_name, net=net, seed=seed, episode_seconds=episode_seconds,
                max_actions=max_actions, recovery=(_new_recovery_controller() if use_recovery else None),
            )
            for seed in family_seeds
        ]
        per_layout[layout_name] = _summarize_episodes(layout_name, policy_results, teacher_median_kph)

    return {
        "role": "challenge", "stage": manifest.stage, "assisted": use_recovery,
        "fixed_regression_scenarios": fixed_results, "challenge_family": per_layout,
    }


_PARALLEL_925_EVAL_NET: Any = None


def _init_925_eval_worker(checkpoint_path: str) -> None:
    global _PARALLEL_925_EVAL_NET
    from stable_baselines3 import PPO
    _PARALLEL_925_EVAL_NET = PPO.load(checkpoint_path, device="cpu").policy


def _heldout_925_task(curriculum_path: str, layout_name: str, seed: int, episode_seconds: float, max_actions: int, use_recovery: bool) -> tuple[str, dict[str, Any], dict[str, Any]]:
    from .milestone_evaluator import _new_recovery_controller, _run_teacher_episode
    teacher = _run_teacher_episode(curriculum_path, layout_name, seed=seed, episode_seconds=episode_seconds, max_actions=max_actions)
    policy = run_episode_925(
        curriculum_path, layout_name, net=_PARALLEL_925_EVAL_NET, seed=seed, episode_seconds=episode_seconds,
        max_actions=max_actions, recovery=(_new_recovery_controller() if use_recovery else None),
    )
    return layout_name, teacher, policy


def evaluate_heldout_925_parallel(
    checkpoint_path: str | Path, manifest: Any, *, seeds: list[int], episode_seconds: float, max_actions: int,
    use_recovery: bool = False, n_workers: int = 4,
) -> dict[str, Any]:
    """Same report as evaluate_heldout_925, computed by farming (teacher,
    policy) episode pairs out across n_workers OS processes -- see
    milestone_evaluator.evaluate_heldout_parallel's docstring for the same
    compute-for-wallclock rationale, needed even more here given this
    project's real evaluation scale (episode_seconds=150, max_actions=1000)."""
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor

    from .milestone_evaluator import _summarize_episodes

    tasks = [
        (manifest.curriculum_path, layout_name, seed, episode_seconds, max_actions, use_recovery)
        for layout_name in manifest.layouts for seed in seeds
    ]
    with ProcessPoolExecutor(
        max_workers=max(1, n_workers), initializer=_init_925_eval_worker, initargs=(str(checkpoint_path),),
    ) as pool:
        flat = list(pool.map(_heldout_925_task, *zip(*tasks))) if tasks else []

    teachers_by_layout: dict[str, list[dict[str, Any]]] = {name: [] for name in manifest.layouts}
    policies_by_layout: dict[str, list[dict[str, Any]]] = {name: [] for name in manifest.layouts}
    for layout_name, teacher, policy in flat:
        teachers_by_layout[layout_name].append(teacher)
        policies_by_layout[layout_name].append(policy)

    per_layout = {}
    for layout_name in manifest.layouts:
        teacher_results = teachers_by_layout[layout_name]
        teacher_median_kph = float(np.median([r["kills_per_simulated_hour"] for r in teacher_results])) if teacher_results else None
        per_layout[layout_name] = _summarize_episodes(layout_name, policies_by_layout[layout_name], teacher_median_kph)
    return {"role": "heldout", "stage": manifest.stage, "assisted": use_recovery, "layouts": per_layout}


def _challenge_925_fixed_task(
    curriculum_path: str, layout: str, seed: int, episode_seconds: float, max_actions: int,
    expected_failure_signature: str, use_recovery: bool,
) -> dict[str, Any]:
    from .milestone_evaluator import _new_recovery_controller, _run_teacher_episode
    teacher = _run_teacher_episode(curriculum_path, layout, seed=seed, episode_seconds=episode_seconds, max_actions=max_actions)
    policy = run_episode_925(
        curriculum_path, layout, net=_PARALLEL_925_EVAL_NET, seed=seed, episode_seconds=episode_seconds,
        max_actions=max_actions, recovery=(_new_recovery_controller() if use_recovery else None),
    )
    policy["teacher_ratio"] = policy["kills_per_simulated_hour"] / teacher["kills_per_simulated_hour"] if teacher["kills_per_simulated_hour"] else None
    policy["expected_failure_signature"] = expected_failure_signature
    return {k: v for k, v in policy.items() if not k.startswith("_")}


def evaluate_challenge_925_parallel(
    checkpoint_path: str | Path, manifest: Any, *, family_seeds: list[int], episode_seconds: float, max_actions: int,
    use_recovery: bool = False, n_workers: int = 4,
) -> dict[str, Any]:
    """Same report as evaluate_challenge_925, parallelized the same way as
    evaluate_heldout_925_parallel."""
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor

    from .milestone_evaluator import _summarize_episodes

    fixed_tasks = [
        (s.curriculum_path, s.layout, s.seed, s.episode_seconds, s.max_actions, s.expected_failure_signature, use_recovery)
        for s in manifest.fixed_regression_scenarios
    ]
    family_tasks = [
        (manifest.challenge_family_curriculum_path, layout_name, seed, episode_seconds, max_actions, use_recovery)
        for layout_name in manifest.challenge_family_layouts for seed in family_seeds
    ]

    fixed_results: dict[str, Any] = {}
    per_layout: dict[str, Any] = {}
    with ProcessPoolExecutor(
        max_workers=max(1, n_workers), initializer=_init_925_eval_worker, initargs=(str(checkpoint_path),),
    ) as pool:
        if fixed_tasks:
            fixed_flat = list(pool.map(_challenge_925_fixed_task, *zip(*fixed_tasks)))
            for scenario, result in zip(manifest.fixed_regression_scenarios, fixed_flat):
                fixed_results[scenario.id] = result
        if family_tasks:
            family_flat = list(pool.map(_heldout_925_task, *zip(*family_tasks)))
            teachers_by_layout: dict[str, list[dict[str, Any]]] = {name: [] for name in manifest.challenge_family_layouts}
            policies_by_layout: dict[str, list[dict[str, Any]]] = {name: [] for name in manifest.challenge_family_layouts}
            for layout_name, teacher, policy in family_flat:
                teachers_by_layout[layout_name].append(teacher)
                policies_by_layout[layout_name].append(policy)
            for layout_name in manifest.challenge_family_layouts:
                teacher_results = teachers_by_layout[layout_name]
                teacher_median_kph = float(np.median([r["kills_per_simulated_hour"] for r in teacher_results])) if teacher_results else None
                per_layout[layout_name] = _summarize_episodes(layout_name, policies_by_layout[layout_name], teacher_median_kph)

    return {
        "role": "challenge", "stage": manifest.stage, "assisted": use_recovery,
        "fixed_regression_scenarios": fixed_results, "challenge_family": per_layout,
    }


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
