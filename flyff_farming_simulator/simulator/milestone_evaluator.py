"""Permanent milestone evaluator: run a checkpoint against the immutable
held-out/challenge/generator-validation manifests (simulator.curriculum_manifests)
and report per-layout and per-role results.

Read-only with respect to training: this module only rolls out a loaded
checkpoint and the scripted teacher, it never calls policy.learn(). Combines:
  - teacher-relative kill-rate ratio (per-episode median, not pooled -- see
    factorized_cli._evaluate_env's same fix);
  - steering/event agreement with the teacher on policy-visited states;
  - angle correlation (does steering probability actually track the true
    target angle);
  - movement classification (steering persistence vs physical stagnation vs
    productive sustained turn -- simulator.movement_classification);
  - density-binned EVA behavior (simulator.dagger_v193);
  - geodesic-vs-euclidean target disagreement, tracked as a diagnostic only,
    never as a pass/fail signal (it's a known property of irregular map
    geometry, not evidence of a defect).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from farming.actions import FarmingEvent
from .curriculum_manifests import ChallengeManifest, GeneratorValidationManifest, HeldoutManifest
from .dagger_v193 import _density_binned_eva_report, _merge_density_binned_eva
from .movement_classification import classify_episode_movement
from .scripted_policies import scripted_command
from .synthetic import iter_variant_environments


def _is_durable_recovery(
    end_tick: int | None,
    *,
    unique_cells_trace: list[int],
    contacts_trace: list[int],
    total_distance_trace: list[float],
    window: int = 50,
    min_unique_cell_growth: int = 10,
    max_contact_growth: int = 5,
) -> bool:
    """A recovery is durable only if progress actually held up afterward --
    not merely that the controller's own short exit-check saw one good
    window. Requires meaningful unique-cell growth and displacement over a
    longer post-recovery window, without contacts continuing to pile up
    (which would indicate the player re-wedged shortly after "recovering").
    """

    if end_tick is None:
        return False
    start = end_tick
    end = min(len(unique_cells_trace) - 1, start + window)
    if end <= start:
        # Recovery happened right at episode end -- nothing left to confirm.
        return False
    unique_growth = unique_cells_trace[end] - unique_cells_trace[start]
    distance_growth = total_distance_trace[end] - total_distance_trace[start]
    contact_growth = contacts_trace[end] - contacts_trace[start]
    return unique_growth >= min_unique_cell_growth and distance_growth > 0 and contact_growth <= max_contact_growth


def _policy_forward(net: Any, observation: np.ndarray) -> tuple[int, int, np.ndarray]:
    with torch.no_grad():
        obs_tensor = torch.as_tensor(observation[None, :], device=net.device)
        distribution = net.get_distribution(obs_tensor).distribution
        steering_probs = distribution[0].probs[0].cpu().numpy()
        event_probs = distribution[1].probs[0].cpu().numpy()
    return int(steering_probs.argmax()), int(event_probs.argmax()), steering_probs


def run_episode(
    curriculum_path: str | Path,
    layout_name: str,
    *,
    net: Any,
    seed: int,
    episode_seconds: float,
    max_actions: int,
    recovery: Any = None,
    stage: str = "early",
) -> dict[str, Any]:
    """Roll the policy through one episode, tracking the teacher's oracle
    decision and the real geometric angle at every visited state.

    ``recovery``, when given a fresh ``RecoveryController`` instance, wraps
    the policy's own action selection with the Phase 1 bounded recovery
    override. Omitting it (the default) reproduces the exact raw-policy
    behavior every training/teacher-data/ordinary-scoring caller already
    relies on -- recovery is never silently active.

    ``stage`` must match the curriculum's own internal stage (e.g.
    "intermediate" for an Intermediate-stage manifest) -- defaulting this
    to "early" unconditionally (as this function originally did) silently
    breaks against any non-early curriculum with "Curriculum contains no
    variants for stage 'early'", confirmed the hard way against the first
    real Intermediate evaluation attempt. Callers with a HeldoutManifest/
    ChallengeManifest should pass manifest.stage, never assume "early".
    """

    entry, env = next(
        iter(
            iter_variant_environments(
                curriculum_path, stage=stage, seed=seed, episode_steps=max_actions,
                episode_seconds=episode_seconds, variant_name=layout_name,
            )
        )
    )
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
        angle = env.best_group_relative_angle()
        teacher_command = scripted_command("obstacle_aware", env)
        candidates = env._visible_candidates()
        if candidates:
            player_cell = env.map.native_to_layout_cell(env.player_x, env.player_z)
            geodesic_field = env._geodesic_field(player_cell)
            _potential, best_actor_id = env._group_approach_potential(candidates, geodesic_field)
            if best_actor_id is not None:
                geodesic_euclidean_total += 1
                if candidates[0][1].actor_id != best_actor_id:
                    geodesic_euclidean_disagreements += 1

        steering, event, steering_probs = _policy_forward(net, np.asarray(observation, dtype=np.float32))

        was_recovering = recovery is not None and recovery.state.value == "recovering"
        if recovery is not None:
            steering, event = recovery.step(
                tick=len(steering_choices),
                player_x=env.player_x,
                player_z=env.player_z,
                heading=env.heading,
                displacement_this_tick=info.get("total_distance_cells", 0.0) - previous_distance if info else 0.0,
                contact_this_tick=bool(info.get("contacts", 0) - previous_contacts) if info else False,
                map_model=env.map,
                policy_steering=steering,
                policy_event=event,
            )

        steering_choices.append(steering)
        steering_matches.append(steering == int(teacher_command.steering))
        event_matches.append(event == int(teacher_command.event))
        eva_target_counts.append(int(env.eva_target_count()))
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
        steering_choices=steering_choices,
        unique_cells_trace=unique_cells_trace,
        total_distance_trace=total_distance_trace,
    )

    recovery_summary: dict[str, Any] | None = None
    if recovery is not None:
        durable_count = sum(
            1
            for r in recovery.interventions
            if r.outcome == "recovered"
            and _is_durable_recovery(
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
        "layout": layout_name,
        "seed": int(seed),
        "steps": len(steering_choices),
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


def _stat(results: list[dict[str, Any]], key: str) -> dict[str, float] | None:
    values = [r[key] for r in results if r.get(key) is not None]
    if not values:
        return None
    return {"median": float(np.median(values)), "min": float(np.min(values)), "max": float(np.max(values))}


def _summarize_episodes(label: str, results: list[dict[str, Any]], teacher_median_kph: float | None) -> dict[str, Any]:
    eva_counts = [c for r in results for c in r["_eva_target_counts"]]
    teacher_events = [e for r in results for e in r["_teacher_events"]]
    policy_events = [e for r in results for e in r["_policy_events"]]
    density_bins = _density_binned_eva_report(
        np.asarray(eva_counts, dtype=np.int64), np.asarray(teacher_events, dtype=np.int64), np.asarray(policy_events, dtype=np.int64)
    )
    kph_stat = _stat(results, "kills_per_simulated_hour")
    return {
        "label": label,
        "n_episodes": len(results),
        "kills_per_simulated_hour": kph_stat,
        "teacher_ratio_median": (kph_stat["median"] / teacher_median_kph) if kph_stat and teacher_median_kph else None,
        "valid_eva_casts": _stat(results, "valid_eva_casts"),
        "invalid_eva_attempts": _stat(results, "invalid_eva_attempts"),
        "missed_eva_opportunities": _stat(results, "missed_eva_opportunities"),
        "contacts_per_100_distance": _stat(results, "contacts_per_100_distance"),
        "unique_cells": _stat(results, "unique_cells"),
        "path_efficiency": _stat(results, "path_efficiency"),
        "steering_agreement": _stat(results, "steering_agreement"),
        "event_agreement": _stat(results, "event_agreement"),
        "corr_angle_p_left": _stat(results, "corr_angle_p_left"),
        "corr_angle_p_right": _stat(results, "corr_angle_p_right"),
        "geodesic_euclidean_disagreement_rate": _stat(results, "geodesic_euclidean_disagreement_rate"),
        "max_consecutive_steering_run": _stat(results, "max_consecutive_steering_run"),
        "steering_persistent_episodes": sum(1 for r in results if r["steering_persistent"]),
        "physical_stagnation_episodes": sum(1 for r in results if r["physical_stagnation"]),
        "productive_sustained_turn_episodes": sum(1 for r in results if r["productive_sustained_turn"]),
        "zero_kill_episodes": sum(1 for r in results if r["zero_kill"]),
        "density_binned_eva": density_bins,
        "recovery": _summarize_recovery(results) if results and results[0].get("recovery") is not None else None,
    }


def _summarize_recovery(results: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [r["recovery"] for r in results if r.get("recovery") is not None]
    all_durations = [d for s in summaries for d in s["intervention_durations"]]
    total_interventions = sum(s["intervention_count"] for s in summaries)
    total_recovered = sum(s["recovered_count"] for s in summaries)
    total_durable = sum(s.get("durably_recovered_count", 0) for s in summaries)
    total_gave_up = sum(s["gave_up_count"] for s in summaries)
    return {
        "episodes_with_intervention": sum(1 for s in summaries if s["intervention_count"] > 0),
        "total_interventions": total_interventions,
        "recovered_count": total_recovered,
        "durably_recovered_count": total_durable,
        "gave_up_count": total_gave_up,
        "immediate_recovery_success_rate": (total_recovered / total_interventions) if total_interventions else None,
        "durable_recovery_success_rate": (total_durable / total_interventions) if total_interventions else None,
        "median_intervention_duration_ticks": float(np.median(all_durations)) if all_durations else None,
        "kills_during_intervention_total": sum(s["kills_during_intervention"] for s in summaries),
        "episodes_ending_given_up": sum(1 for s in summaries if s["final_state"] == "given_up"),
    }


def evaluate_heldout(
    model: Any, manifest: HeldoutManifest, *, seeds: list[int], episode_seconds: float, max_actions: int,
    use_recovery: bool = False,
) -> dict[str, Any]:
    net = model.policy
    per_layout: dict[str, Any] = {}
    for layout_name in manifest.layouts:
        teacher_results = [
            _run_teacher_episode(manifest.curriculum_path, layout_name, seed=seed, episode_seconds=episode_seconds, max_actions=max_actions, stage=manifest.stage)
            for seed in seeds
        ]
        teacher_median_kph = float(np.median([r["kills_per_simulated_hour"] for r in teacher_results]))
        policy_results = [
            run_episode(
                manifest.curriculum_path, layout_name, net=net, seed=seed, episode_seconds=episode_seconds,
                max_actions=max_actions, recovery=(_new_recovery_controller() if use_recovery else None), stage=manifest.stage,
            )
            for seed in seeds
        ]
        per_layout[layout_name] = _summarize_episodes(layout_name, policy_results, teacher_median_kph)
    return {"role": "heldout", "stage": manifest.stage, "assisted": use_recovery, "layouts": per_layout}


def _new_recovery_controller() -> Any:
    from .recovery_controller import RecoveryController

    return RecoveryController()


def evaluate_challenge(
    model: Any, manifest: ChallengeManifest, *, family_seeds: list[int], episode_seconds: float, max_actions: int,
    use_recovery: bool = False,
) -> dict[str, Any]:
    net = model.policy
    fixed_results: dict[str, Any] = {}
    for scenario in manifest.fixed_regression_scenarios:
        teacher = _run_teacher_episode(
            scenario.curriculum_path, scenario.layout, seed=scenario.seed,
            episode_seconds=scenario.episode_seconds, max_actions=scenario.max_actions, stage=manifest.stage,
        )
        policy = run_episode(
            scenario.curriculum_path, scenario.layout, net=net, seed=scenario.seed,
            episode_seconds=scenario.episode_seconds, max_actions=scenario.max_actions,
            recovery=(_new_recovery_controller() if use_recovery else None), stage=manifest.stage,
        )
        policy["teacher_ratio"] = policy["kills_per_simulated_hour"] / teacher["kills_per_simulated_hour"] if teacher["kills_per_simulated_hour"] else None
        policy["expected_failure_signature"] = scenario.expected_failure_signature
        fixed_results[scenario.id] = {k: v for k, v in policy.items() if not k.startswith("_")}

    per_layout: dict[str, Any] = {}
    for layout_name in manifest.challenge_family_layouts:
        teacher_results = [
            _run_teacher_episode(manifest.challenge_family_curriculum_path, layout_name, seed=seed, episode_seconds=episode_seconds, max_actions=max_actions, stage=manifest.stage)
            for seed in family_seeds
        ]
        teacher_median_kph = float(np.median([r["kills_per_simulated_hour"] for r in teacher_results]))
        policy_results = [
            run_episode(
                manifest.challenge_family_curriculum_path, layout_name, net=net, seed=seed, episode_seconds=episode_seconds,
                max_actions=max_actions, recovery=(_new_recovery_controller() if use_recovery else None), stage=manifest.stage,
            )
            for seed in family_seeds
        ]
        per_layout[layout_name] = _summarize_episodes(layout_name, policy_results, teacher_median_kph)

    return {
        "role": "challenge", "stage": manifest.stage, "assisted": use_recovery,
        "fixed_regression_scenarios": fixed_results, "challenge_family": per_layout,
    }


_PARALLEL_EVAL_NET: Any = None


def _init_full_eval_worker(checkpoint_path: str) -> None:
    global _PARALLEL_EVAL_NET
    from stable_baselines3 import PPO
    _PARALLEL_EVAL_NET = PPO.load(checkpoint_path, device="cpu").policy


def _heldout_episode_task(
    curriculum_path: str, layout_name: str, seed: int, episode_seconds: float, max_actions: int, use_recovery: bool,
    stage: str = "early",
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    teacher = _run_teacher_episode(curriculum_path, layout_name, seed=seed, episode_seconds=episode_seconds, max_actions=max_actions, stage=stage)
    policy = run_episode(
        curriculum_path, layout_name, net=_PARALLEL_EVAL_NET, seed=seed, episode_seconds=episode_seconds,
        max_actions=max_actions, recovery=(_new_recovery_controller() if use_recovery else None), stage=stage,
    )
    return layout_name, teacher, policy


def evaluate_heldout_parallel(
    checkpoint_path: str | Path, manifest: HeldoutManifest, *, seeds: list[int], episode_seconds: float,
    max_actions: int, use_recovery: bool = False, n_workers: int = 4,
) -> dict[str, Any]:
    """Same report as `evaluate_heldout`, computed by farming (teacher,
    policy) episode pairs out across `n_workers` OS processes instead of
    one sequential loop -- at this project's real evaluation scale
    (episode_seconds=150, max_actions=1000), a full heldout+challenge pass
    sequential would take hours; see evaluate_basic_milestone_parallel's
    docstring for the same compute-for-wallclock tradeoff rationale.
    Requires a checkpoint on disk (not an in-memory model), since each
    worker process loads its own copy."""

    from concurrent.futures import ProcessPoolExecutor

    tasks = [
        (manifest.curriculum_path, layout_name, seed, episode_seconds, max_actions, use_recovery, manifest.stage)
        for layout_name in manifest.layouts for seed in seeds
    ]
    with ProcessPoolExecutor(
        max_workers=max(1, n_workers), initializer=_init_full_eval_worker, initargs=(str(checkpoint_path),),
    ) as pool:
        flat = list(pool.map(_heldout_episode_task, *zip(*tasks))) if tasks else []

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


def _challenge_fixed_task(
    curriculum_path: str, layout: str, seed: int, episode_seconds: float, max_actions: int,
    expected_failure_signature: str, use_recovery: bool, stage: str = "early",
) -> dict[str, Any]:
    teacher = _run_teacher_episode(curriculum_path, layout, seed=seed, episode_seconds=episode_seconds, max_actions=max_actions, stage=stage)
    policy = run_episode(
        curriculum_path, layout, net=_PARALLEL_EVAL_NET, seed=seed, episode_seconds=episode_seconds,
        max_actions=max_actions, recovery=(_new_recovery_controller() if use_recovery else None), stage=stage,
    )
    policy["teacher_ratio"] = (
        policy["kills_per_simulated_hour"] / teacher["kills_per_simulated_hour"] if teacher["kills_per_simulated_hour"] else None
    )
    policy["expected_failure_signature"] = expected_failure_signature
    return {k: v for k, v in policy.items() if not k.startswith("_")}


def evaluate_challenge_parallel(
    checkpoint_path: str | Path, manifest: ChallengeManifest, *, family_seeds: list[int], episode_seconds: float,
    max_actions: int, use_recovery: bool = False, n_workers: int = 4,
) -> dict[str, Any]:
    """Same report as `evaluate_challenge`, parallelized the same way as
    `evaluate_heldout_parallel` -- see that function's docstring."""

    from concurrent.futures import ProcessPoolExecutor

    fixed_tasks = [
        (s.curriculum_path, s.layout, s.seed, s.episode_seconds, s.max_actions, s.expected_failure_signature, use_recovery, manifest.stage)
        for s in manifest.fixed_regression_scenarios
    ]
    family_tasks = [
        (manifest.challenge_family_curriculum_path, layout_name, seed, episode_seconds, max_actions, use_recovery, manifest.stage)
        for layout_name in manifest.challenge_family_layouts for seed in family_seeds
    ]

    fixed_results: dict[str, Any] = {}
    per_layout: dict[str, Any] = {}
    with ProcessPoolExecutor(
        max_workers=max(1, n_workers), initializer=_init_full_eval_worker, initargs=(str(checkpoint_path),),
    ) as pool:
        if fixed_tasks:
            fixed_flat = list(pool.map(_challenge_fixed_task, *zip(*fixed_tasks)))
            for scenario, result in zip(manifest.fixed_regression_scenarios, fixed_flat):
                fixed_results[scenario.id] = result
        if family_tasks:
            family_flat = list(pool.map(_heldout_episode_task, *zip(*family_tasks)))
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


def _run_teacher_episode(curriculum_path: str, layout_name: str, *, seed: int, episode_seconds: float, max_actions: int, stage: str = "early") -> dict[str, Any]:
    entry, env = next(
        iter(
            iter_variant_environments(
                curriculum_path, stage=stage, seed=seed, episode_steps=max_actions,
                episode_seconds=episode_seconds, variant_name=layout_name,
            )
        )
    )
    observation, _ = env.reset(seed=seed)
    info: dict[str, Any] = {}
    for _ in range(int(max_actions)):
        command = scripted_command("obstacle_aware", env)
        observation, _reward, terminated, truncated, info = env.step(
            np.asarray([int(command.steering), int(command.event)], dtype=np.int64)
        )
        if terminated or truncated:
            break
    env.close()
    return {"kills_per_simulated_hour": float(info.get("total_kills", 0)) * 3600.0 / max(1e-9, float(info.get("elapsed_seconds", 0.0)))}
