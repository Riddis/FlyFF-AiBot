"""Basic-stage milestone evaluator: assisted-mode metrics only.

Deliberately does NOT gate on raw (recovery-off) Beginner-navigation
performance -- that is Beginner's graduation bar, not Basic's (see
simulator.curriculum_stages' module docstring). Running this module's
zero-shot-raw diagnostic (evaluate_basic_milestone's `raw_diagnostic` field)
is informational only: a starting baseline for Beginner, never a pass/fail
condition here.

Basic graduates on stable ASSISTED competence: useful steering/event
learning is happening, recovery is preventing catastrophic dead episodes
without doing all the work, intervention frequency/severity is not
exploding, and failures are not concentrated in one map/seed family.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from navigation.navigation_evidence import CALIBRATED_EXPECTED_CLEAR_PATH_DISPLACEMENT, CALIBRATED_HISTORY_WINDOW

from .basic_environment import _roll_basic_episode
from .progress_reporting import ProgressPrinter


def _stat(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {"median": float(np.median(values)), "min": float(np.min(values)), "max": float(np.max(values))}


def _episode_record(
    model: Any, navigation_steering: Any, curriculum_path: str, layout_name: str, seed: int, *,
    episode_seconds: float, max_actions: int, history_window: int, expected_clear_path_displacement: float,
) -> dict[str, Any]:
    """One episode's contribution to the milestone report -- the unit of
    work `evaluate_basic_milestone`'s sequential loop and its parallel
    ProcessPoolExecutor counterpart both reduce to, so the two paths can
    never silently diverge in what they compute per episode.

    `navigation_steering` (a `simulator.navigation_subpolicy.
    FrozenNavigationSteering`) supplies executed steering, matching Basic's
    training-time rollout exactly -- see `_roll_basic_episode`'s docstring."""
    records, summary = _roll_basic_episode(
        curriculum_path, layout_name, seed=seed, model=model, navigation_steering=navigation_steering,
        episode_seconds=episode_seconds, max_actions=max_actions,
        history_window=history_window, expected_clear_path_displacement=expected_clear_path_displacement,
    )
    steps = max(1, len(records))
    contacts = sum(1 for r in records if r.contact)
    # Reported separately, not combined into one "disagreement" flag: a
    # single combined rate cannot tell an operator whether steering or
    # event is the actual source of teacher disagreement, and this
    # project's own event-head collapse sat behind a flat ~76% combined
    # rate every round without anyone being able to attribute it to either
    # head specifically.
    steering_disagreement = float(np.mean([
        r.teacher_steering != r.policy_steering for r in records
    ])) if records else 0.0
    event_disagreement = float(np.mean([
        r.teacher_event != r.policy_event for r in records
    ])) if records else 0.0
    mean_displacement = float(np.mean([r.displacement for r in records])) if records else 0.0
    return {
        "layout": layout_name, "seed": int(seed),
        "intervention_count": summary["intervention_count"],
        "intervention_ticks_fraction": summary["intervention_ticks"] / steps,
        "contacts_per_step": contacts / steps,
        "steering_disagreement": steering_disagreement,
        "event_disagreement": event_disagreement,
        "mean_displacement": mean_displacement,
        "final_state": summary["final_state"],
    }


def _aggregate_records(layout_names: list[str], episode_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Rebuild the milestone report from a flat, order-independent list of
    `_episode_record` outputs -- shared by the sequential and parallel
    entry points so aggregation logic exists exactly once."""
    by_layout: dict[str, list[dict[str, Any]]] = {name: [] for name in layout_names}
    for rec in episode_records:
        by_layout.setdefault(rec["layout"], []).append(rec)

    per_layout: dict[str, Any] = {}
    all_intervention_counts = [float(r["intervention_count"]) for r in episode_records]
    all_intervention_ticks_fraction = [r["intervention_ticks_fraction"] for r in episode_records]
    all_steering_disagreement_rates = [r["steering_disagreement"] for r in episode_records]
    all_event_disagreement_rates = [r["event_disagreement"] for r in episode_records]
    all_contacts_per_step = [r["contacts_per_step"] for r in episode_records]
    all_mean_displacement = [r["mean_displacement"] for r in episode_records]
    all_final_states = [r["final_state"] for r in episode_records]

    for layout_name in layout_names:
        layout_recs = by_layout.get(layout_name, [])
        per_layout[layout_name] = {
            "n_episodes": len(layout_recs),
            "intervention_count": _stat([float(r["intervention_count"]) for r in layout_recs]),
            "contacts_per_step": _stat([r["contacts_per_step"] for r in layout_recs]),
            "steering_disagreement_rate": _stat([r["steering_disagreement"] for r in layout_recs]),
            "event_disagreement_rate": _stat([r["event_disagreement"] for r in layout_recs]),
        }

    gave_up_fraction = float(np.mean([s == "given_up" for s in all_final_states])) if all_final_states else 0.0
    dominant_layout_share = (
        max(v["intervention_count"]["median"] for v in per_layout.values() if v["intervention_count"])
        / max(1e-9, sum(v["intervention_count"]["median"] for v in per_layout.values() if v["intervention_count"]))
    ) if any(v["intervention_count"] for v in per_layout.values()) else 0.0

    return {
        "role": "basic_milestone",
        "per_layout": per_layout,
        "intervention_count": _stat([float(v) for v in all_intervention_counts]),
        "intervention_ticks_fraction": _stat(all_intervention_ticks_fraction),
        "steering_disagreement_rate": _stat(all_steering_disagreement_rates),
        "event_disagreement_rate": _stat(all_event_disagreement_rates),
        "contacts_per_step": _stat(all_contacts_per_step),
        "mean_displacement_per_tick": _stat(all_mean_displacement),
        "gave_up_episode_fraction": gave_up_fraction,
        "dominant_layout_intervention_share": dominant_layout_share,
        "n_episodes": len(all_intervention_counts),
        "notes": (
            "Assisted-mode metrics only. Basic graduates on these trending "
            "stable/improving across successive milestones (intervention "
            "frequency/severity not exploding, disagreement not pathological, "
            "no single layout dominating interventions, gave_up_episode_fraction "
            "low) -- NOT on reaching zero interventions or matching raw "
            "Beginner-navigation criteria. Run a separate raw (recovery-off) "
            "rollout via milestone_evaluator for a Beginner-starting-point "
            "diagnostic; that number is informational here, never a gate."
        ),
    }


def evaluate_basic_milestone(
    model: Any,
    curriculum_path: str,
    layout_names: list[str],
    *,
    seeds: list[int],
    episode_seconds: float,
    max_actions: int,
    history_window: int = CALIBRATED_HISTORY_WINDOW,
    expected_clear_path_displacement: float = CALIBRATED_EXPECTED_CLEAR_PATH_DISPLACEMENT,
    progress_every_seconds: float = 15.0,
) -> dict[str, Any]:
    """Roll recovery-assisted episodes across layout_names x seeds and
    report the assisted-mode signals Basic graduation should actually look
    at. `layout_names` should be a mix the caller controls -- typically a
    held-out-ish subset of training-side layouts plus a few from every
    template, so no single map/seed family can dominate the result without
    it being visible in `per_layout`. Sequential/in-process -- see
    `evaluate_basic_milestone_parallel` for a multi-process equivalent that
    trades CPU for wall-clock time against a saved checkpoint."""

    from .navigation_subpolicy import FrozenNavigationSteering
    navigation_steering = FrozenNavigationSteering.load_frozen(device="cpu")

    total = len(layout_names) * len(seeds)
    progress = ProgressPrinter(total, label="basic_milestone_eval", min_interval_seconds=progress_every_seconds)
    done = 0
    episode_records = []
    for layout_name in layout_names:
        for seed in seeds:
            episode_records.append(_episode_record(
                model, navigation_steering, curriculum_path, layout_name, seed,
                episode_seconds=episode_seconds, max_actions=max_actions,
                history_window=history_window, expected_clear_path_displacement=expected_clear_path_displacement,
            ))
            done += 1
            progress.update(done, extra=f"layout={layout_name}")
    progress.finish()
    return _aggregate_records(layout_names, episode_records)


_PARALLEL_WORKER_MODEL: Any = None
_PARALLEL_WORKER_NAVIGATION_STEERING: Any = None


def _init_parallel_worker(checkpoint_path: str) -> None:
    global _PARALLEL_WORKER_MODEL, _PARALLEL_WORKER_NAVIGATION_STEERING
    from stable_baselines3 import PPO
    from .navigation_subpolicy import FrozenNavigationSteering
    _PARALLEL_WORKER_MODEL = PPO.load(checkpoint_path, device="cpu")
    _PARALLEL_WORKER_NAVIGATION_STEERING = FrozenNavigationSteering.load_frozen(device="cpu")


def _parallel_worker_task(
    curriculum_path: str, layout_name: str, seed: int, episode_seconds: float, max_actions: int,
    history_window: int, expected_clear_path_displacement: float,
) -> dict[str, Any]:
    return _episode_record(
        _PARALLEL_WORKER_MODEL, _PARALLEL_WORKER_NAVIGATION_STEERING, curriculum_path, layout_name, seed,
        episode_seconds=episode_seconds, max_actions=max_actions,
        history_window=history_window, expected_clear_path_displacement=expected_clear_path_displacement,
    )


def evaluate_basic_milestone_parallel(
    checkpoint_path: str,
    curriculum_path: str,
    layout_names: list[str],
    *,
    seeds: list[int],
    episode_seconds: float,
    max_actions: int,
    history_window: int = CALIBRATED_HISTORY_WINDOW,
    expected_clear_path_displacement: float = CALIBRATED_EXPECTED_CLEAR_PATH_DISPLACEMENT,
    n_workers: int = 4,
) -> dict[str, Any]:
    """Same report as `evaluate_basic_milestone`, computed by farming
    episodes out across `n_workers` OS processes (each loading its own copy
    of `checkpoint_path` once, via `_init_parallel_worker`) instead of one
    sequential loop. Trades CPU (n_workers copies of the model + simulator
    in memory, real but bounded on a machine with cores to spare) for
    wall-clock time -- deliberately, per the project's "prefer wasting
    compute over wallclock time" preference for this kind of embarrassingly
    parallel, independent-episode workload. Requires a checkpoint on disk
    (not an in-memory model) since each worker process loads its own copy;
    use the sequential function when only an in-memory model is available."""

    from concurrent.futures import ProcessPoolExecutor

    tasks = [(curriculum_path, layout_name, seed, episode_seconds, max_actions, history_window, expected_clear_path_displacement)
              for layout_name in layout_names for seed in seeds]
    with ProcessPoolExecutor(
        max_workers=max(1, n_workers), initializer=_init_parallel_worker, initargs=(checkpoint_path,),
    ) as pool:
        episode_records = list(pool.map(_parallel_worker_task, *zip(*tasks)))
    return _aggregate_records(layout_names, episode_records)
