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

from .basic_environment import _roll_basic_episode
from .navigation_history import CALIBRATED_EXPECTED_CLEAR_PATH_DISPLACEMENT, CALIBRATED_HISTORY_WINDOW
from .progress_reporting import ProgressPrinter


def _stat(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {"median": float(np.median(values)), "min": float(np.min(values)), "max": float(np.max(values))}


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
    it being visible in `per_layout`."""

    per_layout: dict[str, Any] = {}
    all_intervention_counts: list[int] = []
    all_intervention_ticks_fraction: list[float] = []
    all_teacher_disagreement_rates: list[float] = []
    all_contacts_per_step: list[float] = []
    all_mean_displacement: list[float] = []
    all_final_states: list[str] = []
    total = len(layout_names) * len(seeds)
    progress = ProgressPrinter(total, label="basic_milestone_eval", min_interval_seconds=progress_every_seconds)
    done = 0

    for layout_name in layout_names:
        layout_intervention_counts: list[int] = []
        layout_contacts_per_step: list[float] = []
        layout_disagreement: list[float] = []
        for seed in seeds:
            records, summary = _roll_basic_episode(
                curriculum_path, layout_name, seed=seed, model=model,
                episode_seconds=episode_seconds, max_actions=max_actions,
                history_window=history_window, expected_clear_path_displacement=expected_clear_path_displacement,
            )
            done += 1
            progress.update(done, extra=f"layout={layout_name}")

            steps = max(1, len(records))
            contacts = sum(1 for r in records if r.contact)
            disagreement = float(np.mean([
                (r.teacher_steering, r.teacher_event) != (r.policy_steering, r.policy_event) for r in records
            ])) if records else 0.0
            mean_displacement = float(np.mean([r.displacement for r in records])) if records else 0.0
            intervention_ticks_fraction = summary["intervention_ticks"] / steps

            layout_intervention_counts.append(summary["intervention_count"])
            layout_contacts_per_step.append(contacts / steps)
            layout_disagreement.append(disagreement)
            all_intervention_counts.append(summary["intervention_count"])
            all_intervention_ticks_fraction.append(intervention_ticks_fraction)
            all_teacher_disagreement_rates.append(disagreement)
            all_contacts_per_step.append(contacts / steps)
            all_mean_displacement.append(mean_displacement)
            all_final_states.append(summary["final_state"])

        per_layout[layout_name] = {
            "n_episodes": len(seeds),
            "intervention_count": _stat([float(v) for v in layout_intervention_counts]),
            "contacts_per_step": _stat(layout_contacts_per_step),
            "teacher_disagreement_rate": _stat(layout_disagreement),
        }
    progress.finish()

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
        "teacher_disagreement_rate": _stat(all_teacher_disagreement_rates),
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
