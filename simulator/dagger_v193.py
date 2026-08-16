"""DAgger-style rollout diagnostic: roll the learned policy and, at every
visited state, also query the scripted teacher -- the same oracle function
used to generate its training data, callable against arbitrary environment
state without consuming a step. This never changes what the policy does
(its own action drives the rollout); it only records what the teacher would
have done, so the recorded dataset reflects states the CURRENT policy
actually visits rather than the states the scripted teacher visits on its
own. That distinction is the entire point: behavior-cloning covariate shift
means the two visit different states, and only the policy's own trajectory
can reveal where its steering/event decisions diverge from the teacher.

The collected dataset is saved in the same schema
(``collect_teacher_dataset_v193``) the hybrid trainer already consumes, so it
can be used directly as an additional aggregation round without a second
collector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from farming.actions import FarmingEvent

from .scripted_policies import scripted_command

ACTION_CONTRACT_ID = "latched-forward-factorized-steering-event-v1"


def _policy_forward(net: Any, observation: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray]:
    """One forward pass: deterministic action plus both heads' probabilities."""

    import torch

    net.eval()
    with torch.no_grad():
        obs_tensor = torch.as_tensor(observation[None, :], device=net.device)
        distribution = net.get_distribution(obs_tensor).distribution
        if not isinstance(distribution, (list, tuple)) or len(distribution) != 2:
            raise ValueError("Policy does not expose two MultiDiscrete categorical heads")
        steering_probs = distribution[0].probs[0].cpu().numpy()
        event_probs = distribution[1].probs[0].cpu().numpy()
    steering_action = int(np.argmax(steering_probs))
    event_action = int(np.argmax(event_probs))
    return steering_action, event_action, steering_probs, event_probs


_EVA_DENSITY_BINS: tuple[tuple[str, int, int], ...] = (
    ("0", 0, 0),
    ("1-2", 1, 2),
    ("3-5", 3, 5),
    ("6-10", 6, 10),
    ("10+", 11, 10_000),
)


def _density_binned_eva_report(
    eva_target_count: np.ndarray, teacher_event: np.ndarray, policy_event: np.ndarray
) -> dict[str, Any]:
    """EVA behaviour broken down by local target density.

    A single global EVA action-rate target conflates dense and sparse
    situations; a dense region should plausibly produce more frequent or
    higher-value EVA opportunities than scattered monsters, and the policy
    should be allowed -- expected -- to respond differently per bin rather
    than being pushed toward one overall ratio.
    """

    report: dict[str, Any] = {}
    for name, low, high in _EVA_DENSITY_BINS:
        mask = (eva_target_count >= low) & (eva_target_count <= high)
        ticks = int(mask.sum())
        if ticks == 0:
            report[name] = {"ticks": 0}
            continue
        teacher_eva = teacher_event[mask] == int(FarmingEvent.CAST_EVA)
        policy_eva = policy_event[mask] == int(FarmingEvent.CAST_EVA)
        teacher_opportunities = int(teacher_eva.sum())
        policy_eva_ticks = int(policy_eva.sum())
        policy_recall_hits = int(policy_eva[teacher_eva].sum()) if teacher_opportunities else 0
        report[name] = {
            "ticks": ticks,
            "teacher_eva_ticks": teacher_opportunities,
            "policy_eva_ticks": policy_eva_ticks,
            "policy_recall_hit_ticks": policy_recall_hits,
            "teacher_eva_rate": float(np.mean(teacher_eva)),
            "policy_eva_rate": float(np.mean(policy_eva)),
            "teacher_eva_opportunity_ticks": teacher_opportunities,
            "policy_eva_recall_on_teacher_eva_states": (
                float(policy_recall_hits / teacher_opportunities) if teacher_opportunities else None
            ),
        }
    return report


def _merge_density_binned_eva(per_layout_bins: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for name, _low, _high in _EVA_DENSITY_BINS:
        total_ticks = sum(int(bins.get(name, {}).get("ticks", 0)) for bins in per_layout_bins)
        if total_ticks == 0:
            merged[name] = {"ticks": 0}
            continue
        teacher_ticks = sum(int(bins.get(name, {}).get("teacher_eva_ticks", 0)) for bins in per_layout_bins)
        policy_ticks = sum(int(bins.get(name, {}).get("policy_eva_ticks", 0)) for bins in per_layout_bins)
        recall_hits = sum(int(bins.get(name, {}).get("policy_recall_hit_ticks", 0)) for bins in per_layout_bins)
        merged[name] = {
            "ticks": total_ticks,
            "teacher_eva_rate": float(teacher_ticks / total_ticks),
            "policy_eva_rate": float(policy_ticks / total_ticks),
            "teacher_eva_opportunity_ticks": teacher_ticks,
            "policy_eva_recall_on_teacher_eva_states": (
                float(recall_hits / teacher_ticks) if teacher_ticks else None
            ),
        }
    return merged


def collect_policy_rollout_with_teacher_oracle_v193(
    env: Any,
    model: Any,
    *,
    layout_name: str,
    layout_id: int,
    episodes: int,
    max_actions: int,
    seed: int,
    teacher_policy: str = "obstacle_aware",
) -> dict[str, Any]:
    """Roll ``model``'s deterministic policy through ``env`` for one layout.

    Returns ``{"report": {...}, "dataset": {...}}``. ``report`` is a JSON-safe
    summary; ``dataset`` holds the raw per-tick arrays for this layout
    (observations paired with the teacher's oracle label at that exact
    visited state), meant to be concatenated across layouts by the caller.
    """

    net = model.policy
    minimum_targets = int(env.reward_calculator.config.missed_eva_minimum_targets)

    observations: list[np.ndarray] = []
    teacher_actions: list[tuple[int, int]] = []
    episode_ids: list[int] = []
    step_ids: list[int] = []
    steering_run_lengths: list[int] = []

    teacher_steering_all: list[int] = []
    policy_steering_all: list[int] = []
    teacher_event_all: list[int] = []
    policy_event_all: list[int] = []
    eva_target_count_all: list[int] = []
    policy_eva_probability_when_teacher_eva: list[float] = []
    policy_eva_probability_when_teacher_none: list[float] = []
    missed_opportunity_count = 0
    teacher_eva_opportunity_count = 0
    policy_eva_on_teacher_eva_count = 0

    for episode in range(int(episodes)):
        observation, _ = env.reset(seed=seed + episode)
        previous_steering: int | None = None
        run_length = 0
        for step_index in range(int(max_actions)):
            teacher_command = scripted_command(teacher_policy, env)
            teacher_steering = int(teacher_command.steering)
            teacher_event = int(teacher_command.event)

            policy_steering, policy_event, _steering_probs, event_probs = _policy_forward(
                net, np.asarray(observation, dtype=np.float32)
            )

            run_length = run_length + 1 if policy_steering == previous_steering else 1
            previous_steering = policy_steering

            eva_available = bool(env.eva_available())
            eva_target_count = int(env.eva_target_count())
            missed_opportunity = bool(
                policy_event != int(FarmingEvent.CAST_EVA)
                and eva_available
                and eva_target_count >= minimum_targets
            )
            missed_opportunity_count += int(missed_opportunity)

            eva_probability = float(event_probs[int(FarmingEvent.CAST_EVA)])
            if teacher_event == int(FarmingEvent.CAST_EVA):
                teacher_eva_opportunity_count += 1
                policy_eva_probability_when_teacher_eva.append(eva_probability)
                policy_eva_on_teacher_eva_count += int(policy_event == int(FarmingEvent.CAST_EVA))
            else:
                policy_eva_probability_when_teacher_none.append(eva_probability)

            observations.append(np.asarray(observation, dtype=np.float32).copy())
            teacher_actions.append((teacher_steering, teacher_event))
            episode_ids.append(episode)
            step_ids.append(step_index)
            steering_run_lengths.append(run_length)
            teacher_steering_all.append(teacher_steering)
            policy_steering_all.append(policy_steering)
            teacher_event_all.append(teacher_event)
            policy_event_all.append(policy_event)
            eva_target_count_all.append(eva_target_count)

            action = np.asarray([policy_steering, policy_event], dtype=np.int64)
            observation, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break

    teacher_steering_arr = np.asarray(teacher_steering_all, dtype=np.int64)
    policy_steering_arr = np.asarray(policy_steering_all, dtype=np.int64)
    teacher_event_arr = np.asarray(teacher_event_all, dtype=np.int64)
    policy_event_arr = np.asarray(policy_event_all, dtype=np.int64)
    eva_target_count_arr = np.asarray(eva_target_count_all, dtype=np.int64)

    steering_confusion = np.zeros((3, 3), dtype=np.int64)
    if len(teacher_steering_arr):
        np.add.at(steering_confusion, (teacher_steering_arr, policy_steering_arr), 1)

    density_binned_eva = _density_binned_eva_report(
        eva_target_count_arr, teacher_event_arr, policy_event_arr
    )

    report = {
        "layout": layout_name,
        "layout_id": int(layout_id),
        "episodes": int(episodes),
        "visited_states": int(len(teacher_steering_all)),
        "steering_agreement_with_teacher": (
            float(np.mean(teacher_steering_arr == policy_steering_arr))
            if len(teacher_steering_arr)
            else None
        ),
        "event_agreement_with_teacher": (
            float(np.mean(teacher_event_arr == policy_event_arr))
            if len(teacher_event_arr)
            else None
        ),
        # rows = teacher's choice, columns = policy's choice
        "steering_confusion_matrix_teacher_rows_policy_cols": steering_confusion.tolist(),
        "maximum_consecutive_steering_run": int(max(steering_run_lengths, default=0)),
        "missed_eva_opportunity_ticks": int(missed_opportunity_count),
        "teacher_eva_opportunity_ticks": int(teacher_eva_opportunity_count),
        "policy_eva_recall_on_teacher_eva_states": (
            float(policy_eva_on_teacher_eva_count / teacher_eva_opportunity_count)
            if teacher_eva_opportunity_count
            else None
        ),
        "mean_policy_eva_probability_when_teacher_says_eva": (
            float(np.mean(policy_eva_probability_when_teacher_eva))
            if policy_eva_probability_when_teacher_eva
            else None
        ),
        "mean_policy_eva_probability_when_teacher_says_none": (
            float(np.mean(policy_eva_probability_when_teacher_none))
            if policy_eva_probability_when_teacher_none
            else None
        ),
        "density_binned_eva": density_binned_eva,
    }
    dataset = {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(teacher_actions, dtype=np.int64),
        "episode_index": np.asarray(episode_ids, dtype=np.int64),
        "step_index": np.asarray(step_ids, dtype=np.int64),
        "layout_index": np.full(len(observations), int(layout_id), dtype=np.int64),
        "steering_run_length": np.asarray(steering_run_lengths, dtype=np.int64),
        "policy_steering": policy_steering_arr,
        "policy_event": policy_event_arr,
    }
    return {"report": report, "dataset": dataset}


def collect_dagger_diagnostic_v193(
    curriculum: str | Path,
    checkpoint: str | Path,
    *,
    stage: str,
    episodes: int,
    episode_seconds: float,
    max_actions: int,
    seed: int,
    device: str,
    teacher_policy: str,
    dataset_output: str | Path,
) -> dict[str, Any]:
    """Run the rollout-with-teacher-oracle diagnostic across every layout in
    ``stage`` and save a DAgger-ready aggregated teacher-labeled dataset.

    Duplicate-observation detection: identical (or near-identical, within a
    small tolerance) observation vectors that received different teacher
    labels indicate a genuinely ambiguous or conflicting supervision signal
    at that state, distinct from ordinary disagreement -- reported
    separately so it does not get folded into an otherwise-normal-looking
    disagreement rate.
    """

    from stable_baselines3 import PPO

    from .factorized_training import validate_factorized_policy_contract
    from .synthetic import iter_variant_environments

    model = PPO.load(str(checkpoint), device=device)
    validate_factorized_policy_contract(model)

    layout_reports: list[dict[str, Any]] = []
    all_observations: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_episode_index: list[np.ndarray] = []
    all_layout_index: list[np.ndarray] = []
    layout_names: list[str] = []

    for layout_id, (entry, env) in enumerate(
        iter_variant_environments(
            curriculum,
            stage=stage,
            seed=seed,
            episode_steps=max_actions,
            episode_seconds=episode_seconds,
        )
    ):
        layout_names.append(entry.name)
        result = collect_policy_rollout_with_teacher_oracle_v193(
            env,
            model,
            layout_name=entry.name,
            layout_id=layout_id,
            episodes=episodes,
            max_actions=max_actions,
            seed=seed + layout_id * 100,
            teacher_policy=teacher_policy,
        )
        env.close()
        layout_reports.append(result["report"])
        dataset = result["dataset"]
        all_observations.append(dataset["observations"])
        all_actions.append(dataset["actions"])
        all_episode_index.append(dataset["episode_index"] + layout_id * 1_000_000)
        all_layout_index.append(dataset["layout_index"])

    observations = np.concatenate(all_observations, axis=0) if all_observations else np.zeros((0, 923), dtype=np.float32)
    actions = np.concatenate(all_actions, axis=0) if all_actions else np.zeros((0, 2), dtype=np.int64)
    episode_index = np.concatenate(all_episode_index, axis=0) if all_episode_index else np.zeros((0,), dtype=np.int64)
    layout_index = np.concatenate(all_layout_index, axis=0) if all_layout_index else np.zeros((0,), dtype=np.int64)

    duplicate_conflicts = _count_conflicting_duplicate_observations(observations, actions)

    total_visited = sum(int(r["visited_states"]) for r in layout_reports)
    total_missed = sum(int(r["missed_eva_opportunity_ticks"]) for r in layout_reports)
    total_teacher_opportunities = sum(int(r["teacher_eva_opportunity_ticks"]) for r in layout_reports)
    steering_agreements = [
        r["steering_agreement_with_teacher"] for r in layout_reports if r["steering_agreement_with_teacher"] is not None
    ]
    event_agreements = [
        r["event_agreement_with_teacher"] for r in layout_reports if r["event_agreement_with_teacher"] is not None
    ]

    aggregate = {
        "visited_states": int(total_visited),
        "mean_steering_agreement_with_teacher": float(np.mean(steering_agreements)) if steering_agreements else None,
        "mean_event_agreement_with_teacher": float(np.mean(event_agreements)) if event_agreements else None,
        "missed_eva_opportunity_ticks": int(total_missed),
        "teacher_eva_opportunity_ticks": int(total_teacher_opportunities),
        "policy_eva_recall_on_teacher_eva_states": (
            float(
                sum(
                    r["policy_eva_recall_on_teacher_eva_states"] * r["teacher_eva_opportunity_ticks"]
                    for r in layout_reports
                    if r["policy_eva_recall_on_teacher_eva_states"] is not None
                )
                / total_teacher_opportunities
            )
            if total_teacher_opportunities
            else None
        ),
        "conflicting_duplicate_observations": int(duplicate_conflicts),
        "density_binned_eva": _merge_density_binned_eva([r["density_binned_eva"] for r in layout_reports]),
    }

    dataset_path = Path(dataset_output)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dataset_path,
        observations=observations,
        actions=actions,
        episode_index=episode_index,
        layout_index=layout_index,
        layout_names=np.asarray(layout_names, dtype=str),
        action_contract_id=np.asarray([ACTION_CONTRACT_ID]),
        action_nvec=np.asarray([3, 3], dtype=np.int64),
    )

    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "teacher_policy": teacher_policy,
        "stage": stage,
        "episodes_per_layout": int(episodes),
        "dataset": str(dataset_path.resolve()),
        "dataset_samples": int(len(observations)),
        "layouts": layout_reports,
        "aggregate": aggregate,
    }


def steering_geometry_diagnostic_v193(
    curriculum: str | Path,
    checkpoint: str | Path,
    *,
    stage: str,
    seed: int,
    device: str,
    angle_samples: int = 300,
    rollout_max_actions: int = 200,
) -> dict[str, Any]:
    """Permanent regression check for the specific failure the split-branch
    architecture was built to fix: near-zero angle sensitivity and
    episode-long single-direction steering lock. Read-only, no training.

    Reports, per layout:
      * correlation between the true geometric target angle
        (env.best_group_relative_angle(), sampled from fresh resets so it
        covers a wide angle range rather than one narrow trajectory) and the
        policy's LEFT/RIGHT probability -- should be strongly positive for
        LEFT and strongly negative for RIGHT;
      * the longest run of identical steering choices in a short deterministic
        rollout -- a near-total lock (run length close to the rollout length)
        is exactly the symptom that motivated this diagnostic;
      * mirror consistency of the steering branch itself, when the model is a
        SplitSteeringEventPolicy: negating the geometry features' angle
        component must negate the steering branch's own logits' LEFT/RIGHT
        ordering, proving the branch didn't merely fit an incidental
        left/right imbalance in its training data.
    """

    import torch
    from stable_baselines3 import PPO

    from .factorized_training import validate_factorized_policy_contract
    from .geometry_features import GEOMETRY_FEATURE_SIZE
    from .synthetic import iter_variant_environments

    model = PPO.load(str(checkpoint), device=device)
    validate_factorized_policy_contract(model)
    net = model.policy

    has_split_steering_branch = hasattr(net, "mlp_extractor") and hasattr(net.mlp_extractor, "steering_net")

    layout_reports: list[dict[str, Any]] = []
    for layout_id, (entry, env) in enumerate(
        iter_variant_environments(curriculum, stage=stage, seed=seed, episode_steps=1, episode_seconds=5.0)
    ):
        angles: list[float] = []
        left_probs: list[float] = []
        right_probs: list[float] = []
        for i in range(int(angle_samples)):
            observation, _ = env.reset(seed=seed + layout_id * 9013 + i)
            angle = env.best_group_relative_angle()
            if angle is None:
                continue
            _steering, _event, steering_probs, _event_probs = _policy_forward(
                net, np.asarray(observation, dtype=np.float32)
            )
            angles.append(angle)
            left_probs.append(float(steering_probs[1]))
            right_probs.append(float(steering_probs[2]))
        env.close()

        angle_arr = np.asarray(angles)
        left_arr = np.asarray(left_probs)
        right_arr = np.asarray(right_probs)
        correlation = {
            "samples_with_target": int(len(angle_arr)),
            "corr_angle_p_left": float(np.corrcoef(angle_arr, left_arr)[0, 1]) if len(angle_arr) > 5 else None,
            "corr_angle_p_right": float(np.corrcoef(angle_arr, right_arr)[0, 1]) if len(angle_arr) > 5 else None,
        }

        entry, env = next(
            iter(
                iter_variant_environments(
                    curriculum, stage=stage, seed=seed, episode_steps=rollout_max_actions,
                    episode_seconds=30.0, variant_name=entry.name,
                )
            )
        )
        observation, _ = env.reset(seed=seed)
        steering_choices: list[int] = []
        for _ in range(int(rollout_max_actions)):
            action, _state = model.predict(observation, deterministic=True)
            steering_choices.append(int(action[0]))
            observation, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break
        env.close()
        max_run = 1
        run = 1
        for i in range(1, len(steering_choices)):
            run = run + 1 if steering_choices[i] == steering_choices[i - 1] else 1
            max_run = max(max_run, run)

        layout_reports.append(
            {
                "layout": entry.name,
                **correlation,
                "steering_counts": np.bincount(steering_choices, minlength=3).tolist(),
                "rollout_length": len(steering_choices),
                "maximum_consecutive_steering_run": max_run,
            }
        )

    mirror_report: dict[str, Any] | None = None
    if has_split_steering_branch:
        rng = np.random.default_rng(seed)
        geometry = np.zeros((16, GEOMETRY_FEATURE_SIZE), dtype=np.float32)
        angle = rng.uniform(-np.pi, np.pi, size=16)
        geometry[:, 0] = np.sin(angle)
        geometry[:, 1] = np.cos(angle)
        geometry[:, 2] = rng.uniform(0.0, 1.0, size=16)
        geometry[:, 3] = rng.uniform(0.0, 1.0, size=16)
        geometry[:, 4] = rng.uniform(0.0, 1.0, size=16)
        geometry[:, 5] = 1.0
        mirrored = geometry.copy()
        mirrored[:, 0] = -geometry[:, 0]

        with torch.no_grad():
            latent = net.mlp_extractor.steering_net(torch.as_tensor(geometry))
            logits = net.action_net.steering_out(latent).numpy()
            mirrored_latent = net.mlp_extractor.steering_net(torch.as_tensor(mirrored))
            mirrored_logits = net.action_net.steering_out(mirrored_latent).numpy()

        # LEFT (index 1) and RIGHT (index 2) should swap under mirroring;
        # STRAIGHT (index 0) should stay roughly put.
        left_right_swap_error = float(
            np.mean(np.abs(logits[:, 1] - mirrored_logits[:, 2]) + np.abs(logits[:, 2] - mirrored_logits[:, 1]))
        )
        straight_drift = float(np.mean(np.abs(logits[:, 0] - mirrored_logits[:, 0])))
        mirror_report = {
            "samples": 16,
            "left_right_swap_error": left_right_swap_error,
            "straight_logit_drift": straight_drift,
        }

    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "has_split_steering_branch": has_split_steering_branch,
        "layouts": layout_reports,
        "mirror_consistency": mirror_report,
    }


def _count_conflicting_duplicate_observations(
    observations: np.ndarray, actions: np.ndarray, *, tolerance: float = 1.0e-6
) -> int:
    """Count near-identical observations (within ``tolerance``) that were
    given different teacher labels -- a genuine supervision conflict, not
    just two different states the teacher happened to treat differently.
    Restricted to exact float rounding first (cheap hash bucket) since a
    full pairwise distance search is O(n^2) and this dataset can be large;
    two visits that truly round to the same 6 decimals are close enough for
    this to be a meaningful signal rather than a precise duplicate proof.
    """

    if len(observations) < 2:
        return 0
    rounded = np.round(observations / max(tolerance, 1e-12)).astype(np.int64)
    _unique_rows, inverse, counts = np.unique(rounded, axis=0, return_inverse=True, return_counts=True)
    conflicts = 0
    for bucket in np.flatnonzero(counts > 1):
        members = np.flatnonzero(inverse == bucket)
        if len(np.unique(actions[members], axis=0)) > 1:
            conflicts += len(members)
    return int(conflicts)
