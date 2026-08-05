from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from farming.actions import FarmingEvent, SteeringAction

from .factorized_training import (
    _event_sampling_fractions,
    _head_predictions,
    _train_balanced_factorized_heads,
    _training_values,
    factorized_stage_gate,
)

ACTION_CONTRACT_ID = "latched-forward-factorized-steering-event-v1"


def _class_priors(labels: np.ndarray, indices: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    source = np.asarray(indices, dtype=np.int64).reshape(-1)
    counts = np.bincount(values[source], minlength=int(count)).astype(np.float64)
    counts += 1.0e-6
    return counts / counts.sum()


def _sqrt_inverse_class_weights(
    labels: np.ndarray,
    indices: np.ndarray,
    count: int,
    *,
    maximum: float = 3.0,
) -> np.ndarray:
    priors = _class_priors(labels, indices, count)
    weights = np.sqrt(float(np.max(priors)) / priors)
    weights /= float(np.mean(weights))
    return np.clip(weights, 0.35, float(maximum)).astype(np.float32)


def _prediction_diagnostics(
    policy: Any,
    observations: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, Any]:
    import torch

    device = policy.device
    steering_probabilities: list[np.ndarray] = []
    event_probabilities: list[np.ndarray] = []
    policy.eval()
    with torch.no_grad():
        for start in range(0, len(observations), int(batch_size)):
            obs = torch.as_tensor(observations[start : start + int(batch_size)], device=device)
            distribution = policy.get_distribution(obs).distribution
            if not isinstance(distribution, (list, tuple)) or len(distribution) != 2:
                raise ValueError("Policy does not expose two MultiDiscrete categorical heads")
            steering_probabilities.append(distribution[0].probs.cpu().numpy())
            event_probabilities.append(distribution[1].probs.cpu().numpy())

    steering_probs = np.concatenate(steering_probabilities, axis=0)
    event_probs = np.concatenate(event_probabilities, axis=0)
    predictions = np.column_stack(
        (steering_probs.argmax(axis=1), event_probs.argmax(axis=1))
    ).astype(np.int64)
    gate = factorized_stage_gate(labels, predictions)

    def head_report(probabilities: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
        sorted_probs = np.sort(probabilities, axis=1)
        margins = sorted_probs[:, -1] - sorted_probs[:, -2]
        true_priors = np.bincount(truth, minlength=3).astype(np.float64)
        true_priors /= max(1.0, float(true_priors.sum()))
        predicted_priors = np.bincount(probabilities.argmax(axis=1), minlength=3).astype(np.float64)
        predicted_priors /= max(1.0, float(predicted_priors.sum()))
        return {
            "mean_probabilities": probabilities.mean(axis=0).tolist(),
            "deterministic_priors": predicted_priors.tolist(),
            "true_priors": true_priors.tolist(),
            "mean_margin": float(np.mean(margins)),
            "p10_margin": float(np.quantile(margins, 0.10)),
            "p50_margin": float(np.quantile(margins, 0.50)),
            "p90_margin": float(np.quantile(margins, 0.90)),
            "prior_l1": float(np.abs(predicted_priors - true_priors).sum()),
        }

    return {
        "gate": gate,
        "steering": head_report(steering_probs, labels[:, 0]),
        "event": head_report(event_probs, labels[:, 1]),
    }


def _calibration_score(report: dict[str, Any]) -> float:
    gate = report["gate"]
    steering = gate["heads"]["steering"]
    event = gate["heads"]["event"]
    eva = event["per_class"][int(FarmingEvent.CAST_EVA)]
    prior_penalty = float(report["steering"]["prior_l1"] + report["event"]["prior_l1"])
    return float(
        gate["exact_command_accuracy"]
        + 0.5 * steering["accuracy"]
        + 0.75 * event["accuracy"]
        + 0.50 * eva["recall"]
        + 0.25 * eva["precision"]
        - 0.35 * prior_penalty
    )


def _apply_prior_bias_correction(
    policy: Any,
    *,
    steering_target: np.ndarray,
    event_target: np.ndarray,
    event_sampling_fractions: dict[int, float],
    maximum_delta: float = 2.5,
) -> dict[str, Any]:
    """Correct the output-head priors after oversampled recognition training.

    Recognition batches intentionally oversample rare labels.  The standard
    prior correction adds log(target_prior / sampled_prior) to each class logit.
    This preserves learned conditional features while restoring realistic class
    priors before low-rate natural-distribution calibration.
    """

    import torch

    action_net = getattr(policy, "action_net", None)
    bias = getattr(action_net, "bias", None)
    if bias is None or int(bias.numel()) != 6:
        return {"applied": False, "reason": "policy.action_net.bias is not a six-logit vector"}

    steering_source = np.full(3, 1.0 / 3.0, dtype=np.float64)
    event_source = np.asarray(event_target, dtype=np.float64).copy()
    for value, fraction in event_sampling_fractions.items():
        event_source[int(value)] = float(fraction)
    event_source = np.maximum(event_source, 1.0e-6)
    event_source /= event_source.sum()

    def delta(target: np.ndarray, source: np.ndarray) -> np.ndarray:
        raw = np.log(np.maximum(target, 1.0e-6) / np.maximum(source, 1.0e-6))
        raw -= float(np.mean(raw))
        return np.clip(raw, -float(maximum_delta), float(maximum_delta))

    steering_delta = delta(np.asarray(steering_target, dtype=np.float64), steering_source)
    event_delta = delta(np.asarray(event_target, dtype=np.float64), event_source)
    combined = np.concatenate((steering_delta, event_delta)).astype(np.float32)
    with torch.no_grad():
        bias.add_(torch.as_tensor(combined, device=bias.device, dtype=bias.dtype))
    return {
        "applied": True,
        "steering_target": steering_target.tolist(),
        "steering_source": steering_source.tolist(),
        "steering_logit_delta": steering_delta.tolist(),
        "event_target": event_target.tolist(),
        "event_source": event_source.tolist(),
        "event_logit_delta": event_delta.tolist(),
    }


def _train_natural_prior_epoch(
    policy: Any,
    optimizer: Any,
    observations: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    *,
    batch_size: int,
    steering_weights: np.ndarray,
    event_weights: np.ndarray,
    event_loss_scale: float,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    import torch
    import torch.nn.functional as F

    order = rng.permutation(np.asarray(train_indices, dtype=np.int64))
    steering_weight_tensor = torch.as_tensor(steering_weights, device=policy.device)
    event_weight_tensor = torch.as_tensor(event_weights, device=policy.device)
    losses: list[float] = []
    policy.train()
    for start in range(0, len(order), int(batch_size)):
        indices = order[start : start + int(batch_size)]
        obs = torch.as_tensor(observations[indices], device=policy.device)
        targets = torch.as_tensor(labels[indices], device=policy.device)
        categories = policy.get_distribution(obs).distribution
        if not isinstance(categories, (list, tuple)) or len(categories) != 2:
            raise ValueError("PPO policy is not MultiDiscrete([3, 3])")
        steering_loss = F.cross_entropy(
            categories[0].logits,
            targets[:, 0],
            weight=steering_weight_tensor,
        )
        event_loss = F.cross_entropy(
            categories[1].logits,
            targets[:, 1],
            weight=event_weight_tensor,
        )
        loss = steering_loss + float(event_loss_scale) * event_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
    return {
        "updates": int(len(losses)),
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
    }


def _layout_stratified_episode_split(
    episode_index: np.ndarray,
    layout_index: np.ndarray,
    labels: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Split complete episodes while guaranteeing every layout in validation."""

    episodes = np.unique(episode_index)
    rng = np.random.default_rng(seed)
    validation_episodes: set[int] = set()
    for layout in np.unique(layout_index):
        candidates = np.unique(episode_index[layout_index == layout])
        if len(candidates) < 2:
            raise ValueError(f"Layout {int(layout)} needs at least two teacher episodes")
        validation_episodes.add(int(rng.choice(candidates)))

    target_count = max(len(validation_episodes), int(round(len(episodes) * validation_fraction)))
    for episode in rng.permutation(episodes):
        if len(validation_episodes) >= target_count:
            break
        validation_episodes.add(int(episode))

    required = (
        (0, int(SteeringAction.STRAIGHT)),
        (0, int(SteeringAction.LEFT)),
        (0, int(SteeringAction.RIGHT)),
        (1, int(FarmingEvent.NONE)),
        (1, int(FarmingEvent.CAST_EVA)),
    )
    for head, value in required:
        current = np.isin(episode_index, tuple(validation_episodes))
        if np.any(labels[current, head] == value):
            continue
        candidates = [
            int(ep)
            for ep in episodes
            if int(ep) not in validation_episodes
            and np.any(labels[episode_index == ep, head] == value)
        ]
        if not candidates:
            raise ValueError(f"No complete teacher episode contains head {head} value {value}")
        validation_episodes.add(candidates[0])

    validation_mask = np.isin(episode_index, tuple(validation_episodes))
    return (
        np.flatnonzero(~validation_mask),
        np.flatnonzero(validation_mask),
        sorted(validation_episodes),
    )


def collect_teacher_dataset_v193(
    curriculum_path: str | Path,
    *,
    stage: str,
    samples: int,
    episode_seconds: float,
    max_actions: int,
    teacher_policy: str,
    seed: int,
    output: str | Path,
) -> dict[str, Any]:
    from .scripted_policies import scripted_command
    from .synthetic import iter_variant_environments

    environments = list(
        iter_variant_environments(
            curriculum_path,
            stage=stage,
            seed=seed,
            episode_steps=max_actions,
            episode_seconds=episode_seconds,
        )
    )
    if not environments:
        raise ValueError("No synthetic variants matched the teacher stage")

    observations: list[np.ndarray] = []
    actions: list[tuple[int, int]] = []
    episode_ids: list[int] = []
    layout_ids: list[int] = []
    episode_id = 0
    try:
        while len(observations) < int(samples):
            for layout_id, (_entry, env) in enumerate(environments):
                observation, _ = env.reset(seed=seed + episode_id * 1009 + layout_id * 37)
                for _ in range(int(max_actions)):
                    command = scripted_command(teacher_policy, env)
                    observations.append(np.asarray(observation, dtype=np.float32).copy())
                    actions.append(command.as_array())
                    episode_ids.append(episode_id)
                    layout_ids.append(layout_id)
                    observation, _, terminated, truncated, _ = env.step(command.as_array())
                    if len(observations) >= int(samples) or terminated or truncated:
                        break
                episode_id += 1
                if len(observations) >= int(samples):
                    break
    finally:
        for _, env in environments:
            env.close()

    obs = np.asarray(observations[:samples], dtype=np.float32)
    labels = np.asarray(actions[:samples], dtype=np.int64)
    episode_index = np.asarray(episode_ids[:samples], dtype=np.int64)
    layout_index = np.asarray(layout_ids[:samples], dtype=np.int64)
    if obs.shape != (int(samples), 923) or labels.shape != (int(samples), 2):
        raise ValueError(f"Unexpected teacher arrays: observations={obs.shape}, actions={labels.shape}")

    train_indices, validation_indices, validation_episodes = _layout_stratified_episode_split(
        episode_index,
        layout_index,
        labels,
        validation_fraction=0.20,
        seed=seed,
    )
    layout_names = np.asarray([entry.name for entry, _env in environments], dtype=str)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        observations=obs,
        actions=labels,
        episode_index=episode_index,
        layout_index=layout_index,
        layout_names=layout_names,
        train_indices=train_indices,
        validation_indices=validation_indices,
        action_contract_id=np.asarray([ACTION_CONTRACT_ID]),
        action_nvec=np.asarray([3, 3], dtype=np.int64),
    )
    return {
        "path": str(path.resolve()),
        "samples": int(len(obs)),
        "episodes": int(len(np.unique(episode_index))),
        "layouts": layout_names.tolist(),
        "layout_sample_counts": {
            str(layout_names[index]): int(np.count_nonzero(layout_index == index))
            for index in range(len(layout_names))
        },
        "validation_episodes": validation_episodes,
        "training_samples": int(len(train_indices)),
        "validation_samples": int(len(validation_indices)),
        "steering_counts": np.bincount(labels[:, 0], minlength=3).tolist(),
        "event_counts": np.bincount(labels[:, 1], minlength=3).tolist(),
    }


def train_teacher_clone_v193(
    model: Any,
    dataset_path: str | Path,
    *,
    recognition_epochs: int = 12,
    recognition_learning_rate: float = 3.0e-4,
    calibration_epochs: int = 8,
    calibration_learning_rate: float = 3.0e-5,
    batch_size: int = 256,
    seed: int = 0,
) -> dict[str, Any]:
    import torch

    with np.load(Path(dataset_path), allow_pickle=False) as data:
        observations = np.asarray(data["observations"], dtype=np.float32)
        labels = np.asarray(data["actions"], dtype=np.int64)
        train_indices = np.asarray(data["train_indices"], dtype=np.int64)
        validation_indices = np.asarray(data["validation_indices"], dtype=np.int64)
        layout_names = np.asarray(data["layout_names"], dtype=str).tolist()
        layout_index = np.asarray(data["layout_index"], dtype=np.int64)

    policy = model.policy
    rng = np.random.default_rng(seed)
    steering_values = _training_values(
        labels[:, 0],
        train_indices,
        required=(0, 1, 2),
    )
    event_values = _training_values(
        labels[:, 1],
        train_indices,
        required=(int(FarmingEvent.NONE), int(FarmingEvent.CAST_EVA)),
        optional=(int(FarmingEvent.JUMP),),
        minimum_optional_support=16,
    )
    event_fractions = _event_sampling_fractions(event_values)

    recognition_optimizer = torch.optim.Adam(
        policy.parameters(), lr=float(recognition_learning_rate)
    )
    recognition = _train_balanced_factorized_heads(
        policy,
        recognition_optimizer,
        observations,
        labels,
        steering_indices=train_indices,
        event_indices=train_indices,
        steering_values=steering_values,
        event_values=event_values,
        epochs=int(recognition_epochs),
        batch_size=int(batch_size),
        event_loss_scale=1.5,
        rng=rng,
        event_target_fractions=event_fractions,
    )
    recognition_validation = _prediction_diagnostics(
        policy,
        observations[validation_indices],
        labels[validation_indices],
        batch_size=batch_size,
    )
    if not recognition_validation["gate"]["passed"]:
        return {
            "passed": False,
            "phase": "recognition",
            "dataset": str(Path(dataset_path).resolve()),
            "recognition": recognition,
            "recognition_validation": recognition_validation,
            "reasons": recognition_validation["gate"]["reasons"],
        }

    steering_target = _class_priors(labels[:, 0], train_indices, 3)
    event_target = _class_priors(labels[:, 1], train_indices, 3)
    bias_correction = _apply_prior_bias_correction(
        policy,
        steering_target=steering_target,
        event_target=event_target,
        event_sampling_fractions=event_fractions,
    )

    steering_weights = _sqrt_inverse_class_weights(labels[:, 0], train_indices, 3)
    event_weights = _sqrt_inverse_class_weights(labels[:, 1], train_indices, 3)
    calibration_optimizer = torch.optim.Adam(
        policy.parameters(), lr=float(calibration_learning_rate)
    )
    calibration_rounds: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_score = float("-inf")
    for epoch in range(1, int(calibration_epochs) + 1):
        training = _train_natural_prior_epoch(
            policy,
            calibration_optimizer,
            observations,
            labels,
            train_indices,
            batch_size=batch_size,
            steering_weights=steering_weights,
            event_weights=event_weights,
            event_loss_scale=1.15,
            rng=rng,
        )
        validation = _prediction_diagnostics(
            policy,
            observations[validation_indices],
            labels[validation_indices],
            batch_size=batch_size,
        )
        score = _calibration_score(validation)
        calibration_rounds.append(
            {"epoch": epoch, "score": score, "training": training, "validation": validation}
        )
        if validation["gate"]["passed"] and score > best_score:
            best_score = score
            best_state = copy.deepcopy(policy.state_dict())

    if best_state is None:
        last = calibration_rounds[-1]["validation"] if calibration_rounds else recognition_validation
        return {
            "passed": False,
            "phase": "calibration",
            "dataset": str(Path(dataset_path).resolve()),
            "recognition": recognition,
            "recognition_validation": recognition_validation,
            "bias_correction": bias_correction,
            "calibration_rounds": calibration_rounds,
            "reasons": last["gate"]["reasons"],
        }

    policy.load_state_dict(best_state)
    final_validation = _prediction_diagnostics(
        policy,
        observations[validation_indices],
        labels[validation_indices],
        batch_size=batch_size,
    )
    per_layout_validation: dict[str, Any] = {}
    for layout_id, layout_name in enumerate(layout_names):
        mask = layout_index[validation_indices] == layout_id
        if not np.any(mask):
            per_layout_validation[str(layout_name)] = {"passed": False, "reason": "no validation samples"}
            continue
        selected = validation_indices[mask]
        per_layout_validation[str(layout_name)] = _prediction_diagnostics(
            policy,
            observations[selected],
            labels[selected],
            batch_size=batch_size,
        )

    return {
        "passed": bool(final_validation["gate"]["passed"]),
        "phase": "complete",
        "dataset": str(Path(dataset_path).resolve()),
        "recognition_epochs": int(recognition_epochs),
        "calibration_epochs": int(calibration_epochs),
        "recognition": recognition,
        "recognition_validation": recognition_validation,
        "bias_correction": bias_correction,
        "natural_steering_priors": steering_target.tolist(),
        "natural_event_priors": event_target.tolist(),
        "calibration_steering_weights": steering_weights.tolist(),
        "calibration_event_weights": event_weights.tolist(),
        "calibration_rounds": calibration_rounds,
        "selected_calibration_score": float(best_score),
        "validation": final_validation,
        "per_layout_validation": per_layout_validation,
        "reasons": final_validation["gate"]["reasons"],
    }


def rehearse_factorized_policy_v193(
    model: Any,
    dataset_path: str | Path,
    *,
    recognition_epochs: int = 1,
    calibration_epochs: int = 2,
    learning_rate: float = 2.0e-5,
    batch_size: int = 256,
    seed: int = 0,
) -> dict[str, Any]:
    """Mixed rehearsal: rare-class recognition followed by natural calibration."""

    with np.load(Path(dataset_path), allow_pickle=False) as data:
        observations = np.asarray(data["observations"], dtype=np.float32)
        labels = np.asarray(data["actions"], dtype=np.int64)
        train_indices = np.asarray(data["train_indices"], dtype=np.int64)
        validation_indices = np.asarray(data["validation_indices"], dtype=np.int64)

    policy = model.policy
    optimizer = getattr(policy, "optimizer", None)
    if optimizer is None:
        raise ValueError("PPO policy does not expose its optimiser for rehearsal")
    original_lrs = [float(group.get("lr", learning_rate)) for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)
    try:
        before = _prediction_diagnostics(
            policy, observations[validation_indices], labels[validation_indices], batch_size=batch_size
        )
        steering_values = (0, 1, 2)
        event_values = _training_values(
            labels[:, 1],
            train_indices,
            required=(0, 1),
            optional=(2,),
            minimum_optional_support=16,
        )
        rehearsal_fractions = (
            {0: 0.75, 1: 0.23, 2: 0.02}
            if 2 in event_values
            else {0: 0.75, 1: 0.25}
        )
        recognition = _train_balanced_factorized_heads(
            policy,
            optimizer,
            observations,
            labels,
            steering_indices=train_indices,
            event_indices=train_indices,
            steering_values=steering_values,
            event_values=event_values,
            epochs=int(recognition_epochs),
            batch_size=int(batch_size),
            event_loss_scale=1.35,
            rng=np.random.default_rng(seed),
            event_target_fractions=rehearsal_fractions,
        )
        steering_weights = _sqrt_inverse_class_weights(labels[:, 0], train_indices, 3)
        event_weights = _sqrt_inverse_class_weights(labels[:, 1], train_indices, 3)
        calibration: list[dict[str, Any]] = []
        rng = np.random.default_rng(seed + 1)
        for epoch in range(1, int(calibration_epochs) + 1):
            calibration.append(
                {
                    "epoch": epoch,
                    **_train_natural_prior_epoch(
                        policy,
                        optimizer,
                        observations,
                        labels,
                        train_indices,
                        batch_size=batch_size,
                        steering_weights=steering_weights,
                        event_weights=event_weights,
                        event_loss_scale=1.10,
                        rng=rng,
                    ),
                }
            )
        after = _prediction_diagnostics(
            policy, observations[validation_indices], labels[validation_indices], batch_size=batch_size
        )
    finally:
        for group, original_lr in zip(optimizer.param_groups, original_lrs, strict=True):
            group["lr"] = original_lr

    return {
        "passed": bool(after["gate"]["passed"]),
        "dataset": str(Path(dataset_path).resolve()),
        "before": before,
        "recognition": recognition,
        "calibration": calibration,
        "validation": after,
        "reasons": after["gate"]["reasons"],
    }
