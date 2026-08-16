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
    steering_valid_mask: np.ndarray | None = None,
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
    gate = factorized_stage_gate(labels, predictions, steering_valid_mask=steering_valid_mask)
    mask = (
        np.ones(len(labels), dtype=np.bool_)
        if steering_valid_mask is None
        else np.asarray(steering_valid_mask, dtype=np.bool_)
    )

    def head_report(probabilities: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
        if probabilities.shape[0] == 0:
            # e.g. an eva_only-role session's own validation slice, once
            # masked to steering-valid rows, can legitimately be empty --
            # this session simply has nothing to say about steering.
            return {
                "mean_probabilities": [0.0, 0.0, 0.0],
                "deterministic_priors": [0.0, 0.0, 0.0],
                "true_priors": [0.0, 0.0, 0.0],
                "mean_margin": 0.0,
                "p10_margin": 0.0,
                "p50_margin": 0.0,
                "p90_margin": 0.0,
                "prior_l1": 0.0,
                "samples": 0,
            }
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
            "samples": int(probabilities.shape[0]),
        }

    return {
        "gate": gate,
        "steering": head_report(steering_probs[mask], labels[mask, 0]),
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
    # A plain factorized head is one Linear(hidden, 6) with a single fused
    # bias; SplitSteeringEventHead (simulator.split_branch_policy) instead
    # holds two independent Linear(*, 3) sub-heads with separate biases,
    # since the steering branch never sees the same latent as the event
    # branch. Both cases apply the identical correction, split across
    # whichever bias tensor(s) actually exist.
    steering_bias = getattr(action_net, "bias", None)
    event_bias = None
    if steering_bias is None or int(steering_bias.numel()) != 6:
        steering_out = getattr(action_net, "steering_out", None)
        event_out = getattr(action_net, "event_out", None)
        steering_bias = getattr(steering_out, "bias", None)
        event_bias = getattr(event_out, "bias", None)
        if (
            steering_bias is None
            or event_bias is None
            or int(steering_bias.numel()) != 3
            or int(event_bias.numel()) != 3
        ):
            return {"applied": False, "reason": "policy.action_net has no recognizable six-logit bias layout"}

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
    with torch.no_grad():
        if event_bias is None:
            combined = np.concatenate((steering_delta, event_delta)).astype(np.float32)
            steering_bias.add_(torch.as_tensor(combined, device=steering_bias.device, dtype=steering_bias.dtype))
        else:
            steering_bias.add_(
                torch.as_tensor(steering_delta.astype(np.float32), device=steering_bias.device, dtype=steering_bias.dtype)
            )
            event_bias.add_(
                torch.as_tensor(event_delta.astype(np.float32), device=event_bias.device, dtype=event_bias.dtype)
            )
    return {
        "applied": True,
        "steering_target": steering_target.tolist(),
        "steering_source": steering_source.tolist(),
        "steering_logit_delta": steering_delta.tolist(),
        "event_target": event_target.tolist(),
        "event_source": event_source.tolist(),
        "event_logit_delta": event_delta.tolist(),
    }


def expand_steering_input_from_checkpoint(old_policy: Any, new_policy: Any) -> None:
    """Transplant a SplitSteeringEventPolicy's weights into a freshly
    constructed SplitSteeringNavigationPolicy (simulator.split_branch_policy)
    so the expanded policy is provably identical to the source checkpoint at
    initialization -- Phase 2's zero-init weight surgery.

    Only steering_net's first Linear layer changes shape (old steering input
    dim -> STEERING_NAVIGATION_FEATURE_SIZE): the old input's columns are
    copied verbatim, the new columns are zero-initialized, bias is copied
    unchanged. Every other layer (remaining steering hidden layers,
    steering_out, all of event_net/vf_net/event_out, value_net) is a pure
    identity copy -- their shapes never change between the two policies.

    Verify with realistic NONZERO new-feature values, not zeros -- an
    all-zero probe would pass even if the new columns were accidentally left
    at their random initialization, since random_weight * 0 == 0 regardless
    of the weight.
    """

    import torch

    old_mlp = old_policy.mlp_extractor
    new_mlp = new_policy.mlp_extractor

    def _copy_linear(old_layer: Any, new_layer: Any) -> None:
        with torch.no_grad():
            new_layer.weight.copy_(old_layer.weight)
            new_layer.bias.copy_(old_layer.bias)

    def _copy_sequential(old_seq: Any, new_seq: Any) -> None:
        for old_layer, new_layer in zip(old_seq, new_seq):
            if isinstance(old_layer, torch.nn.Linear):
                _copy_linear(old_layer, new_layer)

    old_first = old_mlp.steering_net[0]
    new_first = new_mlp.steering_net[0]
    if not isinstance(old_first, torch.nn.Linear) or not isinstance(new_first, torch.nn.Linear):
        raise TypeError("expand_steering_input_from_checkpoint expects steering_net to start with nn.Linear")
    old_in = old_first.in_features
    if new_first.in_features <= old_in:
        raise ValueError(
            f"new_policy's steering input dim ({new_first.in_features}) must exceed "
            f"old_policy's ({old_in}) -- this is an expansion, not a same-shape copy"
        )
    with torch.no_grad():
        new_first.weight.zero_()
        new_first.weight[:, :old_in].copy_(old_first.weight)
        new_first.bias.copy_(old_first.bias)

    _copy_sequential(old_mlp.steering_net[1:], new_mlp.steering_net[1:])
    _copy_sequential(old_mlp.event_net, new_mlp.event_net)
    _copy_sequential(old_mlp.vf_net, new_mlp.vf_net)

    _copy_linear(old_policy.action_net.steering_out, new_policy.action_net.steering_out)
    _copy_linear(old_policy.action_net.event_out, new_policy.action_net.event_out)
    _copy_linear(old_policy.value_net, new_policy.value_net)


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
    steering_valid_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Natural-prior calibration pass.

    ``steering_valid_mask`` (aligned with ``observations``/``labels``, not
    ``train_indices``) excludes rows with no trustworthy steering label --
    e.g. click-to-move human sessions -- from the steering loss term for
    whichever batches contain them, while those same rows still contribute
    their (always-trustworthy) event label to the event loss term. Omitting
    the mask reproduces the original unmasked behaviour exactly.
    """

    import torch
    import torch.nn.functional as F

    order = rng.permutation(np.asarray(train_indices, dtype=np.int64))
    steering_weight_tensor = torch.as_tensor(steering_weights, device=policy.device)
    event_weight_tensor = torch.as_tensor(event_weights, device=policy.device)
    full_mask = (
        None if steering_valid_mask is None else np.asarray(steering_valid_mask, dtype=np.bool_)
    )
    losses: list[float] = []
    policy.train()
    for start in range(0, len(order), int(batch_size)):
        indices = order[start : start + int(batch_size)]
        obs = torch.as_tensor(observations[indices], device=policy.device)
        targets = torch.as_tensor(labels[indices], device=policy.device)
        categories = policy.get_distribution(obs).distribution
        if not isinstance(categories, (list, tuple)) or len(categories) != 2:
            raise ValueError("PPO policy is not MultiDiscrete([3, 3])")
        event_loss = F.cross_entropy(
            categories[1].logits,
            targets[:, 1],
            weight=event_weight_tensor,
        )
        if full_mask is None:
            steering_loss = F.cross_entropy(
                categories[0].logits,
                targets[:, 0],
                weight=steering_weight_tensor,
            )
        else:
            batch_valid = torch.as_tensor(full_mask[indices], device=policy.device)
            if bool(torch.any(batch_valid)):
                steering_loss = F.cross_entropy(
                    categories[0].logits[batch_valid],
                    targets[batch_valid, 0],
                    weight=steering_weight_tensor,
                )
            else:
                steering_loss = torch.zeros((), device=policy.device)
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


def _natural_human_event_target(
    human_labels: np.ndarray,
    human_event_train_indices: np.ndarray,
    human_continuous_session_mask: np.ndarray,
    scripted_event_target: np.ndarray,
) -> np.ndarray:
    """Estimate the "natural" human event prior from continuous, unfiltered
    direct-keyboard play only.

    eva_only-role sessions only ever export the frames where the player
    actually cast (demonstrations.py's ``recognized`` rule), so every one of
    their samples has event=CAST_EVA. That is a real positive recognition
    example, but folding it into a natural-frequency estimate makes EVA look
    far more common in ordinary play than it is. Falls back to the scripted
    dataset's own event prior when no continuous session data is available.
    """

    natural_indices = human_event_train_indices[
        human_continuous_session_mask[human_event_train_indices]
    ]
    if not len(natural_indices):
        return scripted_event_target
    return _class_priors(human_labels[:, 1], natural_indices, 3)


def _human_session_stratified_split(
    session_index: np.ndarray,
    steering_valid: np.ndarray,
    event_valid: np.ndarray,
    labels: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole human sessions, but guarantee validation still
    contains every steering/event class that exists anywhere in the
    label-valid data.

    A naive random session holdout can land entirely on e.g. eva_only-role
    sessions, whose samples are all event=CAST_EVA with no steering-valid or
    event=NONE examples at all -- that would fail the stage gate regardless
    of how well the policy actually performs, exactly the same failure mode
    ``_layout_stratified_episode_split`` already guards against for the
    scripted dataset.
    """

    sessions = np.unique(session_index)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(sessions)
    validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
    validation_sessions: set[int] = {int(s) for s in shuffled[:validation_count]}

    required: list[tuple[np.ndarray, int, int]] = []
    for value in (0, 1, 2):
        if np.any(labels[steering_valid, 0] == value):
            required.append((steering_valid, value, 0))
    for value in (int(FarmingEvent.NONE), int(FarmingEvent.CAST_EVA)):
        if np.any(labels[event_valid & (labels[:, 1] == value), 1] == value):
            required.append((event_valid, value, 1))

    for valid_mask, value, head in required:
        current = np.isin(session_index, tuple(validation_sessions)) & valid_mask
        if np.any(labels[current, head] == value):
            continue
        candidates = [
            int(session)
            for session in sessions
            if int(session) not in validation_sessions
            and np.any(labels[(session_index == session) & valid_mask, head] == value)
        ]
        if candidates:
            validation_sessions.add(candidates[0])
        # No session anywhere has this class (e.g. no recorded jump at all):
        # nothing to add. The gate itself treats a genuinely absent class as
        # "no examples", never a failure, so this is a safe no-op.

    validation_mask = np.isin(session_index, tuple(validation_sessions))
    return np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)


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


# Hybrid scripted-teacher + human-demonstration training -------------------
#
# The scripted synthetic teacher (scripted_policies.py, rolling through
# synthetic maps) remains the source of broad layout diversity,
# group-selection behavior, obstacle avoidance, and curriculum coverage --
# human recordings were never used to derive it and do not replace it here.
# Human demonstrations instead supervise realistic steering/event
# combinations and regularize the policy toward real control behavior. An
# explicit round budget (not raw sample-count concatenation) keeps either
# source from overwhelming the other, and validation is reported separately
# per source (plus combined, per synthetic layout, and per human session) so
# a regression in one source's behavior can never hide inside an aggregate
# number.


def collect_human_demonstration_dataset_v193(
    recording_paths: list[Path],
    eva_only_recording_paths: list[Path],
    *,
    output: str | Path,
    map_model: Any = None,
) -> dict[str, Any] | None:
    """Step B: export every eligible human recording into one dataset.

    ``recording_paths`` must already be filtered to archives where
    movement_classification == keyboard_wasd and provenance is explicit
    (see ``recording_discovery.discover_direct_demonstration_eligible``);
    they supervise both steering and event. ``eva_only_recording_paths``
    (see ``discover_eva_only_supplementary``) supervise the event head only
    -- their movement is click-to-move, mixed, or otherwise unattested and
    must never supervise steering; ``export_demonstrations`` already enforces
    this per-sample via ``steering_label_valid``.

    Returns ``None``, not an empty dataset, when there is nothing to export,
    so callers can cleanly fall back to the scripted-only path.
    """

    if not recording_paths and not eva_only_recording_paths:
        return None
    from .demonstrations import export_demonstrations

    path = export_demonstrations(
        recording_paths,
        output,
        eva_only_recording_paths=eva_only_recording_paths,
        map_model=map_model,
    )
    with np.load(path, allow_pickle=False) as data:
        actions = np.asarray(data["actions"], dtype=np.int64)
        sessions = np.asarray(data["session_index"], dtype=np.int64)
        steering_valid = np.asarray(data["steering_label_valid"], dtype=np.bool_)
        event_valid = np.asarray(data["event_label_valid"], dtype=np.bool_)
    return {
        "path": str(Path(path).resolve()),
        "samples": int(len(actions)),
        "sessions": int(len(np.unique(sessions))),
        "steering_capable_sessions": int(len(np.unique(sessions[steering_valid]))) if np.any(steering_valid) else 0,
        "steering_counts": np.bincount(actions[steering_valid, 0], minlength=3).tolist(),
        "event_counts": np.bincount(actions[event_valid, 1], minlength=3).tolist(),
        "direct_demonstration_recordings": [str(p) for p in recording_paths],
        "eva_only_recordings": [str(p) for p in eva_only_recording_paths],
    }


def _interleave_source_schedule(scripted_rounds: int, human_rounds: int) -> list[str]:
    """Evenly interleave two labeled round counts, e.g. 8 scripted + 4 human
    becomes [s, s, h, s, s, h, s, s, h, s, s, h] rather than running one
    source to completion before the other -- a genuinely mixed curriculum,
    not extra epochs tacked on at the end for whichever source goes last.
    """

    total = int(scripted_rounds) + int(human_rounds)
    if total <= 0:
        return []
    schedule: list[str] = []
    scripted_done = human_done = 0
    for _ in range(total):
        scripted_progress = (
            scripted_done / scripted_rounds if scripted_rounds else float("inf")
        )
        human_progress = human_done / human_rounds if human_rounds else float("inf")
        if scripted_progress <= human_progress and scripted_done < scripted_rounds:
            schedule.append("scripted")
            scripted_done += 1
        else:
            schedule.append("human")
            human_done += 1
    return schedule


def train_hybrid_factorized_teacher_v193(
    model: Any,
    scripted_dataset_path: str | Path,
    human_dataset_path: str | Path | None,
    *,
    recognition_epochs: int = 12,
    recognition_learning_rate: float = 3.0e-4,
    calibration_epochs: int = 8,
    calibration_learning_rate: float = 3.0e-5,
    batch_size: int = 256,
    human_fraction: float = 0.35,
    minimum_human_sessions: int = 2,
    seed: int = 0,
) -> dict[str, Any]:
    """Steps C-E: hybrid scripted-teacher + human-demonstration training.

    Falls back to the pure scripted-only ``train_teacher_clone_v193`` path
    when there is no usable human dataset (fewer than
    ``minimum_human_sessions`` independent sessions with steering-valid
    samples), so this stays safe to call against an empty
    recordings/training/ directory -- e.g. in CI or a fresh checkout.
    """

    import torch

    with np.load(Path(scripted_dataset_path), allow_pickle=False) as data:
        scripted_observations = np.asarray(data["observations"], dtype=np.float32)
        scripted_labels = np.asarray(data["actions"], dtype=np.int64)
        scripted_train_indices = np.asarray(data["train_indices"], dtype=np.int64)
        scripted_validation_indices = np.asarray(data["validation_indices"], dtype=np.int64)
        scripted_layout_names = np.asarray(data["layout_names"], dtype=str).tolist()
        scripted_layout_index = np.asarray(data["layout_index"], dtype=np.int64)

    human_observations = human_labels = human_sessions = None
    human_steering_valid = human_event_valid = None
    human_continuous_session_mask = None
    human_train_indices = human_validation_indices = np.zeros(0, dtype=np.int64)
    human_ready = False
    if human_dataset_path is not None:
        with np.load(Path(human_dataset_path), allow_pickle=False) as data:
            human_observations = np.asarray(data["observations"], dtype=np.float32)
            human_labels = np.asarray(data["actions"], dtype=np.int64)
            human_sessions = np.asarray(data["session_index"], dtype=np.int64)
            human_steering_valid = np.asarray(data["steering_label_valid"], dtype=np.bool_)
            human_event_valid = np.asarray(data["event_label_valid"], dtype=np.bool_)
            # eva_only-role sessions only ever export their EVA-cast frames
            # (see demonstrations.py), so their per-sample event label is
            # always CAST_EVA -- a real positive recognition example, but not
            # a natural sample of how often EVA should fire during ordinary
            # play. Restrict the "natural" event prior below to sessions
            # that recorded continuous, unfiltered direct-keyboard play.
            session_roles = np.asarray(
                data.get(
                    "source_recording_role",
                    np.full(int(human_sessions.max()) + 1 if len(human_sessions) else 0, "direct_keyboard"),
                ),
                dtype=str,
            )
        continuous_session_ids = np.flatnonzero(session_roles == "direct_keyboard")
        human_continuous_session_mask = np.isin(human_sessions, continuous_session_ids)
        steering_capable_sessions = np.unique(human_sessions[human_steering_valid])
        if len(steering_capable_sessions) >= int(minimum_human_sessions):
            human_ready = True
            human_train_indices, human_validation_indices = _human_session_stratified_split(
                human_sessions,
                human_steering_valid,
                human_event_valid,
                human_labels,
                validation_fraction=0.2,
                seed=seed,
            )

    policy = model.policy
    rng = np.random.default_rng(seed)

    if not human_ready:
        fallback = train_teacher_clone_v193(
            model,
            scripted_dataset_path,
            recognition_epochs=recognition_epochs,
            recognition_learning_rate=recognition_learning_rate,
            calibration_epochs=calibration_epochs,
            calibration_learning_rate=calibration_learning_rate,
            batch_size=batch_size,
            seed=seed,
        )
        reason = (
            "no human dataset supplied"
            if human_dataset_path is None
            else (
                f"only {len(np.unique(human_sessions[human_steering_valid])) if human_sessions is not None else 0} "
                f"human session(s) with steering-valid samples; need at least {minimum_human_sessions}"
            )
        )
        return {**fallback, "human_dataset_used": False, "human_fallback_reason": reason}

    # --- Step C: recognition phase, source-aware round-robin over one optimizer ---
    scripted_steering_values = _training_values(
        scripted_labels[:, 0], scripted_train_indices, required=(0, 1, 2)
    )
    scripted_event_values = _training_values(
        scripted_labels[:, 1],
        scripted_train_indices,
        required=(int(FarmingEvent.NONE), int(FarmingEvent.CAST_EVA)),
        optional=(int(FarmingEvent.JUMP),),
        minimum_optional_support=16,
    )
    scripted_event_fractions = _event_sampling_fractions(scripted_event_values)

    human_steering_train_indices = human_train_indices[human_steering_valid[human_train_indices]]
    human_event_train_indices = human_train_indices[human_event_valid[human_train_indices]]
    human_steering_values = _training_values(
        human_labels[:, 0], human_steering_train_indices,
        required=(), optional=(0, 1, 2), minimum_optional_support=8,
    )
    human_event_values = _training_values(
        human_labels[:, 1], human_event_train_indices,
        required=(int(FarmingEvent.NONE), int(FarmingEvent.CAST_EVA)),
        optional=(int(FarmingEvent.JUMP),),
        minimum_optional_support=8,
    )
    # Equal-count balancing (event_target_fractions=None) would promote a
    # handful of real human jump presses to a full third of every human
    # event batch, repeating each one dozens of times. Use the same capped
    # fractional scheme as the scripted side instead.
    human_event_fractions = (
        _event_sampling_fractions(human_event_values) if human_event_values else None
    )

    recognition_optimizer = torch.optim.Adam(policy.parameters(), lr=float(recognition_learning_rate))
    human_rounds = max(1, int(round(recognition_epochs * human_fraction)))
    scripted_rounds = max(1, int(recognition_epochs) - human_rounds)
    schedule = _interleave_source_schedule(scripted_rounds, human_rounds)

    recognition_rounds: list[dict[str, Any]] = []
    for source in schedule:
        if source == "scripted":
            report = _train_balanced_factorized_heads(
                policy, recognition_optimizer, scripted_observations, scripted_labels,
                steering_indices=scripted_train_indices, event_indices=scripted_train_indices,
                steering_values=scripted_steering_values, event_values=scripted_event_values,
                epochs=1, batch_size=batch_size, event_loss_scale=1.5, rng=rng,
                event_target_fractions=scripted_event_fractions,
            )
        else:
            report = _train_balanced_factorized_heads(
                policy, recognition_optimizer, human_observations, human_labels,
                steering_indices=human_steering_train_indices, event_indices=human_event_train_indices,
                steering_values=human_steering_values, event_values=human_event_values,
                epochs=1, batch_size=batch_size, event_loss_scale=1.5, rng=rng,
                event_target_fractions=human_event_fractions,
            )
        recognition_rounds.append({"source": source, **report})

    recognition_validation = {
        "scripted": _prediction_diagnostics(
            policy, scripted_observations[scripted_validation_indices],
            scripted_labels[scripted_validation_indices], batch_size=batch_size,
        ),
        "human": _prediction_diagnostics(
            policy, human_observations[human_validation_indices],
            human_labels[human_validation_indices], batch_size=batch_size,
            steering_valid_mask=human_steering_valid[human_validation_indices],
        ),
    }
    # Step E: only continue once the policy performs acceptably on BOTH sources.
    if not (
        recognition_validation["scripted"]["gate"]["passed"]
        and recognition_validation["human"]["gate"]["passed"]
    ):
        return {
            "passed": False,
            "phase": "recognition",
            "human_dataset_used": True,
            "recognition_rounds": recognition_rounds,
            "recognition_validation": recognition_validation,
            "reasons": (
                recognition_validation["scripted"]["gate"]["reasons"]
                + recognition_validation["human"]["gate"]["reasons"]
            ),
        }

    # --- bias correction: target a human_fraction-weighted blend of both
    # sources' natural priors, so calibration does not immediately fight the
    # mix ratio recognition training was just given. ---
    scripted_steering_target = _class_priors(scripted_labels[:, 0], scripted_train_indices, 3)
    scripted_event_target = _class_priors(scripted_labels[:, 1], scripted_train_indices, 3)
    human_steering_target = (
        _class_priors(human_labels[:, 0], human_steering_train_indices, 3)
        if len(human_steering_train_indices)
        else scripted_steering_target
    )
    human_event_target = _natural_human_event_target(
        human_labels, human_event_train_indices, human_continuous_session_mask, scripted_event_target,
    )
    blended_steering_target = (
        (1.0 - human_fraction) * scripted_steering_target + human_fraction * human_steering_target
    )
    blended_event_target = (
        (1.0 - human_fraction) * scripted_event_target + human_fraction * human_event_target
    )
    blended_steering_target = blended_steering_target / blended_steering_target.sum()
    blended_event_target = blended_event_target / blended_event_target.sum()

    bias_correction = _apply_prior_bias_correction(
        policy,
        steering_target=blended_steering_target,
        event_target=blended_event_target,
        event_sampling_fractions=scripted_event_fractions,
    )

    # --- Step C continued: calibration phase, mixed-source natural-prior CE ---
    scripted_steering_weights = _sqrt_inverse_class_weights(scripted_labels[:, 0], scripted_train_indices, 3)
    scripted_event_weights = _sqrt_inverse_class_weights(scripted_labels[:, 1], scripted_train_indices, 3)
    human_steering_weights = (
        _sqrt_inverse_class_weights(human_labels[:, 0], human_steering_train_indices, 3)
        if len(human_steering_train_indices)
        else scripted_steering_weights
    )
    human_event_weights = (
        _sqrt_inverse_class_weights(human_labels[:, 1], human_event_train_indices, 3)
        if len(human_event_train_indices)
        else scripted_event_weights
    )
    calibration_optimizer = torch.optim.Adam(policy.parameters(), lr=float(calibration_learning_rate))
    calibration_human_rounds = max(1, int(round(calibration_epochs * human_fraction)))
    calibration_scripted_rounds = max(1, int(calibration_epochs) - calibration_human_rounds)
    calibration_schedule = _interleave_source_schedule(calibration_scripted_rounds, calibration_human_rounds)

    calibration_rounds: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_score = float("-inf")
    for epoch, source in enumerate(calibration_schedule, start=1):
        if source == "scripted":
            training = _train_natural_prior_epoch(
                policy, calibration_optimizer, scripted_observations, scripted_labels, scripted_train_indices,
                batch_size=batch_size, steering_weights=scripted_steering_weights,
                event_weights=scripted_event_weights, event_loss_scale=1.15, rng=rng,
            )
        else:
            training = _train_natural_prior_epoch(
                policy, calibration_optimizer, human_observations, human_labels, human_train_indices,
                batch_size=batch_size, steering_weights=human_steering_weights,
                event_weights=human_event_weights, event_loss_scale=1.15, rng=rng,
                steering_valid_mask=human_steering_valid,
            )
        validation = {
            "scripted": _prediction_diagnostics(
                policy, scripted_observations[scripted_validation_indices],
                scripted_labels[scripted_validation_indices], batch_size=batch_size,
            ),
            "human": _prediction_diagnostics(
                policy, human_observations[human_validation_indices],
                human_labels[human_validation_indices], batch_size=batch_size,
                steering_valid_mask=human_steering_valid[human_validation_indices],
            ),
        }
        both_passed = validation["scripted"]["gate"]["passed"] and validation["human"]["gate"]["passed"]
        score = _calibration_score(validation["scripted"]) + _calibration_score(validation["human"])
        calibration_rounds.append(
            {"epoch": epoch, "source": source, "score": score, "training": training, "validation": validation}
        )
        if both_passed and score > best_score:
            best_score = score
            best_state = copy.deepcopy(policy.state_dict())

    if best_state is None:
        last = calibration_rounds[-1]["validation"] if calibration_rounds else recognition_validation
        return {
            "passed": False,
            "phase": "calibration",
            "human_dataset_used": True,
            "recognition_rounds": recognition_rounds,
            "recognition_validation": recognition_validation,
            "bias_correction": bias_correction,
            "calibration_rounds": calibration_rounds,
            "reasons": last["scripted"]["gate"]["reasons"] + last["human"]["gate"]["reasons"],
        }

    policy.load_state_dict(best_state)
    final_scripted = _prediction_diagnostics(
        policy, scripted_observations[scripted_validation_indices],
        scripted_labels[scripted_validation_indices], batch_size=batch_size,
    )
    final_human = _prediction_diagnostics(
        policy, human_observations[human_validation_indices],
        human_labels[human_validation_indices], batch_size=batch_size,
        steering_valid_mask=human_steering_valid[human_validation_indices],
    )
    final_combined = _prediction_diagnostics(
        policy,
        np.concatenate(
            (scripted_observations[scripted_validation_indices], human_observations[human_validation_indices])
        ),
        np.concatenate(
            (scripted_labels[scripted_validation_indices], human_labels[human_validation_indices])
        ),
        steering_valid_mask=np.concatenate(
            (
                np.ones(len(scripted_validation_indices), dtype=np.bool_),
                human_steering_valid[human_validation_indices],
            )
        ),
        batch_size=batch_size,
    )

    # Step D: separate reporting per synthetic layout and per human session,
    # so a single bad layout or a single unusual friend's session can never
    # hide inside an aggregate validation number.
    per_layout_validation: dict[str, Any] = {}
    for layout_id, layout_name in enumerate(scripted_layout_names):
        mask = scripted_layout_index[scripted_validation_indices] == layout_id
        if not np.any(mask):
            per_layout_validation[str(layout_name)] = {"passed": False, "reason": "no validation samples"}
            continue
        selected = scripted_validation_indices[mask]
        per_layout_validation[str(layout_name)] = _prediction_diagnostics(
            policy, scripted_observations[selected], scripted_labels[selected], batch_size=batch_size
        )
    per_session_validation: dict[str, Any] = {}
    for session in np.unique(human_sessions[human_validation_indices]):
        selected = human_validation_indices[human_sessions[human_validation_indices] == session]
        per_session_validation[str(int(session))] = _prediction_diagnostics(
            policy, human_observations[selected], human_labels[selected], batch_size=batch_size,
            steering_valid_mask=human_steering_valid[selected],
        )

    passed = bool(final_scripted["gate"]["passed"] and final_human["gate"]["passed"])
    return {
        "passed": passed,
        "phase": "complete",
        "human_dataset_used": True,
        "human_fraction_target": float(human_fraction),
        "scripted_dataset": str(Path(scripted_dataset_path).resolve()),
        "human_dataset": str(Path(human_dataset_path).resolve()),
        "recognition_epochs": int(recognition_epochs),
        "calibration_epochs": int(calibration_epochs),
        "recognition_rounds": recognition_rounds,
        "recognition_validation": recognition_validation,
        "bias_correction": bias_correction,
        "blended_steering_target": blended_steering_target.tolist(),
        "blended_event_target": blended_event_target.tolist(),
        "calibration_rounds": calibration_rounds,
        "selected_calibration_score": float(best_score),
        "validation": {
            "scripted": final_scripted,
            "human": final_human,
            "combined": final_combined,
        },
        "per_layout_validation": per_layout_validation,
        "per_session_validation": per_session_validation,
        "reasons": final_scripted["gate"]["reasons"] + final_human["gate"]["reasons"],
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


def fine_tune_steering_branch_v193(
    model: Any,
    dataset_path: str | Path,
    *,
    epochs: int = 8,
    learning_rate: float = 3.0e-4,
    batch_size: int = 128,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, Any]:
    """Phase 2 scoped fine-tune: trains ONLY steering_net/steering_out on
    the stratified navigation dataset (simulator.navigation_dataset),
    freezing event_net/vf_net/event_out/value_net so event/value behavior
    is provably unaffected -- see the `event_head_unaffected` field below.

    `model.policy` must be a `SplitSteeringNavigationPolicy`
    (simulator.split_branch_policy). `dataset_path` must be an .npz saved by
    simulator.navigation_dataset.mine_navigation_dataset's caller, holding
    925-value (raw+sidecar) observations, `actions` (steering, event) pairs,
    `layout_index`, and `episode_index`.
    """

    import torch
    import torch.nn.functional as F

    from .navigation_history import POLICY_INPUT_SIZE
    from .split_branch_policy import STEERING_NAVIGATION_FEATURE_SIZE

    with np.load(Path(dataset_path), allow_pickle=False) as data:
        observations = np.asarray(data["observations"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.int64)
        layout_index = np.asarray(data["layout_index"], dtype=np.int64)
        episode_index = np.asarray(data["episode_index"], dtype=np.int64)

    if observations.shape[1] != POLICY_INPUT_SIZE:
        raise ValueError(
            f"dataset observations must be {POLICY_INPUT_SIZE}-valued (raw+sidecar), got {observations.shape[1]}"
        )

    train_idx, val_idx, val_episodes = _layout_stratified_episode_split(
        episode_index, layout_index, actions, validation_fraction=validation_fraction, seed=seed,
    )

    policy = model.policy
    steering_first_layer = getattr(getattr(policy, "mlp_extractor", None), "steering_net", [None])[0]
    actual_steering_input = getattr(steering_first_layer, "in_features", None)
    if actual_steering_input != STEERING_NAVIGATION_FEATURE_SIZE:
        raise ValueError(
            f"policy.mlp_extractor.steering_net expects {actual_steering_input} steering inputs, "
            f"but this fine-tune is scoped to the {STEERING_NAVIGATION_FEATURE_SIZE}-feature Phase 2 "
            "variant -- is model.policy a SplitSteeringNavigationPolicy (not SplitSteeringEventPolicy)?"
        )

    trainable_params = []
    for name, param in policy.named_parameters():
        is_steering = "mlp_extractor.steering_net" in name or "action_net.steering_out" in name
        param.requires_grad = is_steering
        if is_steering:
            trainable_params.append(param)
    assert trainable_params, "unreachable: steering_input check above already validated the architecture"

    policy.eval()
    before = _prediction_diagnostics(policy, observations[val_idx], actions[val_idx], batch_size=batch_size)

    optimizer = torch.optim.Adam(trainable_params, lr=float(learning_rate))
    policy.train()
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        order = rng.permutation(train_idx)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(order), int(batch_size)):
            batch_idx = order[start : start + int(batch_size)]
            obs = torch.as_tensor(observations[batch_idx], device=policy.device)
            steering_labels = torch.as_tensor(actions[batch_idx, 0], device=policy.device, dtype=torch.long)
            distribution = policy.get_distribution(obs).distribution
            loss = F.cross_entropy(distribution[0].logits, steering_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        history.append({"epoch": epoch, "mean_loss": epoch_loss / max(1, n_batches)})

    policy.eval()
    after = _prediction_diagnostics(policy, observations[val_idx], actions[val_idx], batch_size=batch_size)

    for _name, param in policy.named_parameters():
        param.requires_grad = True  # restore, standard SB3 policies expect all-trainable

    event_unaffected = (
        before["event"]["samples"] == after["event"]["samples"]
        and np.allclose(before["event"]["mean_probabilities"], after["event"]["mean_probabilities"], atol=1e-6)
    )

    return {
        "dataset": str(Path(dataset_path).resolve()),
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
        "validation_episodes": val_episodes,
        "history": history,
        "before": before,
        "after": after,
        "event_head_unaffected": bool(event_unaffected),
    }
