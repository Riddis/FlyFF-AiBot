from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from farming.actions import FarmingEvent, SteeringAction
from farming.model_contract import ModelContractMetadata, ModelSpaceSignature, validate_model_contract


def validate_factorized_policy_contract(model: Any) -> None:
    metadata = getattr(model, "farming_contract_metadata", None)
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint is missing factorized farming contract metadata")
    validate_model_contract(
        ModelSpaceSignature.from_spaces(model.observation_space, model.action_space),
        metadata=metadata,
    )


def atomic_save_policy(model: Any, path: str | Path) -> Path:
    requested = Path(path)
    target = requested if requested.suffix.lower() == ".zip" else Path(f"{requested}.zip")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.{uuid4().hex}.tmp.zip")
    try:
        model.save(str(temporary))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _head_predictions(policy: Any, observations: np.ndarray, batch_size: int) -> np.ndarray:
    import torch

    result: list[np.ndarray] = []
    device = policy.device
    policy.eval()
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            obs = torch.as_tensor(observations[start : start + batch_size], device=device)
            distribution = policy.get_distribution(obs)
            categoricals = distribution.distribution
            if not isinstance(categoricals, (list, tuple)) or len(categoricals) != 2:
                raise ValueError("Policy does not expose two MultiDiscrete categorical heads")
            result.append(
                np.column_stack(
                    [cat.probs.argmax(dim=1).cpu().numpy() for cat in categoricals]
                ).astype(np.int64)
            )
    return np.concatenate(result, axis=0)


def factorized_stage_gate(
    expected: np.ndarray,
    predicted: np.ndarray,
    *,
    steering_valid_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Score both factorized heads, optionally excluding untrustworthy steering rows.

    Some human recordings (click-to-move / mixed control) can supply a real
    event label without any trustworthy steering label -- their steering
    column is a meaningless default, never inferred from mouse movement.
    ``steering_valid_mask`` (aligned with ``expected``/``predicted`` rows)
    keeps those rows out of the steering head's scoring and out of
    exact-command accuracy, while the event head still scores every row.
    Omitting the mask reproduces the original unmasked behaviour exactly.
    """

    truth = np.asarray(expected, dtype=np.int64)
    guesses = np.asarray(predicted, dtype=np.int64)
    if truth.ndim != 2 or truth.shape[1] != 2 or guesses.shape != truth.shape:
        raise ValueError("Factorized stage-gate actions must have shape (N, 2)")
    if steering_valid_mask is None:
        steering_mask = np.ones(len(truth), dtype=np.bool_)
    else:
        steering_mask = np.asarray(steering_valid_mask, dtype=np.bool_)
        if steering_mask.shape != (len(truth),):
            raise ValueError("steering_valid_mask must align with the label rows")
    event_mask = np.ones(len(truth), dtype=np.bool_)
    reasons: list[str] = []
    heads: dict[str, Any] = {}
    specs = (
        ("steering", 3, (0, 1, 2), 0.15, steering_mask),
        ("event", 3, (0, 1), 0.20, event_mask),
    )
    for index, (name, count, required, minimum_recall, head_mask) in enumerate(specs):
        head_truth = truth[head_mask, index]
        head_guesses = guesses[head_mask, index]
        confusion = np.zeros((count, count), dtype=np.int64)
        if head_truth.size:
            np.add.at(confusion, (head_truth, head_guesses), 1)
        predicted_counts = confusion.sum(axis=0)
        per_class = []
        for value in range(count):
            support = int(confusion[value].sum())
            predicted_support = int(confusion[:, value].sum())
            tp = int(confusion[value, value])
            recall = float(tp / max(1, support))
            precision = float(tp / max(1, predicted_support))
            per_class.append(
                {
                    "value": value,
                    "support": support,
                    "predicted": predicted_support,
                    "precision": precision,
                    "recall": recall,
                }
            )
            if value in required:
                if support == 0:
                    reasons.append(f"validation has no {name}={value} examples")
                elif recall < minimum_recall:
                    reasons.append(
                        f"{name}={value} recall {recall:.3f} is below {minimum_recall:.3f}"
                    )
        maximum_fraction = float(predicted_counts.max() / max(1, int(head_truth.size)))
        if name == "steering" and maximum_fraction > 0.90:
            reasons.append(
                f"one steering choice occupies {maximum_fraction:.3f} of predictions"
            )
        if name == "event" and int(predicted_counts[int(FarmingEvent.CAST_EVA)]) == 0:
            reasons.append("validation predicts no EVA events")
        heads[name] = {
            "accuracy": float(np.mean(head_truth == head_guesses)) if head_truth.size else 0.0,
            "maximum_prediction_fraction": maximum_fraction,
            "predicted_counts": predicted_counts.tolist(),
            "confusion_matrix_true_rows": confusion.tolist(),
            "per_class": per_class,
            "samples": int(head_truth.size),
        }
    exact_truth = truth[steering_mask]
    exact_guesses = guesses[steering_mask]
    return {
        "passed": not reasons,
        "samples": int(len(truth)),
        "exact_command_accuracy": (
            float(np.mean(np.all(exact_truth == exact_guesses, axis=1)))
            if exact_truth.size
            else 0.0
        ),
        "heads": heads,
        "reasons": reasons,
    }


def _balanced_head_weights(labels: np.ndarray, indices: np.ndarray, count: int) -> np.ndarray:
    """Return diagnostic inverse-frequency weights for one action head.

    Training no longer relies on these weights alone.  Rare event labels are
    explicitly resampled into balanced head-specific batches below; the
    weights remain useful in reports and for callers that only need a compact
    description of the class imbalance.
    """

    counts = np.bincount(labels[indices], minlength=count).astype(np.float64)
    weights = np.zeros(count, dtype=np.float32)
    present = counts > 0
    weights[present] = len(indices) / counts[present]
    if np.any(present):
        weights[present] /= float(np.mean(weights[present]))
    return weights


def _balanced_resample_indices(
    labels: np.ndarray,
    indices: np.ndarray,
    values: tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """Build one equal-class epoch by oversampling only inside train data.

    This is deliberately head-specific.  A factorized policy sees steering on
    every control step but EVA on only a small fraction of steps.  A single
    naturally distributed batch therefore still lets the NONE event dominate
    even when a mild class weight is used.
    """

    source = np.asarray(indices, dtype=np.int64).reshape(-1)
    if source.size < 1:
        raise ValueError("Cannot balance an empty training split")
    buckets: list[np.ndarray] = []
    for value in values:
        bucket = source[np.asarray(labels[source], dtype=np.int64) == int(value)]
        if bucket.size < 1:
            raise ValueError(f"Training split has no examples for required value {value}")
        buckets.append(bucket)
    target = max(int(bucket.size) for bucket in buckets)
    sampled = [
        rng.choice(bucket, size=target, replace=bool(bucket.size < target))
        for bucket in buckets
    ]
    order = np.concatenate(sampled).astype(np.int64, copy=False)
    rng.shuffle(order)
    return order


def _fractional_resample_indices(
    labels: np.ndarray,
    indices: np.ndarray,
    values: tuple[int, ...],
    fractions: dict[int, float],
    rng: np.random.Generator,
    *,
    anchor_values: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Resample one head to explicit class fractions without inflating jump.

    ``anchor_values`` define the classes whose original support determines the
    epoch size. Optional classes such as jump may therefore be capped to a
    small fraction even when they are present, instead of being promoted to an
    equal third of every event epoch.
    """

    source = np.asarray(indices, dtype=np.int64).reshape(-1)
    if source.size < 1:
        raise ValueError("Cannot resample an empty training split")
    normalized_values = tuple(int(value) for value in values)
    if not normalized_values:
        raise ValueError("At least one class value is required")
    buckets: dict[int, np.ndarray] = {}
    for value in normalized_values:
        bucket = source[np.asarray(labels[source], dtype=np.int64) == value]
        if bucket.size < 1:
            raise ValueError(f"Training split has no examples for required value {value}")
        buckets[value] = bucket

    selected_fractions = {value: float(fractions.get(value, 0.0)) for value in normalized_values}
    if any(not np.isfinite(value) or value <= 0.0 for value in selected_fractions.values()):
        raise ValueError("Every sampled class needs a finite positive target fraction")
    total_fraction = float(sum(selected_fractions.values()))
    selected_fractions = {value: fraction / total_fraction for value, fraction in selected_fractions.items()}

    anchors = tuple(int(value) for value in (anchor_values or normalized_values))
    if any(value not in buckets for value in anchors):
        raise ValueError("Every anchor class must also be sampled")
    epoch_size = max(
        int(np.ceil(buckets[value].size / selected_fractions[value]))
        for value in anchors
    )
    target_counts = {
        value: max(1, int(round(epoch_size * selected_fractions[value])))
        for value in normalized_values
    }
    sampled = [
        rng.choice(
            buckets[value],
            size=target_counts[value],
            replace=bool(buckets[value].size < target_counts[value]),
        )
        for value in normalized_values
    ]
    order = np.concatenate(sampled).astype(np.int64, copy=False)
    rng.shuffle(order)
    return order


def _event_sampling_fractions(values: tuple[int, ...]) -> dict[int, float]:
    """Keep NONE dominant, oversample EVA, and cap jump at five percent."""

    normalized = tuple(int(value) for value in values)
    if int(FarmingEvent.JUMP) in normalized:
        return {
            int(FarmingEvent.NONE): 0.55,
            int(FarmingEvent.CAST_EVA): 0.40,
            int(FarmingEvent.JUMP): 0.05,
        }
    return {
        int(FarmingEvent.NONE): 0.55,
        int(FarmingEvent.CAST_EVA): 0.45,
    }


def _training_values(
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    required: tuple[int, ...],
    optional: tuple[int, ...] = (),
    minimum_optional_support: int = 16,
) -> tuple[int, ...]:
    values = list(required)
    source = np.asarray(indices, dtype=np.int64).reshape(-1)
    for value in optional:
        support = int(np.count_nonzero(np.asarray(labels[source], dtype=np.int64) == int(value)))
        if support >= int(minimum_optional_support):
            values.append(int(value))
    return tuple(values)


def _train_balanced_factorized_heads(
    policy: Any,
    optimizer: Any,
    observations: np.ndarray,
    labels: np.ndarray,
    *,
    steering_indices: np.ndarray,
    event_indices: np.ndarray,
    steering_values: tuple[int, ...],
    event_values: tuple[int, ...],
    epochs: int,
    batch_size: int,
    event_loss_scale: float,
    rng: np.random.Generator,
    event_target_fractions: dict[int, float] | None = None,
) -> dict[str, float | int | dict[str, int] | dict[str, float]]:
    """Train both categorical heads from independently balanced batches.

    Each optimiser step may contain one steering batch and one event batch.
    Gradients are accumulated before the step so event repair does not simply
    overwrite the shared representation learned for steering.
    """

    import torch
    import torch.nn.functional as F

    device = policy.device
    losses: list[float] = []
    steering_updates = 0
    event_updates = 0
    policy.train()
    for _ in range(int(epochs)):
        # An empty values tuple means this call has nothing to teach that
        # head this round (e.g. a human-demonstration source contributing
        # only event supervision, no steering-valid samples this session) --
        # skip it rather than asking the resampler to balance zero classes.
        if not steering_values:
            steering_order = np.zeros(0, dtype=np.int64)
        else:
            steering_order = _balanced_resample_indices(
                labels[:, 0], steering_indices, steering_values, rng
            )
        if not event_values:
            event_order = np.zeros(0, dtype=np.int64)
        elif event_target_fractions is None:
            event_order = _balanced_resample_indices(
                labels[:, 1], event_indices, event_values, rng
            )
        else:
            event_order = _fractional_resample_indices(
                labels[:, 1],
                event_indices,
                event_values,
                event_target_fractions,
                rng,
                anchor_values=(int(FarmingEvent.NONE), int(FarmingEvent.CAST_EVA)),
            )
        steering_batches = [
            steering_order[start : start + int(batch_size)]
            for start in range(0, len(steering_order), int(batch_size))
        ]
        event_batches = [
            event_order[start : start + int(batch_size)]
            for start in range(0, len(event_order), int(batch_size))
        ]
        updates = max(len(steering_batches), len(event_batches))
        for update in range(updates):
            loss_terms = []
            if update < len(steering_batches):
                indices = steering_batches[update]
                obs_tensor = torch.as_tensor(observations[indices], device=device)
                target = torch.as_tensor(labels[indices, 0], device=device)
                categories = policy.get_distribution(obs_tensor).distribution
                if not isinstance(categories, (list, tuple)) or len(categories) != 2:
                    raise ValueError("PPO policy is not MultiDiscrete([3, 3])")
                loss_terms.append(F.cross_entropy(categories[0].logits, target))
                steering_updates += 1
            if update < len(event_batches):
                indices = event_batches[update]
                obs_tensor = torch.as_tensor(observations[indices], device=device)
                target = torch.as_tensor(labels[indices, 1], device=device)
                categories = policy.get_distribution(obs_tensor).distribution
                if not isinstance(categories, (list, tuple)) or len(categories) != 2:
                    raise ValueError("PPO policy is not MultiDiscrete([3, 3])")
                loss_terms.append(
                    float(event_loss_scale)
                    * F.cross_entropy(categories[1].logits, target)
                )
                event_updates += 1
            if not loss_terms:
                continue
            loss = sum(loss_terms)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
    event_sample_counts: dict[str, int] = {}
    event_sample_fractions: dict[str, float] = {}
    if 'event_order' in locals():
        sampled_event_labels = np.asarray(labels[event_order, 1], dtype=np.int64)
        total_sampled = max(1, len(sampled_event_labels))
        event_sample_counts = {
            str(value): int(np.count_nonzero(sampled_event_labels == int(value)))
            for value in event_values
        }
        event_sample_fractions = {
            key: float(value / total_sampled) for key, value in event_sample_counts.items()
        }
    return {
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
        "updates": int(len(losses)),
        "steering_updates": int(steering_updates),
        "event_updates": int(event_updates),
        "event_sample_counts_last_epoch": event_sample_counts,
        "event_sample_fractions_last_epoch": event_sample_fractions,
    }


def synthetic_teacher_clone_factorized(
    model: Any,
    curriculum_path: str | Path,
    *,
    stage: str = "early",
    samples: int = 6_000,
    episode_seconds: float = 30.0,
    max_actions: int = 220,
    teacher_policy: str = "obstacle_aware",
    epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 3.0e-4,
    seed: int = 0,
    raise_on_gate_failure: bool = True,
    dataset_output: str | Path | None = None,
) -> dict[str, Any]:
    import torch

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
    episodes: list[int] = []
    episode_id = 0
    try:
        while len(observations) < int(samples):
            for offset, (_entry, env) in enumerate(environments):
                observation, _ = env.reset(seed=seed + episode_id * 1009 + offset * 37)
                for _ in range(max_actions):
                    command = scripted_command(teacher_policy, env)
                    observations.append(np.asarray(observation, dtype=np.float32).copy())
                    actions.append(command.as_array())
                    episodes.append(episode_id)
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
    episode_index = np.asarray(episodes[:samples], dtype=np.int64)
    if obs.shape != (int(samples), 923) or labels.shape != (int(samples), 2):
        raise ValueError(f"Unexpected teacher arrays: observations={obs.shape}, actions={labels.shape}")
    if np.count_nonzero(labels[:, 1] == int(FarmingEvent.CAST_EVA)) < 32:
        raise ValueError("Teacher produced too few EVA examples")
    for steering in SteeringAction:
        if np.count_nonzero(labels[:, 0] == int(steering)) < 32:
            raise ValueError(f"Teacher produced too few {steering.name} steering examples")

    unique_episodes = np.unique(episode_index)
    if len(unique_episodes) < 5:
        raise ValueError("Teacher bootstrap requires at least five complete episodes")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_episodes)
    validation_count = max(1, int(round(len(shuffled) * 0.2)))
    validation_episodes = set(int(v) for v in shuffled[:validation_count])
    # Guarantee every required head value appears in validation without
    # splitting adjacent states from the same episode.
    required_values = (
        (0, int(SteeringAction.STRAIGHT)),
        (0, int(SteeringAction.LEFT)),
        (0, int(SteeringAction.RIGHT)),
        (1, int(FarmingEvent.NONE)),
        (1, int(FarmingEvent.CAST_EVA)),
    )
    for head, value in required_values:
        current_mask = np.isin(episode_index, tuple(validation_episodes))
        if np.any(labels[current_mask, head] == value):
            continue
        candidate = next(
            (
                int(ep)
                for ep in shuffled
                if int(ep) not in validation_episodes
                and np.any(labels[episode_index == ep, head] == value)
            ),
            None,
        )
        if candidate is None:
            raise ValueError(
                f"No complete teacher episode contains required head {head} value {value}"
            )
        validation_episodes.add(candidate)
    validation_mask = np.isin(episode_index, tuple(validation_episodes))
    train_indices = np.flatnonzero(~validation_mask)
    validation_indices = np.flatnonzero(validation_mask)

    steering_weights = _balanced_head_weights(labels[:, 0], train_indices, 3)
    event_weights = _balanced_head_weights(labels[:, 1], train_indices, 3)
    steering_values = _training_values(
        labels[:, 0],
        train_indices,
        required=(
            int(SteeringAction.STRAIGHT),
            int(SteeringAction.LEFT),
            int(SteeringAction.RIGHT),
        ),
    )
    event_values = _training_values(
        labels[:, 1],
        train_indices,
        required=(int(FarmingEvent.NONE), int(FarmingEvent.CAST_EVA)),
        optional=(int(FarmingEvent.JUMP),),
        minimum_optional_support=16,
    )
    event_sampling_fractions = _event_sampling_fractions(event_values)
    if dataset_output is not None:
        dataset_path = Path(dataset_output)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dataset_path,
            observations=obs,
            actions=labels,
            episode_index=episode_index,
            train_indices=train_indices,
            validation_indices=validation_indices,
            action_contract_id=np.asarray(["latched-forward-factorized-steering-event-v1"]),
            action_nvec=np.asarray([3, 3], dtype=np.int64),
        )
    else:
        dataset_path = None

    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(learning_rate))
    training_rounds: list[dict[str, Any]] = []
    validation: dict[str, Any] | None = None
    final_loss = 0.0
    # The first round uses the requested epoch count.  If deterministic EVA
    # recall is still missing, two short event-emphasised repair rounds are
    # allowed.  This remains a hard gate: failure is reported, never ignored.
    round_epochs = (int(epochs), max(4, int(epochs) // 2), max(4, int(epochs) // 2))
    event_scales = (1.5, 2.0, 2.5)
    for round_index, (epochs_this_round, event_scale) in enumerate(
        zip(round_epochs, event_scales, strict=True),
        start=1,
    ):
        training = _train_balanced_factorized_heads(
            policy,
            optimizer,
            obs,
            labels,
            steering_indices=train_indices,
            event_indices=train_indices,
            steering_values=steering_values,
            event_values=event_values,
            epochs=epochs_this_round,
            batch_size=int(batch_size),
            event_loss_scale=event_scale,
            rng=rng,
            event_target_fractions=event_sampling_fractions,
        )
        final_loss = float(training["mean_loss"])
        predictions = _head_predictions(policy, obs[validation_indices], int(batch_size))
        validation = factorized_stage_gate(labels[validation_indices], predictions)
        training_rounds.append(
            {
                "round": round_index,
                "epochs": epochs_this_round,
                "event_loss_scale": event_scale,
                **training,
                "validation": validation,
            }
        )
        if validation["passed"]:
            break
    assert validation is not None

    report = {
        "teacher_policy": teacher_policy,
        "samples": int(samples),
        "episodes": int(len(unique_episodes)),
        "steering_counts": np.bincount(labels[:, 0], minlength=3).tolist(),
        "event_counts": np.bincount(labels[:, 1], minlength=3).tolist(),
        "steering_class_weights": steering_weights.tolist(),
        "event_class_weights": event_weights.tolist(),
        "balanced_steering_values": list(steering_values),
        "balanced_event_values": list(event_values),
        "event_sampling_fractions": {str(k): float(v) for k, v in event_sampling_fractions.items()},
        "teacher_dataset": None if dataset_path is None else str(dataset_path.resolve()),
        "training_samples": int(len(train_indices)),
        "validation_samples": int(len(validation_indices)),
        "requested_epochs": int(epochs),
        "training_rounds": training_rounds,
        "final_loss": final_loss,
        "validation": validation,
    }
    setattr(model, "farming_contract_metadata", ModelContractMetadata.current().as_dict())
    setattr(model, "synthetic_teacher_metadata", report)
    if not validation["passed"] and raise_on_gate_failure:
        raise ValueError(
            "Factorized teacher gate failed after balanced-head repair: "
            + "; ".join(validation["reasons"])
        )
    return report


def rehearse_factorized_policy(
    model: Any,
    dataset_path: str | Path,
    *,
    epochs: int = 2,
    batch_size: int = 256,
    learning_rate: float = 2.5e-5,
    event_loss_scale: float = 1.75,
    seed: int = 0,
) -> dict[str, Any]:
    """Rehearse the scripted teacher after PPO without resetting PPO state.

    The policy's existing PPO optimiser is reused so its Adam moments remain
    coherent. Only the actor distribution participates in the supervised loss;
    jump remains trainable but is capped to five percent of event rehearsal.
    """

    with np.load(Path(dataset_path), allow_pickle=False) as data:
        observations = np.asarray(data["observations"], dtype=np.float32)
        labels = np.asarray(data["actions"], dtype=np.int64)
        train_indices = np.asarray(data["train_indices"], dtype=np.int64)
        validation_indices = np.asarray(data["validation_indices"], dtype=np.int64)
        contract_ids = np.asarray(data.get("action_contract_id", []), dtype=str).reshape(-1)
        nvec = np.asarray(data.get("action_nvec", []), dtype=np.int64).reshape(-1)
    if observations.ndim != 2 or observations.shape[1] != 923:
        raise ValueError(f"Teacher rehearsal observations must have shape (N, 923), got {observations.shape}")
    if labels.shape != (len(observations), 2):
        raise ValueError("Teacher rehearsal actions must have shape (N, 2)")
    if contract_ids.tolist() != ["latched-forward-factorized-steering-event-v1"]:
        raise ValueError("Teacher rehearsal dataset has an incompatible action contract")
    if tuple(int(value) for value in nvec.tolist()) != (3, 3):
        raise ValueError("Teacher rehearsal dataset must use MultiDiscrete([3, 3])")

    steering_values = _training_values(
        labels[:, 0],
        train_indices,
        required=(
            int(SteeringAction.STRAIGHT),
            int(SteeringAction.LEFT),
            int(SteeringAction.RIGHT),
        ),
    )
    event_values = _training_values(
        labels[:, 1],
        train_indices,
        required=(int(FarmingEvent.NONE), int(FarmingEvent.CAST_EVA)),
        optional=(int(FarmingEvent.JUMP),),
        minimum_optional_support=16,
    )
    event_sampling_fractions = _event_sampling_fractions(event_values)
    policy = model.policy
    before_predictions = _head_predictions(
        policy, observations[validation_indices], int(batch_size)
    )
    before_validation = factorized_stage_gate(
        labels[validation_indices], before_predictions
    )
    optimizer = getattr(policy, "optimizer", None)
    if optimizer is None:
        raise ValueError("PPO policy does not expose its optimiser for rehearsal")
    original_learning_rates = [float(group.get("lr", learning_rate)) for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)
    try:
        training = _train_balanced_factorized_heads(
            policy,
            optimizer,
            observations,
            labels,
            steering_indices=train_indices,
            event_indices=train_indices,
            steering_values=steering_values,
            event_values=event_values,
            epochs=int(epochs),
            batch_size=int(batch_size),
            event_loss_scale=float(event_loss_scale),
            rng=np.random.default_rng(seed),
            event_target_fractions=event_sampling_fractions,
        )
    finally:
        for group, original_lr in zip(optimizer.param_groups, original_learning_rates, strict=True):
            group["lr"] = original_lr

    predictions = _head_predictions(policy, observations[validation_indices], int(batch_size))
    validation = factorized_stage_gate(labels[validation_indices], predictions)
    report = {
        "dataset": str(Path(dataset_path).resolve()),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "event_loss_scale": float(event_loss_scale),
        "event_sampling_fractions": {str(k): float(v) for k, v in event_sampling_fractions.items()},
        "before_validation": before_validation,
        "training": training,
        "validation": validation,
    }
    setattr(model, "teacher_rehearsal_metadata", report)
    return report


def _single_head_gate(
    expected: np.ndarray,
    predicted: np.ndarray,
    *,
    name: str,
    required: tuple[int, ...],
    minimum_recall: float,
    maximum_single_fraction: float | None = None,
) -> dict[str, Any]:
    truth = np.asarray(expected, dtype=np.int64).reshape(-1)
    guesses = np.asarray(predicted, dtype=np.int64).reshape(-1)
    if truth.shape != guesses.shape or truth.size < 1:
        raise ValueError(f"{name} validation labels must align and be non-empty")
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (truth, guesses), 1)
    counts = confusion.sum(axis=0)
    reasons: list[str] = []
    per_class = []
    for value in range(3):
        support = int(confusion[value].sum())
        predicted_support = int(confusion[:, value].sum())
        tp = int(confusion[value, value])
        recall = float(tp / max(1, support))
        precision = float(tp / max(1, predicted_support))
        per_class.append({
            "value": value,
            "support": support,
            "predicted": predicted_support,
            "precision": precision,
            "recall": recall,
        })
        if value in required:
            if support == 0:
                reasons.append(f"validation has no {name}={value} examples")
            elif recall < minimum_recall:
                reasons.append(
                    f"{name}={value} recall {recall:.3f} is below {minimum_recall:.3f}"
                )
    maximum_fraction = float(counts.max() / truth.size)
    if maximum_single_fraction is not None and maximum_fraction > maximum_single_fraction:
        reasons.append(
            f"one {name} choice occupies {maximum_fraction:.3f} of predictions"
        )
    return {
        "passed": not reasons,
        "samples": int(truth.size),
        "accuracy": float(np.mean(truth == guesses)),
        "maximum_prediction_fraction": maximum_fraction,
        "predicted_counts": counts.tolist(),
        "confusion_matrix_true_rows": confusion.tolist(),
        "per_class": per_class,
        "reasons": reasons,
    }


def behavior_clone_factorized(
    model: Any,
    demonstrations_path: str | Path,
    *,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1.0e-4,
    seed: int = 0,
) -> dict[str, Any]:
    """Behavior-clone separate steering and event heads from recorded WASD data."""

    import torch

    with np.load(Path(demonstrations_path), allow_pickle=False) as data:
        observations = np.asarray(data["observations"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.int64)
        sessions = np.asarray(data["session_index"], dtype=np.int64)
        steering_valid = np.asarray(
            data.get("steering_label_valid", np.ones(len(actions), dtype=np.bool_)),
            dtype=np.bool_,
        )
        event_valid = np.asarray(
            data.get("event_label_valid", np.ones(len(actions), dtype=np.bool_)),
            dtype=np.bool_,
        )
        contract_ids = np.asarray(data.get("action_contract_id", []), dtype=str).reshape(-1)
        nvec = np.asarray(data.get("action_nvec", []), dtype=np.int64).reshape(-1)
    if observations.ndim != 2 or observations.shape[1] != 923:
        raise ValueError(f"Demonstrations must have shape (N, 923), got {observations.shape}")
    if actions.shape != (observations.shape[0], 2):
        raise ValueError(
            "Factorized demonstrations must have actions with shape (N, 2); "
            "re-export legacy archives with the v1.9 exporter"
        )
    if sessions.shape != (observations.shape[0],):
        raise ValueError("session_index must align with demonstrations")
    if steering_valid.shape != sessions.shape or event_valid.shape != sessions.shape:
        raise ValueError("demonstration label-valid masks must align with actions")
    if not np.any(steering_valid):
        raise ValueError("demonstrations contain no authoritative steering labels")
    if not np.any(event_valid):
        raise ValueError("demonstrations contain no event labels")
    if contract_ids.tolist() != ["latched-forward-factorized-steering-event-v1"]:
        raise ValueError("Demonstration action contract is missing or incompatible")
    if tuple(int(v) for v in nvec.tolist()) != (3, 3):
        raise ValueError(f"Demonstration action_nvec must be [3, 3], got {nvec.tolist()}")
    unique_sessions = np.unique(sessions)
    if len(unique_sessions) < 2:
        raise ValueError("Behavior cloning requires at least two independent sessions")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_sessions)
    validation_count = max(1, int(round(len(shuffled) * 0.2)))
    validation_sessions = shuffled[:validation_count]
    validation_mask = np.isin(sessions, validation_sessions)
    train_indices = np.flatnonzero(~validation_mask)
    validation_indices = np.flatnonzero(validation_mask)
    steering_train_indices = train_indices[steering_valid[train_indices]]
    event_train_indices = train_indices[event_valid[train_indices]]
    if steering_train_indices.size < 1 or event_train_indices.size < 1:
        raise ValueError("session split left one action head without training labels")
    steering_weights = _balanced_head_weights(actions[:, 0], steering_train_indices, 3)
    event_weights = _balanced_head_weights(actions[:, 1], event_train_indices, 3)
    steering_values = _training_values(
        actions[:, 0],
        steering_train_indices,
        required=(
            int(SteeringAction.STRAIGHT),
            int(SteeringAction.LEFT),
            int(SteeringAction.RIGHT),
        ),
    )
    event_values = _training_values(
        actions[:, 1],
        event_train_indices,
        required=(int(FarmingEvent.NONE), int(FarmingEvent.CAST_EVA)),
        optional=(int(FarmingEvent.JUMP),),
        minimum_optional_support=16,
    )

    event_sampling_fractions = _event_sampling_fractions(event_values)
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(learning_rate))
    training = _train_balanced_factorized_heads(
        policy,
        optimizer,
        observations,
        actions,
        steering_indices=steering_train_indices,
        event_indices=event_train_indices,
        steering_values=steering_values,
        event_values=event_values,
        epochs=int(epochs),
        batch_size=int(batch_size),
        event_loss_scale=1.5,
        rng=rng,
        event_target_fractions=event_sampling_fractions,
    )
    final_loss = float(training["mean_loss"])

    predictions = _head_predictions(policy, observations[validation_indices], int(batch_size))
    steering_validation_mask = steering_valid[validation_indices]
    event_validation_mask = event_valid[validation_indices]
    if not np.any(steering_validation_mask) or not np.any(event_validation_mask):
        raise ValueError("session split left one action head without validation labels")
    steering_validation = _single_head_gate(
        actions[validation_indices][steering_validation_mask, 0],
        predictions[steering_validation_mask, 0],
        name="steering",
        required=(0, 1, 2),
        minimum_recall=0.05,
        maximum_single_fraction=0.90,
    )
    event_validation = _single_head_gate(
        actions[validation_indices][event_validation_mask, 1],
        predictions[event_validation_mask, 1],
        name="event",
        required=(0, 1),
        minimum_recall=0.05,
    )
    validation = {
        "passed": steering_validation["passed"] and event_validation["passed"],
        "steering": steering_validation,
        "event": event_validation,
        "reasons": steering_validation["reasons"] + event_validation["reasons"],
    }
    if not validation["passed"]:
        raise ValueError("Factorized BC gate failed: " + "; ".join(validation["reasons"]))
    report = {
        "samples": int(len(observations)),
        "sessions": int(len(unique_sessions)),
        "training_samples": int(len(train_indices)),
        "validation_samples": int(len(validation_indices)),
        "authoritative_steering_samples": int(np.count_nonzero(steering_valid)),
        "event_samples": int(np.count_nonzero(event_valid)),
        "steering_counts": np.bincount(actions[:, 0], minlength=3).tolist(),
        "event_counts": np.bincount(actions[:, 1], minlength=3).tolist(),
        "steering_class_weights": steering_weights.tolist(),
        "event_class_weights": event_weights.tolist(),
        "balanced_steering_values": list(steering_values),
        "balanced_event_values": list(event_values),
        "event_sampling_fractions": {str(k): float(v) for k, v in event_sampling_fractions.items()},
        "epochs": int(epochs),
        "training": training,
        "final_loss": final_loss,
        "validation": validation,
    }
    setattr(model, "farming_contract_metadata", ModelContractMetadata.current().as_dict())
    setattr(model, "behavior_clone_metadata", report)
    return report
