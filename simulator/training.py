from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np

from farming.model_contract import (
    ModelContractMetadata,
    ModelSpaceSignature,
    validate_model_contract,
)


def action_stage_gate(
    expected: np.ndarray,
    predicted: np.ndarray,
    *,
    maximum_single_action_fraction: float = 0.90,
    minimum_required_recall: float = 0.05,
    required_actions: tuple[int, ...] = (0, 1, 2, 3),
) -> dict[str, Any]:
    """Report whether held-out predictions contain useful action coverage."""

    truth = np.asarray(expected, dtype=np.int64).reshape(-1)
    guesses = np.asarray(predicted, dtype=np.int64).reshape(-1)
    if truth.shape != guesses.shape or truth.size < 1:
        raise ValueError("Stage-gate predictions must align and be non-empty")
    if np.any((truth < 0) | (truth >= 5)) or np.any((guesses < 0) | (guesses >= 5)):
        raise ValueError("Stage-gate actions must be within [0, 4]")

    confusion = np.zeros((5, 5), dtype=np.int64)
    np.add.at(confusion, (truth, guesses), 1)
    predicted_counts = confusion.sum(axis=0)
    maximum_fraction = float(predicted_counts.max() / truth.size)
    reasons: list[str] = []
    if maximum_fraction > float(maximum_single_action_fraction):
        reasons.append(
            f"one predicted action occupies {maximum_fraction:.3f} of validation "
            f"samples (limit {maximum_single_action_fraction:.3f})"
        )

    per_action: list[dict[str, float | int]] = []
    for action in range(5):
        support = int(confusion[action].sum())
        predicted_support = int(confusion[:, action].sum())
        true_positive = int(confusion[action, action])
        recall = float(true_positive / max(1, support))
        precision = float(true_positive / max(1, predicted_support))
        per_action.append(
            {
                "action": action,
                "support": support,
                "predicted": predicted_support,
                "precision": precision,
                "recall": recall,
            }
        )
        if action in required_actions:
            if support == 0:
                reasons.append(f"validation has no action {action} examples")
            elif recall < float(minimum_required_recall):
                reasons.append(
                    f"action {action} recall {recall:.3f} is below "
                    f"{minimum_required_recall:.3f}"
                )

    return {
        "passed": not reasons,
        "samples": int(truth.size),
        "accuracy": float(np.mean(truth == guesses)),
        "maximum_single_action_fraction": maximum_fraction,
        "predicted_action_counts": predicted_counts.astype(int).tolist(),
        "confusion_matrix_true_rows": confusion.astype(int).tolist(),
        "per_action": per_action,
        "reasons": reasons,
    }


def validate_policy_contract(model: Any) -> None:
    """Reject a simulator checkpoint before it can be resumed or compared."""

    metadata = getattr(model, "farming_contract_metadata", None)
    if not isinstance(metadata, Mapping):
        raise ValueError(
            "Simulator checkpoint is missing farming_contract_metadata; "
            "refusing to relabel an unknown policy as the current contract"
        )
    validate_model_contract(
        ModelSpaceSignature.from_spaces(
            getattr(model, "observation_space", None),
            getattr(model, "action_space", None),
        ),
        metadata=metadata,
    )


def atomic_save_policy(model: Any, path: str | Path) -> Path:
    """Publish a complete SB3 ZIP without overwriting the last good file."""

    requested = Path(path)
    target = requested if requested.suffix.lower() == ".zip" else Path(f"{requested}.zip")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.stem}.{uuid4().hex}.tmp.zip"
    )
    try:
        model.save(str(temporary))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def behavior_clone(
    model: Any,
    demonstrations_path: str | Path,
    *,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1.0e-4,
    seed: int = 0,
) -> dict[str, Any]:
    """Supervise the PPO policy on recorded human actions before RL.

    This intentionally trains only from the action log-probability. PPO's value
    function remains free to learn from simulated returns during the later RL
    phase. The saved checkpoint remains a normal Stable-Baselines3 PPO model.
    """

    import torch

    with np.load(Path(demonstrations_path), allow_pickle=False) as data:
        required_metadata = {"observation_schema_id", "observation_schema_hash"}
        missing = required_metadata.difference(data.files)
        if missing:
            raise ValueError(
                "Demonstrations are missing contract metadata: "
                + ", ".join(sorted(missing))
            )
        schema_ids = np.asarray(data["observation_schema_id"]).reshape(-1)
        schema_hashes = np.asarray(data["observation_schema_hash"]).reshape(-1)
        expected = ModelContractMetadata.current()
        if schema_ids.tolist() != [expected.observation_schema_id]:
            raise ValueError(
                "Demonstration observation schema mismatch: "
                f"got {schema_ids.tolist()}, expected "
                f"[{expected.observation_schema_id!r}]"
            )
        if [str(value).upper() for value in schema_hashes.tolist()] != [
            expected.observation_schema_hash
        ]:
            raise ValueError(
                "Demonstration observation schema hash does not match the "
                "current farming contract"
            )
        observations = np.asarray(data["observations"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.int64)
        if "session_index" not in data.files:
            raise ValueError(
                "Demonstrations are missing session_index; session-isolated "
                "validation is required"
            )
        session_index = np.asarray(data["session_index"], dtype=np.int64)
    if observations.ndim != 2 or observations.shape[1] != 923:
        raise ValueError(
            f"Demonstrations must have shape (N, 923), got {observations.shape}"
        )
    if actions.shape != (observations.shape[0],):
        raise ValueError("Demonstration actions must align with observations")
    if session_index.shape != actions.shape:
        raise ValueError("Demonstration session_index must align with actions")
    if observations.shape[0] < 1:
        raise ValueError("Demonstration dataset is empty")
    if np.any((actions < 0) | (actions >= 5)):
        raise ValueError("Demonstration actions must be within [0, 4]")

    unique_sessions = np.unique(session_index)
    if unique_sessions.size < 2:
        raise ValueError(
            "Behavior cloning requires at least two independent recording sessions "
            "for leakage-free validation; this dataset contains "
            f"{unique_sessions.size}"
        )
    split_rng = np.random.default_rng(seed)
    shuffled_sessions = split_rng.permutation(unique_sessions)
    validation_session_count = max(1, int(round(unique_sessions.size * 0.20)))
    validation_sessions = shuffled_sessions[:validation_session_count]
    validation_mask = np.isin(session_index, validation_sessions)
    training_mask = ~validation_mask
    training_indices = np.flatnonzero(training_mask)
    validation_indices = np.flatnonzero(validation_mask)
    if training_indices.size < 1 or validation_indices.size < 1:
        raise ValueError("Session-isolated train/validation split is empty")

    policy = model.policy
    device = policy.device
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(learning_rate))
    generator = np.random.default_rng(seed)
    final_loss = 0.0
    final_accuracy = 0.0
    sample_count = int(observations.shape[0])
    training_counts = np.bincount(actions[training_indices], minlength=5).astype(np.float64)
    class_weights = np.zeros(5, dtype=np.float32)
    present = training_counts > 0
    class_weights[present] = np.sqrt(training_indices.size / training_counts[present])
    class_weights[present] /= np.mean(class_weights[present])
    weight_tensor = torch.as_tensor(class_weights, device=device)

    policy.train()
    for _epoch in range(int(epochs)):
        order = generator.permutation(training_indices)
        correct = 0
        total = 0
        losses: list[float] = []
        for start in range(0, training_indices.size, int(batch_size)):
            indices = order[start : start + int(batch_size)]
            obs_tensor = torch.as_tensor(observations[indices], device=device)
            action_tensor = torch.as_tensor(actions[indices], device=device)
            _values, log_prob, _entropy = policy.evaluate_actions(
                obs_tensor,
                action_tensor,
            )
            loss = -(log_prob * weight_tensor[action_tensor]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                distribution = policy.get_distribution(obs_tensor)
                predicted = distribution.distribution.probs.argmax(dim=1)
                correct += int((predicted == action_tensor).sum().item())
                total += int(action_tensor.numel())
            losses.append(float(loss.item()))
        final_loss = float(np.mean(losses)) if losses else 0.0
        final_accuracy = float(correct / max(1, total))

    validation_predictions: list[int] = []
    policy.eval()
    with torch.no_grad():
        for start in range(0, validation_indices.size, int(batch_size)):
            indices = validation_indices[start : start + int(batch_size)]
            obs_tensor = torch.as_tensor(observations[indices], device=device)
            distribution = policy.get_distribution(obs_tensor)
            predicted = distribution.distribution.probs.argmax(dim=1)
            validation_predictions.extend(int(value) for value in predicted.cpu().numpy())
    validation_report = action_stage_gate(
        actions[validation_indices], np.asarray(validation_predictions, dtype=np.int64)
    )
    if not validation_report["passed"]:
        raise ValueError(
            "Behavior-cloning validation stage gate failed: "
            + "; ".join(str(value) for value in validation_report["reasons"])
        )

    setattr(model, "farming_contract_metadata", ModelContractMetadata.current().as_dict())
    setattr(
        model,
        "simulator_imitation_metadata",
        {
            "demonstrations": str(Path(demonstrations_path).resolve()),
            "samples": sample_count,
            "training_samples": int(training_indices.size),
            "validation_samples": int(validation_indices.size),
            "training_sessions": [
                int(value) for value in np.unique(session_index[training_indices])
            ],
            "validation_sessions": [int(value) for value in validation_sessions],
            "class_weights": class_weights.tolist(),
            "validation": validation_report,
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "final_loss": final_loss,
            "final_accuracy": final_accuracy,
        },
    )
    return {
        "samples": sample_count,
        "training_samples": int(training_indices.size),
        "validation_samples": int(validation_indices.size),
        "epochs": int(epochs),
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "validation": validation_report,
    }


def synthetic_teacher_clone(
    model: Any,
    curriculum_path: str | Path,
    *,
    stage: str = "early",
    samples: int = 4_000,
    episode_seconds: float = 30.0,
    max_actions: int = 220,
    teacher_policy: str = "obstacle_aware",
    epochs: int = 8,
    batch_size: int = 256,
    learning_rate: float = 3.0e-4,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap a PPO actor from a feasible scripted synthetic farmer.

    PPO from a random policy must solve two coordinated discoveries: move into a
    useful group and select EVA at the brief moment when targets are in range.
    The previous pilot instead learned that exploratory EVA often incurs an
    immediate penalty and collapsed to movement-only behavior. This bootstrap
    teaches only the action mapping on generated states; PPO still learns the
    value function and may improve the strategy afterward.
    """

    import torch

    from .scripted_policies import scripted_action
    from .synthetic import iter_variant_environments

    requested = int(samples)
    if requested < 500:
        raise ValueError("synthetic teacher bootstrap requires at least 500 samples")
    if not np.isfinite(float(episode_seconds)) or float(episode_seconds) <= 0.0:
        raise ValueError("episode_seconds must be finite and positive")
    if int(max_actions) < 1:
        raise ValueError("max_actions must be positive")

    environments = list(
        iter_variant_environments(
            curriculum_path,
            stage=stage,
            seed=seed,
            episode_steps=int(max_actions),
            episode_seconds=float(episode_seconds),
        )
    )
    if not environments:
        raise ValueError("No synthetic variants matched the teacher bootstrap stage")

    observations: list[np.ndarray] = []
    actions: list[int] = []
    episode_index: list[int] = []
    episode_variants: list[str] = []
    episode = 0
    hard_limit = max(requested * 2, requested + 2_000)
    try:
        while len(observations) < requested and len(observations) < hard_limit:
            for variant_offset, (entry, env) in enumerate(environments):
                episode_seed = int(seed + episode * 1009 + variant_offset * 37)
                observation, _info = env.reset(seed=episode_seed)
                episode_variants.append(entry.name)
                for _step in range(int(max_actions)):
                    action = int(scripted_action(teacher_policy, env))
                    observations.append(np.asarray(observation, dtype=np.float32).copy())
                    actions.append(action)
                    episode_index.append(episode)
                    observation, _reward, terminated, truncated, _info = env.step(action)
                    if len(observations) >= requested or terminated or truncated:
                        break
                episode += 1
                if len(observations) >= requested:
                    break
    finally:
        for _entry, env in environments:
            env.close()

    observation_array = np.asarray(observations[:requested], dtype=np.float32)
    action_array = np.asarray(actions[:requested], dtype=np.int64)
    episode_array = np.asarray(episode_index[:requested], dtype=np.int64)
    if observation_array.shape != (requested, 923):
        raise ValueError(
            f"Synthetic teacher observations must have shape ({requested}, 923), "
            f"got {observation_array.shape}"
        )
    required_actions = (0, 1, 2, 3)
    counts = np.bincount(action_array, minlength=5)
    missing = [action for action in required_actions if counts[action] < 16]
    if missing:
        raise ValueError(
            "Synthetic teacher did not produce enough examples for actions: "
            + ", ".join(str(action) for action in missing)
        )

    unique_episodes = np.unique(episode_array)
    if unique_episodes.size < 5:
        raise ValueError("Synthetic teacher bootstrap requires at least five episodes")
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(unique_episodes))
    validation_count = max(1, int(round(len(shuffled) * 0.20)))
    validation_episodes: set[int] = {int(value) for value in shuffled[:validation_count]}

    # Ensure every required action is represented in held-out episodes without
    # splitting adjacent states from the same episode across train/validation.
    for required in required_actions:
        validation_mask = np.isin(episode_array, tuple(validation_episodes))
        if np.any(action_array[validation_mask] == required):
            continue
        candidate_episode = next(
            (
                int(value)
                for value in shuffled[validation_count:]
                if np.any(
                    action_array[episode_array == int(value)] == required
                )
            ),
            None,
        )
        if candidate_episode is None:
            raise ValueError(
                f"No complete teacher episode contains required action {required}"
            )
        validation_episodes.add(candidate_episode)

    validation_mask = np.isin(episode_array, tuple(validation_episodes))
    training_mask = ~validation_mask
    training_indices = np.flatnonzero(training_mask)
    validation_indices = np.flatnonzero(validation_mask)
    if training_indices.size < 1 or validation_indices.size < 1:
        raise ValueError("Synthetic teacher train/validation split is empty")

    policy = model.policy
    device = policy.device
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(learning_rate))
    training_counts = np.bincount(
        action_array[training_indices], minlength=5
    ).astype(np.float64)
    class_weights = np.zeros(5, dtype=np.float32)
    present = training_counts > 0
    class_weights[present] = np.sqrt(training_indices.size / training_counts[present])
    class_weights[present] /= np.mean(class_weights[present])
    weight_tensor = torch.as_tensor(class_weights, device=device)

    final_loss = 0.0
    final_accuracy = 0.0
    policy.train()
    for _epoch in range(int(epochs)):
        order = rng.permutation(training_indices)
        losses: list[float] = []
        correct = 0
        total = 0
        for start in range(0, training_indices.size, int(batch_size)):
            indices = order[start : start + int(batch_size)]
            obs_tensor = torch.as_tensor(observation_array[indices], device=device)
            action_tensor = torch.as_tensor(action_array[indices], device=device)
            _values, log_prob, _entropy = policy.evaluate_actions(
                obs_tensor, action_tensor
            )
            loss = -(log_prob * weight_tensor[action_tensor]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                distribution = policy.get_distribution(obs_tensor)
                predicted = distribution.distribution.probs.argmax(dim=1)
                correct += int((predicted == action_tensor).sum().item())
                total += int(action_tensor.numel())
            losses.append(float(loss.item()))
        final_loss = float(np.mean(losses)) if losses else 0.0
        final_accuracy = float(correct / max(1, total))

    predictions: list[int] = []
    policy.eval()
    with torch.no_grad():
        for start in range(0, validation_indices.size, int(batch_size)):
            indices = validation_indices[start : start + int(batch_size)]
            obs_tensor = torch.as_tensor(observation_array[indices], device=device)
            distribution = policy.get_distribution(obs_tensor)
            predicted = distribution.distribution.probs.argmax(dim=1)
            predictions.extend(int(value) for value in predicted.cpu().numpy())
    validation = action_stage_gate(
        action_array[validation_indices],
        np.asarray(predictions, dtype=np.int64),
        maximum_single_action_fraction=0.85,
        minimum_required_recall=0.20,
        required_actions=required_actions,
    )
    if not validation["passed"]:
        raise ValueError(
            "Synthetic teacher bootstrap stage gate failed: "
            + "; ".join(str(value) for value in validation["reasons"])
        )

    report = {
        "teacher_policy": str(teacher_policy),
        "curriculum": str(Path(curriculum_path).resolve()),
        "stage": str(stage),
        "samples": int(requested),
        "episodes": int(np.unique(episode_array).size),
        "action_counts": counts.astype(int).tolist(),
        "training_samples": int(training_indices.size),
        "validation_samples": int(validation_indices.size),
        "validation_episodes": sorted(validation_episodes),
        "class_weights": class_weights.tolist(),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "validation": validation,
    }
    setattr(model, "synthetic_teacher_metadata", report)
    return report
