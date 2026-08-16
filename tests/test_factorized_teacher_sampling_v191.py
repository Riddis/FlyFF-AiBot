from __future__ import annotations

import numpy as np

from simulator.factorized_training import (
    _balanced_head_weights,
    _balanced_resample_indices,
    _training_values,
)


def test_balanced_resample_equalizes_required_event_classes() -> None:
    labels = np.asarray([0] * 90 + [1] * 10, dtype=np.int64)
    indices = np.arange(len(labels), dtype=np.int64)

    sampled = _balanced_resample_indices(
        labels,
        indices,
        (0, 1),
        np.random.default_rng(7),
    )

    counts = np.bincount(labels[sampled], minlength=2)
    assert counts.tolist() == [90, 90]
    assert set(sampled.tolist()).issubset(set(indices.tolist()))


def test_training_values_requires_core_events_and_ignores_tiny_jump_class() -> None:
    labels = np.asarray([0] * 90 + [1] * 20 + [2] * 3, dtype=np.int64)
    indices = np.arange(len(labels), dtype=np.int64)

    values = _training_values(
        labels,
        indices,
        required=(0, 1),
        optional=(2,),
        minimum_optional_support=16,
    )

    assert values == (0, 1)


def test_training_values_includes_supported_optional_jump_class() -> None:
    labels = np.asarray([0] * 90 + [1] * 20 + [2] * 16, dtype=np.int64)
    indices = np.arange(len(labels), dtype=np.int64)

    values = _training_values(
        labels,
        indices,
        required=(0, 1),
        optional=(2,),
        minimum_optional_support=16,
    )

    assert values == (0, 1, 2)


def test_diagnostic_weights_give_rare_class_more_weight() -> None:
    labels = np.asarray([0] * 90 + [1] * 10, dtype=np.int64)
    indices = np.arange(len(labels), dtype=np.int64)

    weights = _balanced_head_weights(labels, indices, 3)

    assert weights[1] > weights[0]
    assert weights[2] == 0.0


def test_balanced_head_training_learns_rare_event_without_argmax_collapse() -> None:
    import torch
    from torch import nn
    from torch.distributions import Categorical
    from types import SimpleNamespace

    from simulator.factorized_training import (
        _head_predictions,
        _train_balanced_factorized_heads,
    )

    class TinyPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action_net = nn.Linear(2, 6)
            self.device = torch.device("cpu")

        def get_distribution(self, observations):
            logits = self.action_net(observations.float())
            return SimpleNamespace(
                distribution=(
                    Categorical(logits=logits[:, :3]),
                    Categorical(logits=logits[:, 3:]),
                )
            )

    rng = np.random.default_rng(11)
    observations = rng.normal(size=(1000, 2)).astype(np.float32)
    steering = np.where(
        observations[:, 1] < -0.4,
        1,
        np.where(observations[:, 1] > 0.4, 2, 0),
    ).astype(np.int64)
    # Roughly ten percent EVA examples, linearly separable from NONE.
    event = (observations[:, 0] > 1.25).astype(np.int64)
    labels = np.column_stack((steering, event)).astype(np.int64)
    indices = np.arange(len(labels), dtype=np.int64)

    policy = TinyPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.02)
    _train_balanced_factorized_heads(
        policy,
        optimizer,
        observations,
        labels,
        steering_indices=indices,
        event_indices=indices,
        steering_values=(0, 1, 2),
        event_values=(0, 1),
        epochs=8,
        batch_size=128,
        event_loss_scale=1.5,
        rng=np.random.default_rng(13),
    )

    predictions = _head_predictions(policy, observations, 256)
    eva_mask = event == 1
    eva_recall = float(np.mean(predictions[eva_mask, 1] == 1))
    assert eva_recall >= 0.8
    assert np.count_nonzero(predictions[:, 1] == 1) > 0
