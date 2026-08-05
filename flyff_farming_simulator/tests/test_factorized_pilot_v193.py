from __future__ import annotations

import numpy as np

from simulator.factorized_v193_training import (
    _apply_prior_bias_correction,
    _layout_stratified_episode_split,
)


def test_layout_stratified_split_covers_every_layout_and_keeps_episodes_intact() -> None:
    episode_index = np.repeat(np.arange(12, dtype=np.int64), 10)
    layout_index = np.repeat(np.repeat(np.arange(4, dtype=np.int64), 3), 10)
    labels = np.zeros((120, 2), dtype=np.int64)
    labels[:, 0] = np.tile([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], 12)
    labels[:, 1] = np.tile([0, 0, 1, 0, 0, 1, 0, 0, 0, 0], 12)

    train, validation, validation_episodes = _layout_stratified_episode_split(
        episode_index,
        layout_index,
        labels,
        validation_fraction=0.2,
        seed=4,
    )

    assert set(layout_index[validation].tolist()) == {0, 1, 2, 3}
    assert not set(episode_index[train]).intersection(set(episode_index[validation]))
    assert set(validation_episodes) == set(episode_index[validation].tolist())


def test_prior_bias_correction_reduces_oversampled_eva_prior() -> None:
    import torch
    from torch import nn

    class TinyPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action_net = nn.Linear(2, 6)
            nn.init.zeros_(self.action_net.bias)

    policy = TinyPolicy()
    report = _apply_prior_bias_correction(
        policy,
        steering_target=np.asarray([0.60, 0.20, 0.20]),
        event_target=np.asarray([0.90, 0.08, 0.02]),
        event_sampling_fractions={0: 0.55, 1: 0.40, 2: 0.05},
    )

    assert report["applied"] is True
    bias = policy.action_net.bias.detach().cpu().numpy()
    assert bias[3] > bias[4]  # NONE moves above EVA.
    assert bias[4] < 0.0      # Oversampled EVA receives a negative correction.
    assert bias[0] > bias[1]  # Natural straight prior is restored.
