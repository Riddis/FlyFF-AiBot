from __future__ import annotations

import numpy as np

from simulator.factorized_cli import _policy_gate_reasons
from simulator.factorized_training import (
    _event_sampling_fractions,
    _fractional_resample_indices,
)


def test_event_resampling_keeps_jump_at_five_percent() -> None:
    labels = np.asarray([0] * 900 + [1] * 90 + [2] * 20, dtype=np.int64)
    indices = np.arange(len(labels), dtype=np.int64)
    values = (0, 1, 2)
    fractions = _event_sampling_fractions(values)

    sampled = _fractional_resample_indices(
        labels,
        indices,
        values,
        fractions,
        np.random.default_rng(5),
        anchor_values=(0, 1),
    )

    counts = np.bincount(labels[sampled], minlength=3)
    probabilities = counts / counts.sum()
    assert 0.53 <= probabilities[0] <= 0.57
    assert 0.38 <= probabilities[1] <= 0.42
    assert 0.04 <= probabilities[2] <= 0.06
    assert probabilities[2] < probabilities[1]


def test_event_resampling_without_jump_keeps_none_majority() -> None:
    labels = np.asarray([0] * 900 + [1] * 90, dtype=np.int64)
    indices = np.arange(len(labels), dtype=np.int64)
    values = (0, 1)
    fractions = _event_sampling_fractions(values)

    sampled = _fractional_resample_indices(
        labels,
        indices,
        values,
        fractions,
        np.random.default_rng(7),
        anchor_values=(0, 1),
    )

    counts = np.bincount(labels[sampled], minlength=2)
    probabilities = counts / counts.sum()
    assert 0.53 <= probabilities[0] <= 0.57
    assert 0.43 <= probabilities[1] <= 0.47


def _report(
    *,
    kills_per_hour: float,
    valid_eva: float,
    steering: tuple[float, float, float],
    event: tuple[float, float, float],
    contacts: float = 0.0,
) -> dict[str, object]:
    return {
        "mean_kills": kills_per_hour / 60.0,
        "kills_per_simulated_hour": kills_per_hour,
        "mean_valid_eva_casts": valid_eva,
        "mean_contacts": contacts,
        "steering_probabilities": {str(i): value for i, value in enumerate(steering)},
        "event_probabilities": {str(i): value for i, value in enumerate(event)},
    }


def test_per_layout_gate_rejects_directional_and_jump_collapse() -> None:
    layouts: list[dict[str, object]] = [
        {
            "variant": "healthy",
            "random": _report(
                kills_per_hour=100.0,
                valid_eva=2.0,
                steering=(0.34, 0.33, 0.33),
                event=(0.90, 0.08, 0.02),
            ),
            "policy": _report(
                kills_per_hour=1000.0,
                valid_eva=3.0,
                steering=(0.40, 0.30, 0.30),
                event=(0.93, 0.05, 0.02),
            ),
        },
        {
            "variant": "collapsed",
            "random": _report(
                kills_per_hour=100.0,
                valid_eva=2.0,
                steering=(0.34, 0.33, 0.33),
                event=(0.90, 0.08, 0.02),
            ),
            "policy": _report(
                kills_per_hour=0.0,
                valid_eva=0.0,
                steering=(0.99, 0.0, 0.01),
                event=(0.04, 0.0, 0.96),
                contacts=250.0,
            ),
        },
    ]

    reasons, *_ = _policy_gate_reasons(layouts)

    assert reasons
    assert layouts[0]["gate"]["passed"] is True
    assert layouts[1]["gate"]["passed"] is False
    joined = " ".join(layouts[1]["gate"]["reasons"])
    assert "steering choice reached" in joined
    assert "jump fraction" in joined
    assert "no valid EVA" in joined
    assert "high-contact zero-kill" in joined


def test_per_layout_gate_accepts_diverse_farming_policy() -> None:
    layouts: list[dict[str, object]] = []
    for index in range(4):
        layouts.append(
            {
                "variant": f"layout-{index}",
                "random": _report(
                    kills_per_hour=100.0,
                    valid_eva=2.0,
                    steering=(0.34, 0.33, 0.33),
                    event=(0.90, 0.08, 0.02),
                ),
                "policy": _report(
                    kills_per_hour=800.0 + index * 50.0,
                    valid_eva=3.0,
                    steering=(0.45, 0.25, 0.30),
                    event=(0.92, 0.06, 0.02),
                ),
            }
        )

    reasons, steering, event, _random_kph, policy_kph, ratio = _policy_gate_reasons(layouts)

    assert reasons == []
    assert policy_kph > 0.0
    assert ratio > 1.0
    assert min(steering.values()) >= 0.005
    assert event["1"] > 0.0
    assert all(item["gate"]["passed"] for item in layouts)
