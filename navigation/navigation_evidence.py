"""Canonical, environment-agnostic per-tick navigation evidence and the pure
computation of the 5-value policy-input sidecar
`[recent_progress, recent_contact, prev_straight, prev_left, prev_right]`.

`NavigationStepEvidence` is deliberately environment-agnostic: the simulator
populates it from its own info-dict keys, but a future live-bot port
populates the identical structure from its own native transition data
without needing to match the simulator's specific field names -- this is
the intended migration seam (simulator is canonical, a future live
implementation is refactored to match, not the reverse).

Window size and the `expected_clear_path_displacement` normalizer are
calibration outputs, frozen from the navigation_calibration pool (see
CALIBRATED_HISTORY_WINDOW / CALIBRATED_EXPECTED_CLEAR_PATH_DISPLACEMENT
below and evaluations/navigation_calibration_results.json) -- not guesses.

The calibrated constant-curvature-arc kernel makes `previous_steering` part
of the environment's Markov state (`navigation.movement_kernel.
resolve_signed_turn_radians` is stateful). A 3-way one-hot
(`prev_straight`/`prev_left`/`prev_right`) is appended alongside the 2
temporal values -- sidecar size is 5.

This module must not import gymnasium, stable_baselines3, or any
simulator training/env module -- see
tests/test_navigation_dependency_boundary.py.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .movement_kernel import SteeringDirection

STEERING_POLICY_INPUT_SCHEMA_ID = "steering-nav-sidecar-v2-928"
RAW_OBSERVATION_SIZE = 923
TEMPORAL_SIDECAR_SIZE = 2
PREVIOUS_STEERING_SIDECAR_SIZE = 3
SIDECAR_SIZE = TEMPORAL_SIDECAR_SIZE + PREVIOUS_STEERING_SIDECAR_SIZE
POLICY_INPUT_SIZE = RAW_OBSERVATION_SIZE + SIDECAR_SIZE

# Frozen calibration outputs from the navigation_calibration pool (12
# layouts x 6 templates x 5 seeds, 15k raw policy; see
# evaluations/navigation_calibration_results.json). expected_clear_path_
# displacement is the median per-tick displacement on ticks with fully-open
# clearance (n=23326, median=1.7898 and 1.7920 across two independent runs
# -- robust). history_window=15 matches the contact-rate window used by the
# DAgger mining thresholds in navigation_dataset.MiningConfig, for
# consistency between what the steering feature sees and what mining uses
# to categorize the same signal.
CALIBRATED_HISTORY_WINDOW = 15
CALIBRATED_EXPECTED_CLEAR_PATH_DISPLACEMENT = 1.79


def previous_steering_one_hot(previous_steering: "SteeringDirection | int") -> np.ndarray:
    """Pure function computing the 3-way [prev_straight, prev_left,
    prev_right] one-hot from a SteeringDirection (or its plain int value,
    as arrives via info["previous_steering"]). This is an INSTANTANEOUS
    state read (what steering was active going into the tick that just
    produced this observation), not a windowed statistic like the
    temporal sidecar values -- deliberately a separate function rather
    than folded into the history-windowing logic below."""

    direction = int(previous_steering)
    one_hot = np.zeros((PREVIOUS_STEERING_SIDECAR_SIZE,), dtype=np.float32)
    if 0 <= direction < PREVIOUS_STEERING_SIDECAR_SIZE:
        one_hot[direction] = 1.0
    else:
        raise ValueError(f"previous_steering must be 0 (NONE), 1 (LEFT), or 2 (RIGHT); got {direction}")
    return one_hot


@dataclass(frozen=True, slots=True)
class NavigationStepEvidence:
    """One tick's raw navigation evidence, environment-agnostic.

    `displacement_cells` and `contact` must come from the same raw,
    undecoded per-tick quantities already used by RecoveryController and
    milestone_evaluator (info-dict diffs), never from the observation's
    bipolar-encoded fields.
    """

    displacement_cells: float
    contact: bool
    eva_attempted: bool


def sidecar_values_from_history(
    history: "deque[NavigationStepEvidence] | list[NavigationStepEvidence]",
    previous_steering: "SteeringDirection | int",
    *,
    expected_clear_path_displacement: float = CALIBRATED_EXPECTED_CLEAR_PATH_DISPLACEMENT,
) -> np.ndarray:
    """Pure function computing the full 5-value sidecar
    [recent_progress, recent_contact, prev_straight, prev_left,
    prev_right] from a window of NavigationStepEvidence plus the current
    previous_steering state. The single source of truth for this
    computation -- NavigationHistoryWrapper._sidecar_values (live rollout
    collection) and any offline reconstruction (e.g. from recordings) must
    both call this rather than reimplementing the windowing/EVA-exclusion/
    one-hot logic separately, so they can never silently drift apart."""

    eligible = [e for e in history if not e.eva_attempted]
    if not eligible:
        temporal = np.zeros((TEMPORAL_SIDECAR_SIZE,), dtype=np.float32)
    else:
        recent_progress = float(
            np.clip(
                np.mean([e.displacement_cells for e in eligible]) / expected_clear_path_displacement,
                0.0,
                1.0,
            )
        )
        recent_contact = float(np.mean([1.0 if e.contact else 0.0 for e in eligible]))
        temporal = np.asarray([recent_progress, recent_contact], dtype=np.float32)
    return np.concatenate([temporal, previous_steering_one_hot(previous_steering)])
