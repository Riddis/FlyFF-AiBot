"""Deterministic geometry-feature transform for the steering branch.

The dagger_v193 rollout diagnostic and the observation-only-teacher probe
established that ``best_group_relative_angle()`` is fully recoverable from
the 923-value observation -- it just requires combining each visible actor's
world-frame offset (``dx_over_vision``/``dz_over_vision`` in the direct-actor
block) with the player's own heading (``heading_sin``/``heading_cos`` in the
unified-state block) through an actual rotation, something a plain linear
head cannot represent and a shared MLP evidently did not reliably learn
across layouts. This module computes that rotation explicitly and exposes a
small, fixed-size, layout-invariant feature vector a steering-only branch can
consume directly, without touching the serialized 923-value observation
contract, the recorder schema, or the live bot's observation source -- the
live bot can compute the identical transform from the same raw fields it
already emits.

Feature layout (6 values, in this order):
    0: best_group_angle_sin
    1: best_group_angle_cos
    2: best_group_distance (unit [0, 1], vision-radius normalized)
    3: best_group_density (unit [0, 1], pack-density normalized)
    4: eva_target_count (unit [0, 1], direct-actor-slot count normalized)
    5: has_target (bipolar: 1.0 if any candidate was found, else -1.0)

When ``has_target`` is -1.0, fields 0-4 are all zero -- there is no visible
group to steer toward, and callers must not treat sin=cos=0 as "angle zero"
in that case.
"""

from __future__ import annotations

import numpy as np

DIRECT_ACTOR_SLOTS = 12
ACTOR_FEATURES = 7
DIRECT_ACTOR_START = 839
UNIFIED_STATE_START = 261
HEADING_SIN_INDEX = UNIFIED_STATE_START + 2
HEADING_COS_INDEX = UNIFIED_STATE_START + 3
VISION_RADIUS_CELLS = 50.0
MAXIMUM_PACK_DENSITY = 24.0
GEOMETRY_FEATURE_SIZE = 6


def _score_and_select(dx: np.ndarray, dz: np.ndarray, density_bipolar: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Return the index (per row) of the best active candidate, or -1."""

    distance = np.hypot(dx, dz)
    density = (density_bipolar + 1.0) / 2.0 * MAXIMUM_PACK_DENSITY
    score = 0.75 * np.clip(density - 1.0, 0.0, 12.0) - distance
    score = np.where(active > 0.5, score, -np.inf)
    best = np.argmax(score, axis=1)
    none_active = ~np.any(active > 0.5, axis=1)
    best = np.where(none_active, -1, best)
    return best


def derive_geometry_features(observations: np.ndarray) -> np.ndarray:
    """Compute the 6 geometry features for one or many 923-value observations.

    Accepts shape (923,) or (N, 923); returns shape (6,) or (N, 6) to match.
    """

    single = observations.ndim == 1
    obs = np.atleast_2d(observations).astype(np.float64)
    if obs.shape[1] < DIRECT_ACTOR_START + DIRECT_ACTOR_SLOTS * ACTOR_FEATURES:
        raise ValueError("observations do not contain a direct-actor block of the expected size")

    heading_sin = obs[:, HEADING_SIN_INDEX]
    heading_cos = obs[:, HEADING_COS_INDEX]
    heading = np.arctan2(heading_sin, heading_cos)

    block = obs[:, DIRECT_ACTOR_START : DIRECT_ACTOR_START + DIRECT_ACTOR_SLOTS * ACTOR_FEATURES]
    block = block.reshape(obs.shape[0], DIRECT_ACTOR_SLOTS, ACTOR_FEATURES)
    dx_over_vision = block[:, :, 0]
    dz_over_vision = block[:, :, 1]
    active = block[:, :, 3]
    within_eva = block[:, :, 4]
    density_bipolar = block[:, :, 6]

    dx = dx_over_vision * VISION_RADIUS_CELLS
    dz = dz_over_vision * VISION_RADIUS_CELLS

    best_index = _score_and_select(dx, dz, density_bipolar, active)
    has_target = best_index >= 0
    rows = np.arange(obs.shape[0])
    safe_index = np.clip(best_index, 0, DIRECT_ACTOR_SLOTS - 1)

    best_dx = dx[rows, safe_index]
    best_dz = dz[rows, safe_index]
    world_angle = np.arctan2(best_dz, best_dx)
    relative_angle = np.arctan2(np.sin(world_angle - heading), np.cos(world_angle - heading))

    distance = np.hypot(best_dx, best_dz)
    distance_unit = np.clip(distance / VISION_RADIUS_CELLS, 0.0, 1.0)
    density_unit = np.clip((density_bipolar[rows, safe_index] + 1.0) / 2.0, 0.0, 1.0)
    eva_count_unit = np.clip(np.sum(within_eva > 0.0, axis=1) / float(DIRECT_ACTOR_SLOTS), 0.0, 1.0)

    features = np.zeros((obs.shape[0], GEOMETRY_FEATURE_SIZE), dtype=np.float32)
    features[:, 0] = np.where(has_target, np.sin(relative_angle), 0.0)
    features[:, 1] = np.where(has_target, np.cos(relative_angle), 0.0)
    features[:, 2] = np.where(has_target, distance_unit, 0.0)
    features[:, 3] = np.where(has_target, density_unit, 0.0)
    features[:, 4] = eva_count_unit
    features[:, 5] = np.where(has_target, 1.0, -1.0)

    return features[0] if single else features


def steering_from_geometry_features(features: np.ndarray, *, straight_threshold: float = 0.18) -> int:
    """Reference steering decision from a single geometry-feature vector.

    Mirrors scripted_policies._steering_for_angle's threshold convention:
    STRAIGHT within +/-``straight_threshold`` radians, else LEFT (angle > 0)
    or RIGHT (angle < 0). Returns STRAIGHT when no target is visible.
    """

    if features[5] < 0.0:
        return 0
    angle = float(np.arctan2(features[0], features[1]))
    if abs(angle) <= straight_threshold:
        return 0
    return 1 if angle > 0.0 else 2
