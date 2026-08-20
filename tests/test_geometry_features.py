from __future__ import annotations

import numpy as np

from simulator.geometry_features import (
    DIRECT_ACTOR_START,
    GEOMETRY_FEATURE_SIZE,
    HEADING_COS_INDEX,
    HEADING_SIN_INDEX,
    derive_geometry_features,
    steering_from_geometry_features,
)
from simulator.synthetic import iter_variant_environments

_CURRICULUM = "curricula/synthetic_curriculum/curriculum.json"


def test_derive_geometry_features_matches_environment_teacher_across_layouts() -> None:
    from farming.actions import SteeringAction

    total = 0
    matches = 0
    for layout_id, (entry, env) in enumerate(
        iter_variant_environments(_CURRICULUM, stage="early", seed=21, episode_steps=20, episode_seconds=8.0)
    ):
        for episode in range(10):
            observation, _ = env.reset(seed=4000 + layout_id * 5003 + episode)
            for _ in range(20):
                env_angle = env.best_group_relative_angle()
                features = derive_geometry_features(np.asarray(observation, dtype=np.float32))
                assert features.shape == (GEOMETRY_FEATURE_SIZE,)
                predicted_steering = steering_from_geometry_features(features)
                if env_angle is not None:
                    expected_steering = (
                        int(SteeringAction.STRAIGHT)
                        if abs(env_angle) <= 0.18
                        else (int(SteeringAction.LEFT) if env_angle > 0.0 else int(SteeringAction.RIGHT))
                    )
                    total += 1
                    matches += int(predicted_steering == expected_steering)
                action = np.asarray(
                    [np.random.randint(0, 3), 0 if np.random.random() > 0.15 else 1], dtype=np.int64
                )
                observation, _reward, terminated, truncated, _info = env.step(action)
                if terminated or truncated:
                    break
        env.close()

    assert total > 100
    # Matches the ~92% real-data agreement rate found during diagnosis; a
    # regression here means the transform's geometry broke, not that the
    # threshold was too strict.
    assert matches / total > 0.85


def test_derive_geometry_features_batched_matches_single_row() -> None:
    entry, env = next(iter(iter_variant_environments(_CURRICULUM, stage="early", episode_steps=10, episode_seconds=5.0)))
    observations = []
    observation, _ = env.reset(seed=1)
    for _ in range(10):
        observations.append(np.asarray(observation, dtype=np.float32).copy())
        observation, _reward, terminated, truncated, _info = env.step(np.asarray([1, 0], dtype=np.int64))
        if terminated or truncated:
            break
    env.close()

    batch = np.stack(observations)
    batched_features = derive_geometry_features(batch)
    assert batched_features.shape == (len(observations), GEOMETRY_FEATURE_SIZE)
    for i, obs in enumerate(observations):
        single_features = derive_geometry_features(obs)
        assert np.allclose(single_features, batched_features[i])


def test_derive_geometry_features_reports_no_target_when_nothing_visible() -> None:
    observation = np.zeros(923, dtype=np.float32)
    observation[HEADING_COS_INDEX] = 1.0  # heading = 0, sin defaults to 0

    features = derive_geometry_features(observation)

    assert features[5] == -1.0  # has_target
    assert np.all(features[:5] == 0.0)
    assert steering_from_geometry_features(features) == 0  # STRAIGHT


def test_derive_geometry_features_mirror_consistency() -> None:
    """A world-Z mirror (negate every actor's dz and the heading's sin
    component) must negate the recovered relative angle and swap LEFT/RIGHT,
    proving the sign-handling in the rotation is geometrically correct and
    not an accidental one-sided fit.
    """

    from simulator.geometry_features import ACTOR_FEATURES, DIRECT_ACTOR_SLOTS

    rng = np.random.default_rng(0)
    observation = np.zeros(923, dtype=np.float32)
    heading = rng.uniform(-np.pi, np.pi)
    observation[HEADING_SIN_INDEX] = np.sin(heading)
    observation[HEADING_COS_INDEX] = np.cos(heading)

    # Place a handful of active actors with genuinely asymmetric dz so the
    # mirror is a meaningful transform, not a no-op.
    for slot in range(4):
        base = DIRECT_ACTOR_START + slot * ACTOR_FEATURES
        dx = rng.uniform(-0.8, 0.8)
        dz = rng.uniform(0.1, 0.8) * rng.choice([-1.0, 1.0])
        observation[base + 0] = dx
        observation[base + 1] = dz
        observation[base + 3] = 1.0  # active
        observation[base + 6] = rng.uniform(-1.0, 1.0)  # density_bipolar

    original = derive_geometry_features(observation)
    assert original[5] == 1.0  # sanity: a target was actually found

    mirrored = observation.copy()
    mirrored[HEADING_SIN_INDEX] = -observation[HEADING_SIN_INDEX]
    for slot in range(4):
        base = DIRECT_ACTOR_START + slot * ACTOR_FEATURES
        mirrored[base + 1] = -observation[base + 1]  # negate dz_over_vision

    mirrored_features = derive_geometry_features(mirrored)

    original_angle = np.arctan2(original[0], original[1])
    mirrored_angle = np.arctan2(mirrored_features[0], mirrored_features[1])
    assert np.isclose(mirrored_angle, -original_angle, atol=1e-4)

    original_steering = steering_from_geometry_features(original)
    mirrored_steering = steering_from_geometry_features(mirrored_features)
    if original_steering == 1:
        assert mirrored_steering == 2
    elif original_steering == 2:
        assert mirrored_steering == 1
    else:
        assert mirrored_steering == 0
