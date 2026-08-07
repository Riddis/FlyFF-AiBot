from __future__ import annotations

import math

import numpy as np
import torch

from farming.map_features import LOCAL_MAP_OBSTACLE, LOCAL_MAP_SAFE
from simulator.local_navigation_features import (
    CLEARANCE_FEATURE_SIZE,
    HEADING_COS_INDEX,
    HEADING_SIN_INDEX,
    LOCAL_MAP_RADIUS,
    LOCAL_MAP_SIDE,
    LOCAL_MAP_START,
    LOCAL_MAP_STOP,
    derive_physical_clearance_features,
    derive_physical_clearance_features_torch,
)
from simulator.local_clearance import sample_heading_relative_clearance
from simulator.synthetic import iter_variant_environments


def _synthetic_observation(*, heading: float, local_map: np.ndarray) -> np.ndarray:
    assert local_map.shape == (LOCAL_MAP_SIDE, LOCAL_MAP_SIDE)
    obs = np.zeros((923,), dtype=np.float32)
    obs[HEADING_SIN_INDEX] = math.sin(heading)
    obs[HEADING_COS_INDEX] = math.cos(heading)
    obs[LOCAL_MAP_START:LOCAL_MAP_STOP] = local_map.reshape(-1)
    return obs


def _all_safe_grid() -> np.ndarray:
    return np.full((LOCAL_MAP_SIDE, LOCAL_MAP_SIDE), LOCAL_MAP_SAFE, dtype=np.float32)


def test_shape_single_and_batched():
    obs = _synthetic_observation(heading=0.0, local_map=_all_safe_grid())
    single = derive_physical_clearance_features(obs)
    assert single.shape == (CLEARANCE_FEATURE_SIZE,)
    batched = derive_physical_clearance_features(np.stack([obs, obs]))
    assert batched.shape == (2, CLEARANCE_FEATURE_SIZE)


def test_all_safe_grid_gives_full_clearance_all_directions():
    obs = _synthetic_observation(heading=0.0, local_map=_all_safe_grid())
    left, forward, right = derive_physical_clearance_features(obs)
    assert left == 1.0 and forward == 1.0 and right == 1.0


def test_wall_directly_ahead_blocks_forward_only():
    # heading=0 -> facing native +x -> forward sector samples layout
    # dx=1,2,3, dy=0 (row=radius, col=radius+distance). Block those cells.
    grid = _all_safe_grid()
    r = LOCAL_MAP_RADIUS
    for d in (1, 2, 3, 4, 5):
        grid[r, r + d] = LOCAL_MAP_OBSTACLE
    obs = _synthetic_observation(heading=0.0, local_map=grid)
    left, forward, right = derive_physical_clearance_features(obs)
    assert forward == 0.0
    assert left == 1.0
    assert right == 1.0


def test_native_plus_z_maps_to_negative_layout_dy():
    # heading=pi/2 -> facing native +z -> forward direction_z=1, so per the
    # confirmed native->layout axis flip (map_model.native_to_layout_cells),
    # forward samples should land at NEGATIVE dy (row < radius), not
    # positive. Block the negative-dy side only.
    grid = _all_safe_grid()
    r = LOCAL_MAP_RADIUS
    for d in (1, 2, 3, 4, 5):
        grid[r - d, r] = LOCAL_MAP_OBSTACLE  # negative dy side
    obs = _synthetic_observation(heading=math.pi / 2, local_map=grid)
    left, forward, right = derive_physical_clearance_features(obs)
    assert forward == 0.0  # confirms the flip direction is the blocked one

    grid2 = _all_safe_grid()
    for d in (1, 2, 3, 4, 5):
        grid2[r + d, r] = LOCAL_MAP_OBSTACLE  # positive dy side instead
    obs2 = _synthetic_observation(heading=math.pi / 2, local_map=grid2)
    _left2, forward2, _right2 = derive_physical_clearance_features(obs2)
    assert forward2 == 1.0  # positive-dy side being blocked must NOT affect forward


def test_torch_matches_numpy():
    grid = _all_safe_grid()
    r = LOCAL_MAP_RADIUS
    grid[r, r + 2] = LOCAL_MAP_OBSTACLE
    obs = _synthetic_observation(heading=0.3, local_map=grid)
    numpy_result = derive_physical_clearance_features(obs)
    torch_result = derive_physical_clearance_features_torch(
        torch.as_tensor(obs[None, :], dtype=torch.float32)
    )
    np.testing.assert_allclose(numpy_result, torch_result[0].numpy(), atol=1e-5)


def test_obstacle_buffer_treated_as_traversable_unlike_local_clearance():
    from farming.map_features import LOCAL_MAP_OBSTACLE_BUFFER

    grid = _all_safe_grid()
    r = LOCAL_MAP_RADIUS
    for d in (1, 2, 3, 4, 5):
        grid[r, r + d] = LOCAL_MAP_OBSTACLE_BUFFER
    obs = _synthetic_observation(heading=0.0, local_map=grid)
    _left, forward, _right = derive_physical_clearance_features(obs)
    assert forward == 1.0  # NOT 0.15-like near-blocked, per Step 0's finding


def test_agrees_reasonably_with_live_sample_heading_relative_clearance():
    """Cross-check against the already-validated live signal across many
    real env states -- same spirit as adversarial_clearance_validation.py.
    Perfect agreement isn't expected (different weighting for
    OBSTACLE_BUFFER/TELEPORT_BUFFER by design), but they must broadly track
    -- SAFE vs actually-OBSTACLE distinctions should match."""

    agree = 0
    total = 0
    for entry, env in iter_variant_environments(
        "synthetic_curriculum/curriculum.json", stage="early", seed=0, episode_steps=1, episode_seconds=5.0
    ):
        rng = np.random.default_rng(hash(entry.name) % (2**32))
        for _ in range(50):
            env.reset(seed=int(rng.integers(0, 1_000_000)))
            env.heading = float(rng.uniform(-math.pi, math.pi))
            live_scores = sample_heading_relative_clearance(env.map, env.player_x, env.player_z, env.heading)
            player_cell = env.map.native_to_layout_cell(env.player_x, env.player_z)
            local_map = env.map.features.local_crop(player_cell, side=LOCAL_MAP_SIDE)
            obs = np.zeros((923,), dtype=np.float32)
            obs[HEADING_SIN_INDEX] = math.sin(env.heading)
            obs[HEADING_COS_INDEX] = math.cos(env.heading)
            obs[LOCAL_MAP_START:LOCAL_MAP_STOP] = local_map
            left, forward, right = derive_physical_clearance_features(obs)
            for name, cached in zip(("left", "forward", "right"), (left, forward, right)):
                live_clear = live_scores[name] > 0.5
                cached_clear = cached > 0.5
                total += 1
                agree += int(live_clear == cached_clear)
        env.close()

    assert total > 500
    assert agree / total > 0.7  # broad agreement expected, not identical by design
