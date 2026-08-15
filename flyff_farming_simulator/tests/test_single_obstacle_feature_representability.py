"""2026-08-11: regression test for single-obstacle feature
representability, locking in the finding from
scratchpad_single_obstacle_feature_check.py -- target and clearance
features stay correctly mirrored between matched LEFT-gap/RIGHT-gap
trajectories, and the clearance asymmetry becomes visible/differentiable
well before any collision risk (not a late, near-collision-only signal).
"""
from __future__ import annotations

import numpy as np

from simulator.geometry_features import derive_geometry_features
from simulator.local_navigation_features import derive_physical_clearance_features
from simulator.navigation_history import NavigationHistoryWrapper
from simulator.single_obstacle_env import ObstacleSpec, SingleObstacleWrapper, make_single_obstacle_env
from simulator.static_waypoint_env import SYMMETRIC_MOVEMENT

MAX_TICKS = 20


def _extract_11_features(obs: np.ndarray) -> np.ndarray:
    geometry = derive_geometry_features(obs[:923])
    clearance = derive_physical_clearance_features(obs[:923])
    sidecar = obs[923:925]
    return np.concatenate([geometry, clearance, sidecar])


def _run_straight(spec: ObstacleSpec) -> list[np.ndarray]:
    raw_env = make_single_obstacle_env(spec, episode_steps=MAX_TICKS + 5, seed=0, movement=SYMMETRIC_MOVEMENT)
    env = SingleObstacleWrapper(NavigationHistoryWrapper(raw_env), movement=SYMMETRIC_MOVEMENT, spec_source=lambda rng, s=spec: s)
    obs, _info = env.reset(seed=0)
    rows = [_extract_11_features(obs)]
    for _tick in range(MAX_TICKS):
        obs, _r, term, trunc, _info = env.step(np.asarray([0, 0], dtype=np.int64))
        rows.append(_extract_11_features(obs))
        if term or trunc:
            break
    env.close()
    return rows


class TestMirroredFeatureRepresentability:
    def test_target_and_clearance_features_stay_mirrored_along_straight_approach(self):
        left_spec = ObstacleSpec(gap_side="left", distance_cells=18.0, wall_offset_cells=6, wall_depth_cells=3, half_span_cells=6)
        right_spec = ObstacleSpec(gap_side="right", distance_cells=18.0, wall_offset_cells=6, wall_depth_cells=3, half_span_cells=6)
        left_rows = _run_straight(left_spec)
        right_rows = _run_straight(right_spec)
        assert len(left_rows) == len(right_rows)
        for fl, fr in zip(left_rows, right_rows):
            # Target geometry: sin mirrors, cos/distance/density match.
            assert abs(fl[0] - (-fr[0])) < 0.02
            assert abs(fl[1] - fr[1]) < 0.02
            assert abs(fl[2] - fr[2]) < 0.02
            assert abs(fl[3] - fr[3]) < 0.02
            # Clearance: left(L-gap)~=right(R-gap), right(L-gap)~=left(R-gap), forward matches.
            assert abs(fl[6] - fr[8]) < 0.05
            assert abs(fl[8] - fr[6]) < 0.05
            assert abs(fl[7] - fr[7]) < 0.05

    def test_clearance_asymmetry_becomes_differentiable_before_collision_risk(self):
        """The wall must become visible/distinguishable in the clearance
        features well before the episode would plausibly reach it -- not
        only in the final tick or two before contact."""
        left_spec = ObstacleSpec(gap_side="left", distance_cells=18.0, wall_offset_cells=6, wall_depth_cells=3, half_span_cells=6)
        rows = _run_straight(left_spec)
        differentiable_ticks = [i for i, f in enumerate(rows) if abs(f[6] - f[8]) > 0.15]
        assert differentiable_ticks, "clearance asymmetry never became meaningfully differentiable"
        # Wall offset is 6 cells; a differentiable signal by tick 4 leaves
        # real reaction room (several more ticks before the wall is reached).
        assert differentiable_ticks[0] <= 4
