from __future__ import annotations

import numpy as np

from simulator.factorized_cli import _evaluate_env


class _FakeEnv:
    """Minimal env stub with fully controlled, deterministic per-episode
    kills/elapsed-time so the median-vs-pooled distinction is exact and not
    at the mercy of real simulation randomness.
    """

    def __init__(self, kills_per_episode: list[int], elapsed_per_episode: list[float]) -> None:
        self._kills = kills_per_episode
        self._elapsed = elapsed_per_episode
        self._episode = -1

    def reset(self, seed=None):
        self._episode += 1
        return np.zeros(923, dtype=np.float32), {}

    def step(self, action):
        info = {
            "elapsed_seconds": self._elapsed[self._episode],
            "total_kills": self._kills[self._episode],
            "valid_eva_casts": 0,
            "invalid_eva_attempts": 0,
            "contacts": 0,
            "total_distance_cells": 0.0,
            "path_efficiency": 0.0,
            "reward_component_totals": {},
        }
        return np.zeros(923, dtype=np.float32), 0.0, True, False, info


def _constant_selector(_observation, _env):
    return np.asarray([0, 0], dtype=np.int64)


def test_kills_per_simulated_hour_is_episode_median_not_pooled_rate() -> None:
    # One outlier episode (100 kills) would dominate a pooled sum-over-time
    # rate; the median should instead reflect the two typical episodes.
    env = _FakeEnv(kills_per_episode=[10, 100, 20], elapsed_per_episode=[3600.0, 3600.0, 3600.0])

    report = _evaluate_env(env, _constant_selector, episodes=3, max_actions=1, seed=0, label="test")

    assert report["kills_per_simulated_hour"] == 20.0
    assert report["kills_per_simulated_hour_min"] == 10.0
    assert report["kills_per_simulated_hour_max"] == 100.0
    assert report["pooled_kills_per_simulated_hour"] == 130.0 / 3.0
    assert report["episode_kills_per_simulated_hour"] == [10.0, 100.0, 20.0]


def test_kills_per_simulated_hour_matches_pooled_for_a_single_episode() -> None:
    # episodes=1 must reproduce the exact old behaviour -- median of one
    # value equals that value, so every existing single-episode caller is
    # unaffected by this change.
    env = _FakeEnv(kills_per_episode=[42], elapsed_per_episode=[1800.0])

    report = _evaluate_env(env, _constant_selector, episodes=1, max_actions=1, seed=0, label="test")

    assert report["kills_per_simulated_hour"] == report["pooled_kills_per_simulated_hour"]
    assert report["kills_per_simulated_hour"] == 42 * 3600.0 / 1800.0
