"""Proves simulator.basic_training.reconstruct_session_sidecars (the offline,
recording-derived sidecar reconstruction) produces EXACTLY the same
[recent_progress, recent_contact] values, at every observation index, as
NavigationHistoryWrapper's real online computation -- for the same
underlying sequence of transitions. This is the causal-alignment guarantee
the whole human-BC-bootstrap design depends on: no future-frame leakage, and
no accidental staleness either.

Method: drive the REAL NavigationHistoryWrapper against a small fake inner
env whose step() returns fully controlled (total_distance_cells, contacts)
so the exact online sidecar sequence is known, then feed the same
displacement/event sequence through reconstruct_session_sidecars and assert
index-by-index equality. Not a reimplementation of either side -- both
paths' actual production code run against the same known sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from farming.actions import FarmingEvent
from simulator.basic_training import reconstruct_session_sidecars
from simulator.navigation_history import (
    CALIBRATED_EXPECTED_CLEAR_PATH_DISPLACEMENT,
    CALIBRATED_HISTORY_WINDOW,
    NavigationHistoryWrapper,
)

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None
    spaces = None


@dataclass(frozen=True)
class Tick:
    """One controlled online tick: the action taken and its resulting
    per-tick displacement/contact."""

    event: int
    displacement: float
    contact: bool


class _FakeRawEnv(gym.Env if gym is not None else object):
    """Minimal 923-value-observation env whose step() outcomes are fully
    scripted by a Tick sequence, so the exact per-tick displacement/contact
    NavigationHistoryWrapper sees is known precisely -- not dependent on any
    real map/physics."""

    def __init__(self, ticks: list[Tick]) -> None:
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(923,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([3, 3])
        self._ticks = ticks
        self._cursor = 0
        self._total_distance = 0.0
        self._total_contacts = 0

    def reset(self, *, seed=None, options=None):
        self._cursor = 0
        self._total_distance = 0.0
        self._total_contacts = 0
        return np.zeros(923, dtype=np.float32), {}

    def step(self, action):
        tick = self._ticks[self._cursor]
        self._cursor += 1
        self._total_distance += tick.displacement
        if tick.contact:
            self._total_contacts += 1
        info = {"total_distance_cells": self._total_distance, "contacts": self._total_contacts}
        return np.zeros(923, dtype=np.float32), 0.0, False, False, info


def _collect_online_sidecars(ticks: list[Tick], *, window: int, expected_clear_path_displacement: float) -> np.ndarray:
    env = NavigationHistoryWrapper(
        _FakeRawEnv(ticks), window=window, expected_clear_path_displacement=expected_clear_path_displacement,
    )
    observation, _ = env.reset(seed=0)
    sidecars = [observation[923:925].copy()]
    for tick in ticks:
        action = np.asarray([0, tick.event], dtype=np.int64)
        observation, _r, _term, _trunc, _info = env.step(action)
        sidecars.append(observation[923:925].copy())
    env.close()
    return np.asarray(sidecars, dtype=np.float32)


def _offline_sidecars(ticks: list[Tick], *, window: int, expected_clear_path_displacement: float) -> np.ndarray:
    n = len(ticks) + 1  # +1 for the initial (reset-equivalent) sample
    displacement_cells = np.zeros(n, dtype=np.float32)
    events = np.zeros(n, dtype=np.int64)
    contact = np.zeros(n, dtype=np.bool_)
    for idx, tick in enumerate(ticks):
        displacement_cells[idx + 1] = tick.displacement  # transition idx -> idx+1
        contact[idx + 1] = tick.contact  # transition idx -> idx+1
        events[idx] = tick.event  # action taken AT sample idx, driving idx -> idx+1
    # Real ground-truth contact is passed here deliberately -- this test
    # verifies the alignment algorithm itself, not the separate,
    # human-recording-specific "contact is unknown" limitation (see
    # reconstruct_session_sidecars' docstring and
    # test_basic_training_pipeline.py's neutral-placeholder test for that).
    return reconstruct_session_sidecars(
        displacement_cells, events, contact=contact,
        history_window=window, expected_clear_path_displacement=expected_clear_path_displacement,
    )


def _assert_parity(ticks: list[Tick], *, window: int = CALIBRATED_HISTORY_WINDOW,
                    expected_clear_path_displacement: float = CALIBRATED_EXPECTED_CLEAR_PATH_DISPLACEMENT) -> None:
    online = _collect_online_sidecars(ticks, window=window, expected_clear_path_displacement=expected_clear_path_displacement)
    offline = _offline_sidecars(ticks, window=window, expected_clear_path_displacement=expected_clear_path_displacement)
    assert online.shape == offline.shape
    np.testing.assert_allclose(
        online, offline, atol=1e-6,
        err_msg="online (NavigationHistoryWrapper) and offline (reconstruct_session_sidecars) sidecars diverge",
    )


NONE = int(FarmingEvent.NONE)
EVA = int(FarmingEvent.CAST_EVA)


def test_parity_reset_startup_is_zero_on_both_sides() -> None:
    _assert_parity([])


def test_parity_normal_movement() -> None:
    _assert_parity([Tick(NONE, 1.8, False) for _ in range(5)])


def test_parity_zero_movement() -> None:
    _assert_parity([Tick(NONE, 0.0, False) for _ in range(5)])


def test_parity_eva_cast_excluded_from_both() -> None:
    _assert_parity([
        Tick(NONE, 1.8, False), Tick(NONE, 1.8, False),
        Tick(EVA, 0.0, False),
        Tick(NONE, 1.8, False), Tick(NONE, 1.8, False),
    ])


def test_parity_contact() -> None:
    _assert_parity([
        Tick(NONE, 1.8, False), Tick(NONE, 0.1, True), Tick(NONE, 0.1, True), Tick(NONE, 1.5, False),
    ])


def test_parity_mixed_realistic_sequence() -> None:
    rng = np.random.default_rng(7)
    ticks = []
    for _ in range(40):
        event = EVA if rng.random() < 0.1 else NONE
        contact = bool(rng.random() < 0.15)
        displacement = 0.0 if (event == EVA or contact) else float(rng.uniform(0.5, 2.5))
        ticks.append(Tick(event, displacement, contact))
    _assert_parity(ticks)


def test_parity_partial_history_window_shorter_than_calibrated() -> None:
    """window smaller than the sequence length, so the sliding window
    actually evicts old evidence on both sides -- exercises the boundary
    case where old ticks fall out of scope."""
    ticks = [Tick(NONE, float(1 + (i % 3)), i % 7 == 0) for i in range(25)]
    _assert_parity(ticks, window=5)


def test_parity_window_larger_than_sequence() -> None:
    ticks = [Tick(NONE, 1.2, False) for _ in range(3)]
    _assert_parity(ticks, window=CALIBRATED_HISTORY_WINDOW)
