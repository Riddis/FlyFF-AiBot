from __future__ import annotations

import gymnasium as gym
import numpy as np

from farming.actions import FarmingEvent, SteeringAction
from navigation.movement_kernel import SteeringDirection
from navigation.navigation_evidence import (
    POLICY_INPUT_SIZE,
    RAW_OBSERVATION_SIZE,
)
from simulator.navigation_history import NavigationHistoryWrapper

# Sidecar layout, as shipped: [recent_progress, recent_contact, prev_straight, prev_left, prev_right].
_PROGRESS_OFFSET = RAW_OBSERVATION_SIZE
_CONTACT_OFFSET = RAW_OBSERVATION_SIZE + 1
_ONE_HOT_OFFSET = RAW_OBSERVATION_SIZE + 2


class _FakeEnv(gym.Env):
    """Minimal gym-like env: returns a fixed RAW_OBSERVATION_SIZE-value
    observation, and lets the test script each step's
    total_distance_cells/contacts/previous_steering/termination via a
    queue, matching the real env's cumulative info-dict fields."""

    def __init__(self) -> None:
        self.observation_space = _FakeSpace()
        self.action_space = None
        self._queue: list[dict] = []
        self._total_distance = 0.0
        self._total_contacts = 0

    def queue(self, *, displacement: float, contact: bool, previous_steering: int = int(SteeringDirection.NONE)) -> None:
        self._total_distance += displacement
        self._total_contacts += int(contact)
        self._queue.append(
            {
                "total_distance_cells": self._total_distance,
                "contacts": self._total_contacts,
                "previous_steering": previous_steering,
            }
        )

    def reset(self, **kwargs):
        self._total_distance = 0.0
        self._total_contacts = 0
        self._queue.clear()
        return np.zeros((RAW_OBSERVATION_SIZE,), dtype=np.float32), {"previous_steering": int(SteeringDirection.NONE)}

    def step(self, action):
        info = self._queue.pop(0)
        return np.zeros((RAW_OBSERVATION_SIZE,), dtype=np.float32), 0.0, False, False, info


class _FakeSpace:
    low = np.full((RAW_OBSERVATION_SIZE,), -1.0, dtype=np.float32)
    high = np.full((RAW_OBSERVATION_SIZE,), 1.0, dtype=np.float32)
    shape = (RAW_OBSERVATION_SIZE,)


_STRAIGHT_NONE = np.asarray([int(SteeringAction.STRAIGHT), int(FarmingEvent.NONE)], dtype=np.int64)
_STRAIGHT_EVA = np.asarray([int(SteeringAction.STRAIGHT), int(FarmingEvent.CAST_EVA)], dtype=np.int64)


def _wrapped(window=5, expected=1.0):
    env = _FakeEnv()
    wrapper = NavigationHistoryWrapper(env, window=window, expected_clear_path_displacement=expected)
    wrapper.reset()
    return env, wrapper


def test_observation_size_grows_by_sidecar_size():
    _, wrapper = _wrapped()
    assert wrapper.observation_space.shape == (POLICY_INPUT_SIZE,)


def test_reset_has_zero_temporal_sidecar_and_none_one_hot_with_no_history():
    env = _FakeEnv()
    wrapper = NavigationHistoryWrapper(env)
    obs, _info = wrapper.reset()
    assert obs.shape == (POLICY_INPUT_SIZE,)
    assert obs[_PROGRESS_OFFSET] == 0.0 and obs[_CONTACT_OFFSET] == 0.0
    assert list(obs[_ONE_HOT_OFFSET : _ONE_HOT_OFFSET + 3]) == [1.0, 0.0, 0.0]  # prev_straight (NONE)


def test_normal_progress_reports_full_progress_zero_contact():
    env, wrapper = _wrapped(window=5, expected=1.0)
    for _ in range(5):
        env.queue(displacement=1.0, contact=False)
        obs, *_ = wrapper.step(_STRAIGHT_NONE)
    assert obs[_PROGRESS_OFFSET] == 1.0  # recent_progress: mean displacement / expected == 1.0
    assert obs[_CONTACT_OFFSET] == 0.0  # recent_contact: no contacts


def test_no_progress_with_contacts_reports_low_progress_high_contact():
    env, wrapper = _wrapped(window=5, expected=1.0)
    for _ in range(5):
        env.queue(displacement=0.0, contact=True)
        obs, *_ = wrapper.step(_STRAIGHT_NONE)
    assert obs[_PROGRESS_OFFSET] == 0.0
    assert obs[_CONTACT_OFFSET] == 1.0


def test_eva_ticks_excluded_from_both_statistics():
    env, wrapper = _wrapped(window=5, expected=1.0)
    # One EVA tick with zero displacement (expected -- movement is ignored
    # during cast) surrounded by normal healthy progress; the EVA tick must
    # not drag recent_progress down or count toward recent_contact.
    env.queue(displacement=1.0, contact=False)
    wrapper.step(_STRAIGHT_NONE)
    env.queue(displacement=0.0, contact=False)
    wrapper.step(_STRAIGHT_EVA)
    env.queue(displacement=1.0, contact=False)
    obs, *_ = wrapper.step(_STRAIGHT_NONE)
    assert obs[_PROGRESS_OFFSET] == 1.0  # EVA tick excluded, remaining ticks are full progress
    assert obs[_CONTACT_OFFSET] == 0.0


def test_progress_ratio_is_clipped_to_one():
    env, wrapper = _wrapped(window=3, expected=1.0)
    for _ in range(3):
        env.queue(displacement=5.0, contact=False)  # far above expected clear-path rate
        obs, *_ = wrapper.step(_STRAIGHT_NONE)
    assert obs[_PROGRESS_OFFSET] == 1.0


def test_reset_clears_history_between_episodes():
    env, wrapper = _wrapped(window=5, expected=1.0)
    for _ in range(5):
        env.queue(displacement=0.0, contact=True)
        wrapper.step(_STRAIGHT_NONE)
    wrapper.reset()
    env.queue(displacement=1.0, contact=False)
    obs, *_ = wrapper.step(_STRAIGHT_NONE)
    assert obs[_PROGRESS_OFFSET] == 1.0
    assert obs[_CONTACT_OFFSET] == 0.0


def test_raw_values_pass_through_unchanged():
    env, wrapper = _wrapped()
    env.queue(displacement=1.0, contact=False)
    obs, *_ = wrapper.step(_STRAIGHT_NONE)
    assert obs.shape == (POLICY_INPUT_SIZE,)
    assert np.array_equal(obs[:RAW_OBSERVATION_SIZE], np.zeros((RAW_OBSERVATION_SIZE,), dtype=np.float32))


class TestPreviousSteeringOneHot:
    def test_straight_produces_prev_straight_one_hot(self):
        env, wrapper = _wrapped()
        env.queue(displacement=1.0, contact=False, previous_steering=int(SteeringDirection.NONE))
        obs, *_ = wrapper.step(_STRAIGHT_NONE)
        assert list(obs[_ONE_HOT_OFFSET : _ONE_HOT_OFFSET + 3]) == [1.0, 0.0, 0.0]

    def test_left_produces_prev_left_one_hot(self):
        env, wrapper = _wrapped()
        env.queue(displacement=1.0, contact=False, previous_steering=int(SteeringDirection.LEFT))
        obs, *_ = wrapper.step(_STRAIGHT_NONE)
        assert list(obs[_ONE_HOT_OFFSET : _ONE_HOT_OFFSET + 3]) == [0.0, 1.0, 0.0]

    def test_right_produces_prev_right_one_hot(self):
        env, wrapper = _wrapped()
        env.queue(displacement=1.0, contact=False, previous_steering=int(SteeringDirection.RIGHT))
        obs, *_ = wrapper.step(_STRAIGHT_NONE)
        assert list(obs[_ONE_HOT_OFFSET : _ONE_HOT_OFFSET + 3]) == [0.0, 0.0, 1.0]

    def test_missing_previous_steering_key_defaults_to_none(self):
        """A raw env that doesn't expose previous_steering at all (e.g. an
        un-migrated environment) must default to NONE, not crash -- guards
        the .get() fallback in NavigationHistoryWrapper."""
        env = _FakeEnv()
        wrapper = NavigationHistoryWrapper(env)
        wrapper.reset()
        env._queue = [{"total_distance_cells": 1.0, "contacts": 0}]  # no previous_steering key
        obs, *_ = wrapper.step(_STRAIGHT_NONE)
        assert list(obs[_ONE_HOT_OFFSET : _ONE_HOT_OFFSET + 3]) == [1.0, 0.0, 0.0]

    def test_one_hot_is_not_windowed_reflects_only_the_current_tick(self):
        """Unlike recent_progress/recent_contact, the one-hot must NOT be
        an average over history -- it always reflects only the most
        recent info["previous_steering"], flipping immediately when the
        commanded steering changes."""
        env, wrapper = _wrapped()
        for _ in range(4):
            env.queue(displacement=1.0, contact=False, previous_steering=int(SteeringDirection.LEFT))
            obs, *_ = wrapper.step(_STRAIGHT_NONE)
        assert list(obs[_ONE_HOT_OFFSET : _ONE_HOT_OFFSET + 3]) == [0.0, 1.0, 0.0]
        env.queue(displacement=1.0, contact=False, previous_steering=int(SteeringDirection.RIGHT))
        obs, *_ = wrapper.step(_STRAIGHT_NONE)
        assert list(obs[_ONE_HOT_OFFSET : _ONE_HOT_OFFSET + 3]) == [0.0, 0.0, 1.0]
