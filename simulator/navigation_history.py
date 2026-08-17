"""Collection-side rolling history wrapper that appends the shared,
canonical temporal + steering-state sidecar values to the observation.

Phase 2 design (see the approved plan): the raw 923-value observation
contract is never touched. Instead, a `gym.Wrapper` maintains a short
history of `NavigationStepEvidence` -- built from the same raw,
undecoded quantities `RecoveryController` and `milestone_evaluator`
already use (`total_distance_cells`/`contacts` diffed between steps),
never from the observation's encoded `displacement_bipolar`/
`contact_bipolar` fields -- and appends 2 derived temporal values
(`recent_progress`, `recent_contact`) to produce a POLICY INPUT distinct
from the 923-value recorder/live game-observation contract.

Phase 9: the pure, environment-agnostic evidence core
(`NavigationStepEvidence`, `previous_steering_one_hot`,
`sidecar_values_from_history`, and the sidecar-size constants) moved to
`navigation.navigation_evidence` -- the shared canonical owner, suitable for
a future live-bot port without dragging gymnasium into it. This module
retains only `NavigationHistoryWrapper`, the training/collection-side
`gymnasium.Wrapper`, which consumes the shared pure implementation.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised on minimal installations.
    gym = None
    spaces = None

from farming.actions import FarmingEvent
from navigation.movement_kernel import SteeringDirection
from navigation.navigation_evidence import (
    CALIBRATED_EXPECTED_CLEAR_PATH_DISPLACEMENT,
    CALIBRATED_HISTORY_WINDOW,
    POLICY_INPUT_SIZE,
    RAW_OBSERVATION_SIZE,
    STEERING_POLICY_INPUT_SCHEMA_ID,
    NavigationStepEvidence,
    sidecar_values_from_history,
)

__all__ = [
    "CALIBRATED_EXPECTED_CLEAR_PATH_DISPLACEMENT",
    "CALIBRATED_HISTORY_WINDOW",
    "NavigationHistoryWrapper",
    "NavigationStepEvidence",
    "POLICY_INPUT_SIZE",
    "RAW_OBSERVATION_SIZE",
    "STEERING_POLICY_INPUT_SCHEMA_ID",
    "sidecar_values_from_history",
]


class NavigationHistoryWrapper(gym.Wrapper if gym is not None else object):
    """Appends `[recent_progress, recent_contact, prev_straight,
    prev_left, prev_right]` to the observation.

    Statefulness only matters during this wrapper's own sequential
    collection (rollout/DAgger/eval) -- once appended, the five values are
    ordinary numbers in the stored transition, safe under any downstream
    shuffled-minibatch replay.

    EVA-cast ticks are excluded from both TEMPORAL statistics, mirroring
    RecoveryController._eva_history exactly. Only the single EVA-commanded
    tick is excluded by default; extending this to a multi-tick grace
    window requires measuring real post-cast displacement suppression
    first (see plan Step 1) -- do not assume a longer window. The
    previous-steering one-hot is an instantaneous state read (not
    windowed), so EVA exclusion does not apply to it.
    """

    def __init__(
        self,
        env: Any,
        *,
        window: int = CALIBRATED_HISTORY_WINDOW,
        expected_clear_path_displacement: float = CALIBRATED_EXPECTED_CLEAR_PATH_DISPLACEMENT,
    ) -> None:
        super().__init__(env)
        if window < 1:
            raise ValueError("window must be positive")
        if expected_clear_path_displacement <= 0.0:
            raise ValueError("expected_clear_path_displacement must be positive")
        self.window = int(window)
        self.expected_clear_path_displacement = float(expected_clear_path_displacement)
        self._history: deque[NavigationStepEvidence] = deque(maxlen=self.window)
        self._previous_distance = 0.0
        self._previous_contacts = 0
        if spaces is not None:
            base = env.observation_space
            self.observation_space = spaces.Box(
                low=float(base.low.flat[0]) if hasattr(base.low, "flat") else -1.0,
                high=float(base.high.flat[0]) if hasattr(base.high, "flat") else 1.0,
                shape=(POLICY_INPUT_SIZE,),
                dtype=np.float32,
            )

    def _sidecar_values(self, previous_steering: "SteeringDirection | int") -> np.ndarray:
        return sidecar_values_from_history(
            self._history, previous_steering,
            expected_clear_path_displacement=self.expected_clear_path_displacement,
        )

    def _augment(self, observation: np.ndarray, previous_steering: "SteeringDirection | int" = SteeringDirection.NONE) -> np.ndarray:
        raw = np.asarray(observation, dtype=np.float32).reshape(-1)
        if raw.shape[0] != RAW_OBSERVATION_SIZE:
            raise ValueError(
                f"NavigationHistoryWrapper expects a {RAW_OBSERVATION_SIZE}-value raw observation, "
                f"got {raw.shape[0]}"
            )
        return np.concatenate([raw, self._sidecar_values(previous_steering)])

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self._history.clear()
        self._previous_distance = 0.0
        self._previous_contacts = 0
        previous_steering = info.get("previous_steering", int(SteeringDirection.NONE))
        return self._augment(observation, previous_steering), info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_array = np.asarray(action)
        event = int(action_array[1]) if action_array.shape[0] > 1 else int(FarmingEvent.NONE)
        eva_attempted = event == int(FarmingEvent.CAST_EVA)

        observation, reward, terminated, truncated, info = self.env.step(action)

        total_distance = float(info.get("total_distance_cells", self._previous_distance))
        total_contacts = int(info.get("contacts", self._previous_contacts))
        displacement_this_tick = total_distance - self._previous_distance
        contact_this_tick = total_contacts > self._previous_contacts
        self._previous_distance = total_distance
        self._previous_contacts = total_contacts

        self._history.append(
            NavigationStepEvidence(
                displacement_cells=displacement_this_tick,
                contact=contact_this_tick,
                eva_attempted=eva_attempted,
            )
        )
        previous_steering = info.get("previous_steering", int(SteeringDirection.NONE))
        return self._augment(observation, previous_steering), reward, terminated, truncated, info
