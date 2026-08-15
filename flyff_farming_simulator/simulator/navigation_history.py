"""Canonical per-tick navigation evidence and a collection-side rolling
history wrapper that appends temporal + steering-state sidecar values to
the observation.

Phase 2 design (see the approved plan): the raw 923-value observation
contract is never touched. Instead, a `gym.Wrapper` maintains a short
history of `NavigationStepEvidence` -- built from the same raw,
undecoded quantities `RecoveryController` and `milestone_evaluator`
already use (`total_distance_cells`/`contacts` diffed between steps),
never from the observation's encoded `displacement_bipolar`/
`contact_bipolar` fields -- and appends 2 derived temporal values
(`recent_progress`, `recent_contact`) to produce a POLICY INPUT distinct
from the 923-value recorder/live game-observation contract.

`NavigationStepEvidence` is deliberately environment-agnostic: the
simulator populates it from its own info-dict keys, but a future live-bot
port populates the identical structure from its own native transition
data without needing to match the simulator's specific field names --
this is the intended migration seam (see plan: simulator is canonical,
live bot is refactored to match, not the reverse).

Window size and the `expected_clear_path_displacement` normalizer are
calibration outputs, frozen from the navigation_calibration pool (see
CALIBRATED_HISTORY_WINDOW / CALIBRATED_EXPECTED_CLEAR_PATH_DISPLACEMENT
below and evaluations/navigation_calibration_results.json) -- not guesses.

2026-08-13 addition (calibrated constant-curvature-arc kernel migration):
the corrected movement physics makes `previous_steering` part of the
environment's Markov state (movement_kernel.resolve_signed_turn_radians
is stateful -- the same current action produces a different turn
depending on whether it continues the prior tick's steering or not). This
is NOT privileged simulator information: a live controller always knows
what steering it itself commanded last tick. A 3-way one-hot
(`prev_straight`/`prev_left`/`prev_right`), read directly from
`info["previous_steering"]` (RecordedFarmingEnv exposes this as a plain
int-valued SteeringDirection), is appended alongside the 2 temporal
values -- sidecar size grows from 2 to 5 accordingly. See
run_logs/REPLACEMENT_MOVEMENT_MODEL_SPEC_2026-08-13.md's "Previous
steering is Markov state" section.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised on minimal installations.
    gym = None
    spaces = None

from farming.actions import FarmingEvent

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
