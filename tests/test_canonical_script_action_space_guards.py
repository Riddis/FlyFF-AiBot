"""Regression test for the `_require_farming_policy_action_space` guard each
of RUN_CANONICAL_BEGINNER.py/RUN_CANONICAL_INTERMEDIATE.py/RUN_CANONICAL_
ADVANCED.py defines independently (one per script, not a shared function --
so a drift in any one of them cannot be caught by testing the others).

Written after RUN_CANONICAL_INTERMEDIATE.py's own guard was found still
checking the retired `Discrete(len(FarmingEvent))` event-only contract after
the rest of the pipeline moved to `MultiDiscrete([TARGET_ACTION_SIZE,
len(FarmingEvent)])` -- it would have rejected every legitimate Beginner-
graduated checkpoint. See MISTAKES.md, 2026-08-22, for the incident.
"""
from __future__ import annotations

import importlib

import pytest

from simulator.basic_training import build_fresh_basic_policy

SCRIPT_MODULES = [
    "simulator.tools.RUN_CANONICAL_BEGINNER",
    "simulator.tools.RUN_CANONICAL_INTERMEDIATE",
    "simulator.tools.RUN_CANONICAL_ADVANCED",
]


@pytest.mark.parametrize("module_name", SCRIPT_MODULES)
def test_guard_accepts_current_farming_policy_checkpoint(module_name: str) -> None:
    module = importlib.import_module(module_name)
    model = build_fresh_basic_policy(seed=0, device="cpu")
    module._require_farming_policy_action_space(model, where="test")


@pytest.mark.parametrize("module_name", SCRIPT_MODULES)
def test_guard_rejects_retired_event_only_discrete_checkpoint(module_name: str) -> None:
    """A stale `Discrete(len(FarmingEvent))` checkpoint (the retired
    event-only contract) must never be silently accepted -- it has no
    target-selection head at all."""
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO

    from farming.actions import FarmingEvent

    module = importlib.import_module(module_name)

    class _EventOnlyProbe(gym.Env):
        metadata: dict = {"render_modes": []}

        def __init__(self) -> None:
            super().__init__()
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(923,), dtype="float32")
            self.action_space = spaces.Discrete(len(FarmingEvent))

        def reset(self, *, seed=None, options=None):
            import numpy as np
            return np.zeros(923, dtype="float32"), {}

        def step(self, action):
            import numpy as np
            return np.zeros(923, dtype="float32"), 0.0, True, False, {}

    stale_model = PPO("MlpPolicy", _EventOnlyProbe(), device="cpu")
    with pytest.raises(RuntimeError):
        module._require_farming_policy_action_space(stale_model, where="test")
