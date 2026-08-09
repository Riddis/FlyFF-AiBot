"""Pure-navigation environment wrapper for the 2026-08-09 PPO ablation
experiment: strips farming/EVA/kill reward entirely, rewards only forward
progress, and terminates the episode immediately on any physical contact
with a large negative terminal reward.

Purpose: answer directly whether the hand-engineered steering oracle's
machinery (terminal-continuation gate, robust escape-BFS, target-selection
hysteresis) is actually necessary, or whether a policy trained end-to-end
under an unambiguous "collision ends the episode" incentive learns clean
navigation on its own. Two conditions, same wrapper, only the underlying
env's target-selection stability differs:
  A. stable_waypoint: the env's target-hysteresis margin is set to
     effectively infinite, so the initially-selected target is kept for
     the whole episode (re-targeting only if it dies/goes unreachable).
  B. normal_target: the env's actual current default target-selection
     behavior (hysteresis enabled, margin=3.0 cells, per the 2026-08-09
     shared-infrastructure fix) -- i.e. "the real system as it exists
     today", not the pre-hysteresis pure-greedy baseline.
"""
from __future__ import annotations

import math
from typing import Any, Literal

import gymnasium as gym

TargetMode = Literal["stable_waypoint", "normal_target"]

COLLISION_TERMINAL_REWARD = -5.0
PROGRESS_REWARD_SCALE = 1.0
# Large enough that no candidate could ever beat the sticky target under
# the existing margin comparison (see environment.py's hysteresis logic),
# effectively locking the initial target for the whole episode short of it
# dying/going unreachable.
EFFECTIVELY_INFINITE_HYSTERESIS_MARGIN_CELLS = 1.0e6


def configure_target_mode(wrapped_env: Any, mode: TargetMode) -> None:
    """Apply the requested target-selection mode to the underlying
    RecordedFarmingEnv instance. Uses `.unwrapped` (gymnasium's standard
    recursive-unwrap property) rather than manually walking `.env` --
    gymnasium's Wrapper implements `__getattr__` delegation, so a naive
    `hasattr` probe on the OUTER wrapper would already appear to "have" any
    attribute the base env has, terminating a manual walk immediately and
    setting the attribute on the wrong (outer) object where it would
    silently have no effect. Call once per episode, right after reset,
    since a fresh reset does not reset this configuration itself."""

    env = wrapped_env.unwrapped
    env.target_hysteresis_enabled = True
    if mode == "stable_waypoint":
        env._TARGET_HYSTERESIS_MARGIN_CELLS = EFFECTIVELY_INFINITE_HYSTERESIS_MARGIN_CELLS
    else:
        env._TARGET_HYSTERESIS_MARGIN_CELLS = type(env)._TARGET_HYSTERESIS_MARGIN_CELLS


class PureNavigationWrapper(gym.Wrapper):
    """Reward = per-tick forward progress (cells moved); episode terminates
    immediately with COLLISION_TERMINAL_REWARD on the first physical
    contact. Farming/EVA/kill reward is never consulted."""

    def __init__(self, env: Any, *, target_mode: TargetMode):
        super().__init__(env)
        self.target_mode = target_mode
        self._prev_contacts = 0
        self._prev_distance = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        configure_target_mode(self.env, self.target_mode)
        self._prev_contacts = int(info.get("contacts", 0)) if isinstance(info, dict) else 0
        self._prev_distance = float(info.get("total_distance_cells", 0.0)) if isinstance(info, dict) else 0.0
        return obs, info

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)
        contacts = int(info.get("contacts", 0))
        distance = float(info.get("total_distance_cells", 0.0))
        displacement = distance - self._prev_distance
        contact_this_tick = contacts > self._prev_contacts
        self._prev_contacts = contacts
        self._prev_distance = distance
        if contact_this_tick:
            return obs, COLLISION_TERMINAL_REWARD, True, truncated, info
        reward = PROGRESS_REWARD_SCALE * max(0.0, displacement)
        return obs, reward, terminated, truncated, info
