"""Pure-navigation environment wrapper for the 2026-08-09 PPO ablation
experiment: strips farming/EVA/kill reward entirely, rewards only forward
progress, and terminates the episode immediately on any physical contact
with a large negative terminal reward.

Purpose: answer directly whether the hand-engineered steering oracle's
machinery (terminal-continuation gate, robust escape-BFS, target-selection
hysteresis) is actually necessary, or whether a policy trained end-to-end
under an unambiguous "collision ends the episode" incentive learns clean
navigation on its own. Two conditions, same wrapper, only target stability
differs:
  A. stable_waypoint: the env's target-hysteresis margin is set to
     effectively infinite (initial target held all episode, barring
     death/unreachability) AND the raw observation's direct-actor block is
     masked so only that one target actor is ever "active" -- see the
     CRITICAL FIX note below for why the mask is required.
  B. normal_target: the env's actual current default target-selection
     behavior (hysteresis enabled, margin=3.0 cells) with NO observation
     masking -- i.e. "the real system as it exists today."

CRITICAL FIX (2026-08-09, caught before it produced a misleading result):
the first version of this module only set `target_hysteresis_enabled`/
`_TARGET_HYSTERESIS_MARGIN_CELLS` on the underlying env, assuming that
would make the POLICY's own observation stable too. It does not.
`simulator/geometry_features.py`'s `derive_geometry_features[_torch]` --
which produces the 6 target-geometry values in the policy's actual 11-
feature steering input -- deliberately does NOT read
`_nearest_reachable_actor_id`/`_best_group_actor_id` at all; it
independently recomputes its own stateless, memoryless "best candidate"
selection from the raw observation's direct-actor slots on every call (by
design, so the live bot can compute the identical transform without any
privileged env state). Two PPO runs under the env-only fix produced
byte-identical training statistics at every checkpoint, which is what
caught this. The fix here masks the raw observation directly: only the
current sticky target's direct-actor slot is left "active", forcing
`derive_geometry_features`'s greedy selection to have exactly one
candidate, achieving real stability at the level the policy actually sees
-- without modifying the shared, foundational `geometry_features.py`
transform used elsewhere (DAgger, the live bot, etc.).
"""
from __future__ import annotations

from typing import Any, Literal

import gymnasium as gym
import numpy as np

from farming.observation import ACTOR_FEATURES, DIRECT_ACTOR_SLOTS, DIRECT_ACTOR_START, _DIRECT_ACTOR_FIELD_NAMES

TargetMode = Literal["stable_waypoint", "normal_target"]

COLLISION_TERMINAL_REWARD = -5.0
PROGRESS_REWARD_SCALE = 1.0
# Large enough that no candidate could ever beat the sticky target under
# the existing margin comparison (see environment.py's hysteresis logic),
# effectively locking the initial target for the whole episode short of it
# dying/going unreachable.
EFFECTIVELY_INFINITE_HYSTERESIS_MARGIN_CELLS = 1.0e6
_ACTIVE_FIELD_OFFSET = _DIRECT_ACTOR_FIELD_NAMES.index("active")


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


def _mask_observation_to_sticky_target(obs: np.ndarray, base_env: Any) -> np.ndarray:
    """Return a COPY of `obs` with every direct-actor slot deactivated
    except whichever slot (if any) holds the env's current sticky target
    (`_nearest_reachable_actor_id`, falling back to `_best_group_actor_id`
    -- the same precedence `_obstacle_aware_target_angle` uses). Reuses the
    env's own already-stabilized target ID (set by the hysteresis fix, with
    an effectively-infinite margin under stable_waypoint mode) rather than
    tracking a second, separate notion of "sticky" -- one source of truth
    for which actor counts as the target."""

    sticky_id = base_env._nearest_reachable_actor_id
    if sticky_id is None:
        sticky_id = base_env._best_group_actor_id

    masked = np.array(obs, copy=True)
    candidates = base_env._visible_candidates()[:DIRECT_ACTOR_SLOTS]
    for slot, (_distance, actor, _dx, _dz) in enumerate(candidates):
        if actor.actor_id == sticky_id:
            continue
        offset = DIRECT_ACTOR_START + slot * ACTOR_FEATURES + _ACTIVE_FIELD_OFFSET
        masked[offset] = 0.0
    return masked


class PureNavigationWrapper(gym.Wrapper):
    """Reward = per-tick forward progress (cells moved); episode terminates
    immediately with COLLISION_TERMINAL_REWARD on the first physical
    contact. Farming/EVA/kill reward is never consulted."""

    def __init__(self, env: Any, *, target_mode: TargetMode):
        super().__init__(env)
        self.target_mode = target_mode
        self._prev_contacts = 0
        self._prev_distance = 0.0

    def _maybe_mask(self, obs: np.ndarray) -> np.ndarray:
        if self.target_mode != "stable_waypoint":
            return obs
        return _mask_observation_to_sticky_target(obs, self.env.unwrapped)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        configure_target_mode(self.env, self.target_mode)
        self._prev_contacts = int(info.get("contacts", 0)) if isinstance(info, dict) else 0
        self._prev_distance = float(info.get("total_distance_cells", 0.0)) if isinstance(info, dict) else 0.0
        return self._maybe_mask(obs), info

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)
        contacts = int(info.get("contacts", 0))
        distance = float(info.get("total_distance_cells", 0.0))
        displacement = distance - self._prev_distance
        contact_this_tick = contacts > self._prev_contacts
        self._prev_contacts = contacts
        self._prev_distance = distance
        obs = self._maybe_mask(obs)
        if contact_this_tick:
            return obs, COLLISION_TERMINAL_REWARD, True, truncated, info
        reward = PROGRESS_REWARD_SCALE * max(0.0, displacement)
        return obs, reward, terminated, truncated, info
