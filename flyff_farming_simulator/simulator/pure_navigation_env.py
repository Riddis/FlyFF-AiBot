"""Pure-navigation environment wrapper for the 2026-08-09/10 PPO ablation
experiment: strips farming/EVA/kill reward entirely and terminates the
episode immediately on any physical contact with a large negative terminal
reward.

Purpose: answer directly whether the hand-engineered steering oracle's
machinery (terminal-continuation gate, robust escape-BFS, target-selection
hysteresis) is actually necessary, or whether a policy trained end-to-end
under an unambiguous "collision ends the episode" incentive learns clean
navigation on its own -- and specifically whether TARGET STABILITY is the
deciding factor, isolated from the confound of the oracle's other
machinery.

Two independent axes, per the 2026-08-10 correction (the first version
conflated them):

`reward_mode`:
  "safety" -- reward = per-tick forward progress (any direction). Answers
    "can PPO learn generic collision-free locomotion?" The optimal policy
    under this reward is free to ignore target features entirely (e.g.
    circle in open space), so this must NOT be used to draw any conclusion
    about target selection -- it is a locomotion-only baseline.
  "goal" -- reward = per-tick REDUCTION in distance to the currently
    selected target (stable or normal, per `target_mode`). Requires the
    agent to actually make progress toward an objective, not just avoid
    contact, so ignoring the target features is no longer a viable
    strategy. This is the mode that actually tests target stability.

`target_mode` (only meaningfully distinguishable under reward_mode="goal"):
  A. stable_waypoint: sticky selection (see `select_target` below) -- keep
     the current target while it remains alive, reachable, and
     representable; reselect only when it stops being any of those.
  B. normal_target: greedy selection -- reselect the best representable,
     reachable target every tick, from the identical candidate pool sticky
     mode draws from. Persistence is the ONLY difference between modes.

STEERING-ONLY, per the 2026-08-10 correction: the event action (EVA/jump)
is always forced to NONE before being applied to the underlying env,
regardless of what the policy's event head outputs, so EVA's cast-lock
movement suppression can never become an accidental collision-avoidance
mechanism. This experiment answers whether PPO can learn WALL AVOIDANCE
AT THE STEERING LEVEL (STRAIGHT/LEFT/RIGHT), nothing else.

CRITICAL FIX (2026-08-09): the first version only set env-level hysteresis
state, assuming that would make the POLICY's own observation stable too.
It does not -- `derive_geometry_features[_torch]` independently recomputes
its own target selection from the raw observation's direct-actor slots on
every call, deliberately ignoring any privileged env state (by design, so
the live bot can compute the identical transform without privileged
state). Fixed by masking the raw observation directly so only the selected
target's slot is ever "active".

CRITICAL FIX #2 (2026-08-10): the fix above was only ever applied for
`stable_waypoint`; `normal_target` episodes left masking a no-op, so its
policy-observed "target" (derive_geometry_features' own euclidean-distance
argmax over ALL active slots) could disagree with whichever actor the
reward was computed against. Fixed by making masking mode-independent.

CRITICAL FIX #3 (2026-08-10, after the user reviewed the completed Track A
results and correctly rejected treating them as decisive): fix #2 still
masked/rewarded against `_nearest_reachable_actor_id`/`_best_group_actor_id`
-- environment.py attributes selected by scanning ALL visible candidates
(commonly 300+ actors) via GEODESIC distance, with an env-level hysteresis
margin that only governs whether a candidate is allowed to STEAL the
target, never whether the current target remains within the
DIRECT_ACTOR_SLOTS=12-widest observation window the policy actually sees.
Measured directly: under `stable_waypoint`'s effectively-infinite margin,
the selected target fell outside the representable observation window on
74.0% of ticks (1201/1623 across the 6-episode eval set) -- the policy's
target-geometry features were zero on three-quarters of all ticks while
the reward kept scoring progress toward that same, invisible-to-the-policy
actor. `normal_target`'s margin=3.0 kept this gap to 0.6%, but that
measurement itself predates this fix (fix #2's masking made it representable
by CONSTRUCTION on the ticks it was active, not by ruling out the
possibility that a fresh reselection could still pick something outside
the window under fix #2's borrowed env-level logic) and cannot be trusted
as a clean isolation of "reselection frequency" either.

Fixed by replacing all env-attribute-borrowing with a single, self-
contained target selector (`select_target` below) that is the sole source
of truth for BOTH the policy's masked observation and the reward,
constructed so a selected target can never be anything other than
representable:
  - The candidate pool is always exactly `_representable_candidates` --
    the same DIRECT_ACTOR_SLOTS-nearest-by-raw-distance pool the
    observation's direct-actor block is built from. Never the full
    (hundreds-wide) visible-candidate list.
  - "Reachable" means finite geodesic distance, computed fresh each call.
  - Sticky mode retains the previous target id only if it is still alive,
    still among the representable candidates, AND still reachable this
    tick; otherwise it reselects the nearest reachable representable
    candidate, exactly as greedy mode always does. The two modes therefore
    differ ONLY in persistence, never in candidate pool or ranking metric.
This makes "the reward target is unrepresentable to the policy" a
structurally impossible state rather than a measured-to-be-rare one; see
`tests/test_pure_navigation_env.py::TestNoUnrepresentableRewardTarget`.

Also per this correction: the terminal collision penalty was recalibrated
from measurement rather than left at an unexamined -5.0 (see
`scratchpad_calibrate_pure_nav_reward.py` and this module's
`COLLISION_TERMINAL_REWARD` constant for the reasoning and numbers).
"""
from __future__ import annotations

import math
from typing import Any, Literal

import gymnasium as gym
import numpy as np

from farming.observation import ACTOR_FEATURES, DIRECT_ACTOR_SLOTS, DIRECT_ACTOR_START, _DIRECT_ACTOR_FIELD_NAMES

TargetMode = Literal["stable_waypoint", "normal_target"]
RewardMode = Literal["safety", "goal"]

# Calibrated 2026-08-10 from measurement, not an arbitrary constant (see
# scratchpad_calibrate_pure_nav_reward.py and the OVERNIGHT_20260809_
# PIPELINE.md entry of the same date). Measured with the scripted teacher
# over MAX_ACTIONS=1000 training episodes: per-tick safety reward mean=1.09,
# p90=2.35, observed max=4.69. The THEORETICAL WORST-CASE ceiling -- the
# discounted (gamma=0.99, matching training) return of a trajectory that
# somehow sustained the observed per-tick max for the ENTIRE 1000-tick
# horizon -- is 428.67. Real episodes fall far short of that (discounted
# productive return: safety mean=107/max=111, goal-greedy mean=31/max=46,
# goal-sticky mean=-25/max=-9 -- negative even for the scripted teacher).
# Set comfortably above the theoretical ceiling (not just the realistic
# mean/max) so that a collision is decisively worse than a collision-free
# trajectory under ANY circumstance the training run could produce, not
# just the typical case -- directly addressing the finding that the old
# -5.0 constant was frequently dominated by 80-150+ units of banked safety
# reward before termination.
COLLISION_TERMINAL_REWARD = -500.0
PROGRESS_REWARD_SCALE = 1.0
GOAL_PROGRESS_REWARD_SCALE = 1.0
EVENT_NONE = 0
_ACTIVE_FIELD_OFFSET = _DIRECT_ACTOR_FIELD_NAMES.index("active")


def _representable_candidates(base_env: Any) -> list[tuple[float, Any, float, float]]:
    """The exact top-DIRECT_ACTOR_SLOTS-nearest-by-raw-distance pool the
    observation's direct-actor block is built from. A target selection is
    only ever representable to the policy if it is drawn from exactly this
    pool -- not `_visible_candidates()`'s full (commonly 300+ wide) list,
    which is what `_nearest_reachable_actor_id` used to draw from (see
    module docstring's correction #3)."""
    return base_env._visible_candidates()[:DIRECT_ACTOR_SLOTS]


def _reachable_representable_targets(base_env: Any, candidates) -> list[tuple[int, float]]:
    """(actor_id, geodesic_distance_cells) for every representable
    candidate that currently has a finite (reachable) geodesic path."""
    player_cell = base_env.map.native_to_layout_cell(base_env.player_x, base_env.player_z)
    geodesic_field = base_env._geodesic_field(player_cell)
    result = []
    for _distance, actor, _dx, _dz in candidates:
        actor_cell = base_env.map.native_to_layout_cell(actor.x, actor.z)
        geodesic = math.inf if actor_cell is None else float(geodesic_field.get(actor_cell, math.inf))
        if math.isfinite(geodesic):
            result.append((actor.actor_id, geodesic))
    return result


def select_target(base_env: Any, *, sticky_id: int | None, sticky: bool) -> int | None:
    """Single source-of-truth target selection (2026-08-10 correction #3),
    shared by BOTH the policy's masked observation and the reward. Always
    selects from `_representable_candidates`, so a selected target can
    never be unrepresentable to the policy by construction. Ranks by
    geodesic (path) distance among representable candidates that are
    currently reachable, nearest first.

    sticky=True: retain `sticky_id` if it is still alive, still among the
    representable candidates, AND still reachable this tick; otherwise
    reselect the nearest reachable representable candidate and adopt it as
    the new sticky target.
    sticky=False (greedy): always reselect the nearest reachable
    representable candidate this tick, from the identical candidate pool
    sticky mode uses -- persistence is the ONLY difference between modes.

    Returns None if no representable candidate is currently reachable.
    """
    candidates = _representable_candidates(base_env)
    reachable = _reachable_representable_targets(base_env, candidates)
    if sticky and sticky_id is not None:
        for actor_id, _geodesic in reachable:
            if actor_id == sticky_id:
                return sticky_id
    if not reachable:
        return None
    return min(reachable, key=lambda pair: pair[1])[0]


def _mask_observation_to_target(obs: np.ndarray, base_env: Any, target_id: int | None) -> np.ndarray:
    """Return a COPY of `obs` with every direct-actor slot deactivated
    except whichever slot (if any) holds `target_id`."""
    masked = np.array(obs, copy=True)
    for slot, (_distance, actor, _dx, _dz) in enumerate(_representable_candidates(base_env)):
        if actor.actor_id == target_id:
            continue
        offset = DIRECT_ACTOR_START + slot * ACTOR_FEATURES + _ACTIVE_FIELD_OFFSET
        masked[offset] = 0.0
    return masked


def target_position(base_env: Any, target_id: int | None) -> tuple[float, float] | None:
    """World position of `target_id`, or None if it's not currently alive
    (or no target is selected)."""
    if target_id is None:
        return None
    for actor in base_env.actors:
        if actor.alive and actor.actor_id == target_id:
            return actor.x, actor.z
    return None


class PureNavigationWrapper(gym.Wrapper):
    """Episode terminates immediately with COLLISION_TERMINAL_REWARD on the
    first physical contact. Event action is always forced to NONE. Reward
    otherwise depends on `reward_mode` (see module docstring).

    Owns `_selected_target_id` directly (2026-08-10 correction #3) rather
    than reading it off the underlying env -- the wrapper is now the sole
    source of truth for which actor counts as "the target", for both the
    masked observation and the reward.

    CORRECTED 2026-08-10 (target-switch reward discontinuity): both terms
    of the progress delta are computed relative to the SAME (freshly
    reselected, current-tick) target: distance from the PREVIOUS player
    position to the CURRENT target, minus distance from the CURRENT player
    position to the CURRENT target. Well-defined and discontinuity-free
    regardless of whether the target changed this tick.
    """

    def __init__(self, env: Any, *, target_mode: TargetMode, reward_mode: RewardMode = "safety"):
        super().__init__(env)
        self.target_mode = target_mode
        self.reward_mode = reward_mode
        self._sticky = target_mode == "stable_waypoint"
        self._prev_contacts = 0
        self._prev_distance = 0.0
        self._prev_player_x = 0.0
        self._prev_player_z = 0.0
        self._selected_target_id: int | None = None

    def _mask(self, obs: np.ndarray) -> np.ndarray:
        return _mask_observation_to_target(obs, self.env.unwrapped, self._selected_target_id)

    def _distance_to_selected_target(self, player_x: float, player_z: float) -> float | None:
        base_env = self.env.unwrapped
        target = target_position(base_env, self._selected_target_id)
        if target is None:
            return None
        dx = target[0] - player_x
        dz = target[1] - player_z
        return math.hypot(dx, dz) / base_env.map.native_units_per_cell

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        base_env = self.env.unwrapped
        self._selected_target_id = select_target(base_env, sticky_id=None, sticky=self._sticky)
        self._prev_contacts = int(info.get("contacts", 0)) if isinstance(info, dict) else 0
        self._prev_distance = float(info.get("total_distance_cells", 0.0)) if isinstance(info, dict) else 0.0
        self._prev_player_x = float(base_env.player_x)
        self._prev_player_z = float(base_env.player_z)
        return self._mask(obs), info

    def step(self, action):
        action = np.asarray(action, dtype=np.int64).copy()
        action[1] = EVENT_NONE
        prev_player_x, prev_player_z = self._prev_player_x, self._prev_player_z
        obs, _reward, terminated, truncated, info = self.env.step(action)
        base_env = self.env.unwrapped
        contacts = int(info.get("contacts", 0))
        distance = float(info.get("total_distance_cells", 0.0))
        displacement = distance - self._prev_distance
        contact_this_tick = contacts > self._prev_contacts
        self._prev_contacts = contacts
        self._prev_distance = distance
        self._prev_player_x = float(base_env.player_x)
        self._prev_player_z = float(base_env.player_z)

        # Reselect BEFORE masking/reward, so both are always scored against
        # the exact same, freshly-valid target (single source of truth).
        self._selected_target_id = select_target(base_env, sticky_id=self._selected_target_id, sticky=self._sticky)
        obs = self._mask(obs)

        if contact_this_tick:
            return obs, COLLISION_TERMINAL_REWARD, True, truncated, info

        if self.reward_mode == "safety":
            reward = PROGRESS_REWARD_SCALE * max(0.0, displacement)
        else:
            prev_distance_to_target = self._distance_to_selected_target(prev_player_x, prev_player_z)
            new_distance_to_target = self._distance_to_selected_target(base_env.player_x, base_env.player_z)
            if prev_distance_to_target is not None and new_distance_to_target is not None:
                reward = GOAL_PROGRESS_REWARD_SCALE * (prev_distance_to_target - new_distance_to_target)
            else:
                reward = 0.0
        return obs, reward, terminated, truncated, info
