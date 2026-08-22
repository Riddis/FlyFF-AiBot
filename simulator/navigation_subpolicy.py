"""Frozen-navigation steering oracle for the canonical Basic -> Advanced
curriculum, and the training-time wrapper built on top of it.

RECOVERED ARCHITECTURE (2026-08-22, see docs/architecture/
CURRICULUM_TRAINING_PIPELINE.md section 4 for the full evidence trail):
the full-farming policy the canonical curriculum trains does NOT relearn
steering. Steering execution belongs entirely to the frozen navigation
checkpoint (``models/generalized_waypoint_both_seed2_0051200.zip``) driven
by the production router (``navigation.kinodynamic_route_planner``); the
curriculum's own trainable policy owns only the farming/event (EVA) action.

This module is a faithful extraction of the proven mechanism from the
2026-08-15 "post-router-fix complete bot" 850M monster-approach baseline
(``simulator/scratchpad/scratchpad_monster_approach_baseline_eval.py``) --
target selection is the environment's own native hysteresis
(``_nearest_reachable_actor_id``), ``plan_route`` runs once per target
acquisition/change, ``select_persistent_waypoint``/``TargetPersistenceController``
compress the fixed route into a near-term steering waypoint each tick, and
that waypoint is fed to the frozen checkpoint via a SYNTHETIC single-
candidate observation (``RecordedFarmingEnv._observation(candidates=[...])``)
without touching any real actor's state. The mechanics (synthetic-candidate
construction, side-effect-free observation override, previous-steering
threading) are ported unchanged from that already-validated code, not
reimplemented from scratch.

Two layers:

- ``FrozenNavigationSteering``: the low-level per-tick oracle. Used directly
  by Basic's hand-rolled rollout loop (``simulator/basic_environment.py``),
  which needs to see the proposed steering action BEFORE possibly letting
  ``RecoveryController`` override it -- the same "propose, then possibly
  override" shape recovery already uses for the (now-retired) trainable
  steering head.
- ``FrozenNavigationWrapper``: a ``gym.Wrapper`` built on top of it, exposing
  a ``Discrete(len(FarmingEvent))`` action space to the wrapped policy and
  driving steering automatically every tick. Used by Beginner/Intermediate/
  Advanced's PPO training and by evaluation, where there is no recovery
  controller in the loop (recovery is structurally absent from PPO training
  in this codebase, see ``simulator/basic_environment.py``'s module
  docstring) and no need to intercept the proposed action before it executes.
"""

from __future__ import annotations

import hashlib
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from farming.actions import FarmingEvent
from navigation.kinodynamic_route_planner import TargetPersistenceController, plan_route, select_persistent_waypoint
from navigation.movement_kernel import SteeringDirection

from navigation.navigation_evidence import RAW_OBSERVATION_SIZE

from .environment import RecordedFarmingEnv, SimActor

ROOT = Path(__file__).resolve().parents[1]
FROZEN_NAVIGATION_CHECKPOINT_PATH = ROOT / "models" / "generalized_waypoint_both_seed2_0051200.zip"
FROZEN_NAVIGATION_CHECKPOINT_SHA256 = "87bd8d3e0be88b7f243ad6c9b35ff6d3f8bde1f37b35334febf936ec115cda50"


def verify_frozen_navigation_checkpoint(path: Path = FROZEN_NAVIGATION_CHECKPOINT_PATH) -> str:
    """Refuses to proceed if the frozen checkpoint's bytes have changed --
    docs/agent/PROJECT_RULES.md section 6 (immutable artifacts). Returns the
    verified digest."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != FROZEN_NAVIGATION_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Frozen navigation checkpoint {path} has SHA-256 {digest}, expected "
            f"{FROZEN_NAVIGATION_CHECKPOINT_SHA256} -- refusing to load a checkpoint "
            "whose bytes do not match the qualified, frozen navigation checkpoint."
        )
    return digest


def _synthetic_candidate(base_env: RecordedFarmingEnv, waypoint: tuple[float, float]) -> tuple:
    """One synthetic single-candidate tuple for RecordedFarmingEnv._observation's
    `candidates=` override -- never touches any real actor. Ported unchanged
    from scratchpad_monster_approach_baseline_eval.py's `_synthetic_candidate`."""
    cell_size = base_env.map.native_units_per_cell
    dx_cells = (waypoint[0] - base_env.player_x) / cell_size
    dz_cells = (waypoint[1] - base_env.player_z) / cell_size
    direct_distance = math.hypot(dx_cells, dz_cells)
    virtual_actor = SimActor(actor_id=-1, x=waypoint[0], z=waypoint[1], alive=True)
    return (direct_distance, virtual_actor, dx_cells, dz_cells)


def _observation_without_side_effects(base_env: RecordedFarmingEnv, candidates: list) -> np.ndarray:
    """Computes an observation against a synthetic candidate list without
    perturbing the environment's own native target-selection hysteresis
    bookkeeping. Ported unchanged from scratchpad_monster_approach_baseline_
    eval.py's `_observation_without_side_effects` (verified byte-identical
    before/after in that already-validated mechanism)."""
    saved_best = base_env._best_group_actor_id
    saved_nearest = base_env._nearest_reachable_actor_id
    saved_potential = base_env._approach_potential_cells
    saved_history = deque(base_env._clearance_history, maxlen=base_env._clearance_history.maxlen)
    obs = base_env._observation(candidates=candidates)
    base_env._best_group_actor_id = saved_best
    base_env._nearest_reachable_actor_id = saved_nearest
    base_env._approach_potential_cells = saved_potential
    base_env._clearance_history = saved_history
    return obs


def _actor_position(base_env: RecordedFarmingEnv, actor_id: int) -> tuple[float, float] | None:
    for actor in base_env.actors:
        if actor.actor_id == actor_id and actor.alive:
            return (actor.x, actor.z)
    return None


@dataclass
class SteeringTickResult:
    steering: int
    waypoint: tuple[float, float]
    replanned: bool
    planner_failure: bool


class FrozenNavigationSteering:
    """Per-episode frozen-checkpoint steering oracle. One instance per
    episode (call `reset()` between episodes, or construct fresh)."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self._target_id: int | None = None
        self._route: list | None = None
        self._controller: TargetPersistenceController | None = None
        self._snapshot_pos: tuple[float, float] | None = None

    @classmethod
    def load_frozen(
        cls, *, device: str = "cpu", checkpoint_path: Path = FROZEN_NAVIGATION_CHECKPOINT_PATH,
    ) -> "FrozenNavigationSteering":
        from stable_baselines3 import PPO

        verify_frozen_navigation_checkpoint(checkpoint_path)
        model = PPO.load(str(checkpoint_path), device=device)
        return cls(model)

    def reset(self) -> None:
        self._target_id = None
        self._route = None
        self._controller = None
        self._snapshot_pos = None

    def _plan_to(self, base_env: RecordedFarmingEnv, actor_id: int) -> bool:
        pos = _actor_position(base_env, actor_id)
        if pos is None:
            return False
        route = plan_route(
            base_env.map, start_x=base_env.player_x, start_z=base_env.player_z, start_heading=base_env.heading,
            destination_x=pos[0], destination_z=pos[1],
        )
        if len(route) < 2:
            return False
        self._route = route
        self._snapshot_pos = pos
        self._controller = TargetPersistenceController(base_env.map, pos[0], pos[1])
        self._target_id = actor_id
        return True

    def steering_action(self, env: Any, *, target_actor_id: int) -> SteeringTickResult:
        """Computes this tick's steering action from the frozen checkpoint,
        driven by the production router's persistent-waypoint selection
        toward `target_actor_id`. Replans (a single `plan_route` call)
        whenever `target_actor_id` differs from the previously tracked
        target -- the qualified 850M-evaluation replan trigger (native
        target-selection change), never a continuous moving-target chase.

        `env` must be the NavigationHistoryWrapper-wrapped env (its own
        `_augment` carries the rolling temporal-sidecar history)."""
        base_env = env.unwrapped
        replanned = False
        if self._route is None or self._target_id != target_actor_id:
            if not self._plan_to(base_env, target_actor_id):
                return SteeringTickResult(
                    steering=int(SteeringDirection.NONE), waypoint=(base_env.player_x, base_env.player_z),
                    replanned=False, planner_failure=True,
                )
            replanned = True

        candidate = select_persistent_waypoint(
            base_env.map, self._route, player_x=base_env.player_x, player_z=base_env.player_z, heading=base_env.heading,
        )
        if candidate is None:
            candidate = self._snapshot_pos
        waypoint = self._controller.update(
            candidate, player_x=base_env.player_x, player_z=base_env.player_z, route=self._route,
        )

        synthetic = _synthetic_candidate(base_env, waypoint)
        obs_raw = _observation_without_side_effects(base_env, [synthetic])
        # previous_steering must be threaded explicitly, or NavigationHistoryWrapper.
        # _augment silently defaults it to NONE every tick after the first -- a real,
        # previously-shipped bug (2026-08-14, see MISTAKES.md).
        obs = env._augment(obs_raw, base_env.previous_steering)

        action, _state = self._model.predict(obs, deterministic=True)
        steering = int(np.asarray(action).reshape(-1)[0])
        return SteeringTickResult(steering=steering, waypoint=waypoint, replanned=replanned, planner_failure=False)


class FrozenNavigationWrapper(gym.Wrapper):
    """Training/evaluation-time wrapper exposing a `Discrete(len(FarmingEvent))`
    action space over a `Box(RAW_OBSERVATION_SIZE,)` observation: the wrapped
    (trainable) policy chooses only the farming/event action and sees only
    the raw 923-value observation -- exactly what `SplitBranchExtractor.
    event_net` has always trained on (`simulator/split_branch_policy.py`),
    so a checkpoint's event/value weights transfer to or from this wrapper's
    policy without any input-shape surgery. Every tick's steering is computed
    automatically by a `FrozenNavigationSteering` instance from the FULL
    928-value NavigationHistoryWrapper observation internally (steering needs
    the temporal/previous-steering sidecar; the trainable event-only policy
    does not).

    No recovery hook -- use `FrozenNavigationSteering` directly (as Basic's
    rollout loop does) where a propose-then-possibly-override step is
    needed; PPO training (Beginner onward) has no recovery in its loop by
    construction (see this package's basic_environment.py module
    docstring), so there is nothing to intercept here."""

    def __init__(self, env: Any, steering: FrozenNavigationSteering) -> None:
        super().__init__(env)
        self._steering = steering
        self.action_space = spaces.Discrete(len(FarmingEvent))
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(RAW_OBSERVATION_SIZE,), dtype=np.float32,
        )

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        self._steering.reset()
        return np.asarray(obs, dtype=np.float32)[:RAW_OBSERVATION_SIZE], info

    def step(self, event_action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        base_env = self.env.unwrapped
        target_id = base_env._nearest_reachable_actor_id
        tick_result: SteeringTickResult | None = None
        if target_id is None:
            # No reachable target this tick -- hold heading, let the
            # underlying environment's own reward/termination (timeout,
            # etc.) govern the episode; never a crash.
            steering = int(SteeringDirection.NONE)
        else:
            tick_result = self._steering.steering_action(self.env, target_actor_id=target_id)
            steering = tick_result.steering

        command = np.array([steering, int(event_action)], dtype=np.int64)
        obs, reward, terminated, truncated, info = self.env.step(command)
        info = dict(info)
        info["steering_replanned"] = tick_result.replanned if tick_result else False
        info["steering_planner_failure"] = tick_result.planner_failure if tick_result else False
        info["steering_waypoint"] = tick_result.waypoint if tick_result else None
        return np.asarray(obs, dtype=np.float32)[:RAW_OBSERVATION_SIZE], reward, terminated, truncated, info
