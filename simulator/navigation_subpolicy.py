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


def farming_policy_architecture_contract() -> dict[str, Any]:
    """Provenance fragment for every `SplitFarmingTargetEventPolicy`
    checkpoint (Basic through Advanced -- one shared architecture, docs/
    architecture/CURRICULUM_TRAINING_PIPELINE.md section 19): a full-farming
    checkpoint is not reproducible or even executable by itself without
    knowing WHICH frozen navigation checkpoint it was paired with (steering
    ownership, per section 4/6, lives entirely outside this checkpoint).
    Callers pass this as `simulator.run_provenance.build_run_manifest`'s
    `architecture_contract=` so it overrides that function's own
    `SplitSteeringNavigationPolicy`-shaped default. The navigation
    checkpoint's own implementation version is already covered by the
    manifest's top-level `git.commit` field (router/frozen-navigation code
    and this checkpoint's own training code are versioned together in the
    same repository state), so it is not duplicated here."""
    from farming.actions import FarmingEvent
    from farming.observation import DIRECT_ACTOR_SLOTS

    return {
        "policy_class": "SplitFarmingTargetEventPolicy",
        "action_contract": (
            f"MultiDiscrete([{DIRECT_ACTOR_SLOTS + 1}, {len(FarmingEvent)}]) -- "
            "[target_selection (0=KEEP, 1..12=direct-actor slot), FarmingEvent]; no steering action"
        ),
        "raw_observation_size": RAW_OBSERVATION_SIZE,
        "policy_input_schema_id": "923-value raw production observation contract (no navigation sidecar)",
        # Overrides build_run_manifest's own default_contract["policy_input_size"]
        # (928, the steering-policy navigation-sidecar width) -- this policy's
        # input IS the raw observation, unaugmented, so the two sizes are
        # equal. Without this explicit override the stale 928 default would
        # otherwise survive build_run_manifest's field-by-field dict.update()
        # merge even after raw_observation_size/policy_input_schema_id were
        # correctly overridden -- a real bug caught by
        # tests/test_basic_checkpoint_provenance.py.
        "policy_input_size": RAW_OBSERVATION_SIZE,
        "navigation_checkpoint_path": str(FROZEN_NAVIGATION_CHECKPOINT_PATH),
        "navigation_checkpoint_sha256": FROZEN_NAVIGATION_CHECKPOINT_SHA256,
        "navigation_ownership": "steering is FrozenNavigationSteering's output; this checkpoint never samples/logs it",
    }


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
    bookkeeping. Ported from scratchpad_monster_approach_baseline_eval.py's
    `_observation_without_side_effects` (verified byte-identical before/
    after in that already-validated mechanism); extended to also save/
    restore `_direct_actor_slot_ids` (added to `_observation()` for the
    learned farming-target-selection action, docs/architecture/
    CURRICULUM_TRAINING_PIPELINE.md section 4/6) -- without this, the
    synthetic single-candidate waypoint call here would overwrite the real
    slot->actor_id mapping with a bogus one built from the synthetic
    actor_id=-1 candidate, exactly the kind of navigation-only-state leak
    into farming-target selection this function exists to prevent for
    every other piece of hysteresis bookkeeping."""
    saved_best = base_env._best_group_actor_id
    saved_nearest = base_env._nearest_reachable_actor_id
    saved_potential = base_env._approach_potential_cells
    saved_slot_ids = base_env._direct_actor_slot_ids
    saved_history = deque(base_env._clearance_history, maxlen=base_env._clearance_history.maxlen)
    obs = base_env._observation(candidates=candidates)
    base_env._best_group_actor_id = saved_best
    base_env._nearest_reachable_actor_id = saved_nearest
    base_env._approach_potential_cells = saved_potential
    base_env._direct_actor_slot_ids = saved_slot_ids
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
    docstring), so there is nothing to intercept here.

    Reward: the underlying environment's "approach" reward component
    (`simulator.reward_model`, movement progress toward the target -- purely
    a function of the executed steering action) is excluded from the reward
    this wrapper returns (docs/architecture/CURRICULUM_TRAINING_PIPELINE.md
    section 15): it would otherwise reward/penalize this policy for a
    physical outcome only FrozenNavigationSteering's action determines. The
    full reward and its component breakdown remain available in `info` for
    diagnostics -- only what PPO actually optimizes against is adjusted."""

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
        # The "approach" reward component (simulator.reward_model.
        # SimulatorRewardConfig.approach_reward_scale) rewards movement
        # progress toward the target -- a quantity determined entirely by
        # the executed STEERING action (FarmingEvent never moves the
        # player), which this wrapped policy never samples or influences.
        # Left in the training signal, it would reward/penalize the
        # event-only policy for a physical outcome its own action cannot
        # change -- exactly the "navigation-specific auxiliary reward that
        # exists solely to teach steering" docs/architecture/
        # CURRICULUM_TRAINING_PIPELINE.md section 4 already established must
        # not remain trainable. Excluded here only (this wrapper's own
        # returned reward, i.e. what enters the PPO rollout buffer) -- the
        # underlying environment reward/reward_components in `info` are
        # untouched, so every other consumer (Basic training/evaluation,
        # milestone_evaluator, teacher rollouts) still sees the full reward.
        reward_components = info.get("reward_components") or {}
        approach_component = float(reward_components.get("approach", 0.0))
        event_only_reward = float(reward) - approach_component
        info["raw_reward_before_navigation_exclusion"] = float(reward)
        info["navigation_reward_excluded"] = approach_component
        info["steering_replanned"] = tick_result.replanned if tick_result else False
        info["steering_planner_failure"] = tick_result.planner_failure if tick_result else False
        info["steering_waypoint"] = tick_result.waypoint if tick_result else None
        return np.asarray(obs, dtype=np.float32)[:RAW_OBSERVATION_SIZE], event_only_reward, terminated, truncated, info


def _farming_policy_forward(net: Any, observation_raw: np.ndarray) -> tuple[int, int]:
    """Forward pass for `SplitFarmingTargetEventPolicy`
    (`simulator.split_branch_policy`): `MultiDiscrete([TARGET_ACTION_SIZE,
    len(FarmingEvent)])` over the plain raw observation -- two categorical
    heads, target first."""
    import torch

    with torch.no_grad():
        obs_t = torch.as_tensor(
            np.asarray(observation_raw, dtype=np.float32)[None, :RAW_OBSERVATION_SIZE],
            dtype=torch.float32, device=net.device,
        )
        dist = net.get_distribution(obs_t).distribution
        target_probs = dist[0].probs[0].cpu().numpy()
        event_probs = dist[1].probs[0].cpu().numpy()
    return int(target_probs.argmax()), int(event_probs.argmax())


def run_composed_episode(
    curriculum_path: str, layout_name: str, *, farming_policy: Any, navigation_steering: FrozenNavigationSteering,
    seed: int, episode_seconds: float, max_actions: int, stage: str = "early",
) -> dict[str, Any]:
    """THE canonical composed-episode rollout for every curriculum stage
    (Basic through Advanced -- all four now share one `SplitFarmingTarget
    EventPolicy` architecture, docs/architecture/CURRICULUM_TRAINING_
    PIPELINE.md section 4/6/12) and every evaluator that grades them:
    target selection AND event come from `farming_policy`'s own forward
    pass; steering comes from `navigation_steering` (production router +
    frozen 0051200), driven toward whatever the policy's OWN target action
    resolves to via a `PersistentFarmingTarget` -- never the environment's
    own deterministic best-group/nearest-reachable hysteresis, which is
    available only as a candidate/reachability source, never the runtime
    decision-maker. Composed exactly like `farming_target_policy.
    FarmingPolicyWrapper.step`, but as a plain function returning the same
    rich per-episode result dict `milestone_evaluator.run_episode` does (so
    `milestone_evaluator._summarize_episodes` and the density-binned EVA
    report work unchanged) -- this is what lets `milestone_evaluator.py`,
    `basic_milestone_evaluator.py`, and `beginner_transition.py`'s
    diagnostics share ONE composition implementation instead of four.

    No recovery -- matches training exactly: PPO training (Beginner onward)
    has no recovery in its loop by construction, and Basic's own evaluation
    already uses `_roll_basic_episode` (a separate, recovery-AWARE function)
    for its assisted-mode metrics; this function is for raw/unassisted
    evaluation only (see `simulator/basic_environment.py`'s module
    docstring for the full recovery/PPO rationale).

    `steering_agreement`/`event_agreement`/`target_agreement` compare the
    executed steering/event/target against the scripted teacher's own
    direct-bearing suggestion and `deterministic_target_teacher_action` --
    informational only (the frozen navigator's route-following steering and
    the deterministic target heuristic are DIFFERENT mechanisms from the
    teacher's direct-bearing suggestion by design, so some disagreement is
    expected, not a defect). `corr_angle_p_left`/`corr_angle_p_right` are
    always None: steering is FrozenNavigationSteering's deterministic
    output, not a learned probability distribution to correlate against
    target angle."""

    from .farming_target_policy import PersistentFarmingTarget, deterministic_target_teacher_action
    from .milestone_evaluator import _contact_event_stats
    from .movement_classification import classify_episode_movement
    from .navigation_history import NavigationHistoryWrapper
    from .scripted_policies import scripted_command
    from .synthetic import iter_variant_environments

    entry, base_env = next(iter(iter_variant_environments(
        curriculum_path, stage=stage, seed=seed, episode_steps=max_actions,
        episode_seconds=episode_seconds, variant_name=layout_name,
    )))
    env = NavigationHistoryWrapper(base_env)
    observation, _info = env.reset(seed=seed)
    navigation_steering.reset()
    target_tracker = PersistentFarmingTarget()

    steering_choices: list[int] = []
    unique_cells_trace: list[int] = []
    total_distance_trace: list[float] = []
    contacts_trace: list[int] = []
    steering_matches: list[bool] = []
    event_matches: list[bool] = []
    target_matches: list[bool] = []
    invalid_target_selections = 0
    target_invalidations_by_planner_failure = 0
    eva_target_counts: list[int] = []
    teacher_events: list[int] = []
    policy_events: list[int] = []
    geodesic_euclidean_disagreements = 0
    geodesic_euclidean_total = 0
    info: dict[str, Any] = {}

    for _ in range(int(max_actions)):
        teacher_command = scripted_command("obstacle_aware", base_env)
        teacher_target_action = deterministic_target_teacher_action(base_env)
        candidates = base_env._visible_candidates()
        if candidates:
            player_cell = base_env.map.native_to_layout_cell(base_env.player_x, base_env.player_z)
            geodesic_field = base_env._geodesic_field(player_cell)
            _potential, best_actor_id = base_env._group_approach_potential(candidates, geodesic_field)
            if best_actor_id is not None:
                geodesic_euclidean_total += 1
                if candidates[0][1].actor_id != best_actor_id:
                    geodesic_euclidean_disagreements += 1

        policy_target_action, event = _farming_policy_forward(farming_policy, np.asarray(observation, dtype=np.float32))
        resolved_target_id, invalid_selection = target_tracker.apply_action(base_env, policy_target_action)
        if invalid_selection:
            invalid_target_selections += 1
        if resolved_target_id is None:
            steering = int(SteeringDirection.NONE)
        else:
            tick_result = navigation_steering.steering_action(env, target_actor_id=resolved_target_id)
            steering = tick_result.steering
            if tick_result.planner_failure:
                # Same invalidation rule as FarmingPolicyWrapper.step: the
                # production router could not produce a route to this still
                # alive/present target right now -- objectively non-
                # navigable under the actual planner/reachability contract.
                # Clear both the tracker's target and the steering oracle's
                # own stale route/controller state.
                target_tracker.invalidate()
                navigation_steering.reset()
                resolved_target_id = None
                target_invalidations_by_planner_failure += 1

        steering_choices.append(steering)
        steering_matches.append(steering == int(teacher_command.steering))
        event_matches.append(event == int(teacher_command.event))
        target_matches.append(policy_target_action == teacher_target_action)
        eva_target_counts.append(int(base_env.eva_target_count()))
        teacher_events.append(int(teacher_command.event))
        policy_events.append(event)

        observation, _reward, terminated, truncated, info = env.step(np.asarray([steering, event], dtype=np.int64))
        unique_cells_trace.append(int(info["unique_cells"]))
        total_distance_trace.append(float(info["total_distance_cells"]))
        contacts_trace.append(int(info["contacts"]))
        if terminated or truncated:
            break

    env.close()

    movement = classify_episode_movement(
        steering_choices=steering_choices, unique_cells_trace=unique_cells_trace, total_distance_trace=total_distance_trace,
    )

    return {
        "layout": layout_name, "seed": int(seed), "steps": len(steering_choices),
        "kills_per_simulated_hour": float(info.get("total_kills", 0)) * 3600.0 / max(1e-9, float(info.get("elapsed_seconds", 0.0))),
        "total_kills": int(info.get("total_kills", 0)),
        "valid_eva_casts": int(info.get("valid_eva_casts", 0)),
        "invalid_eva_attempts": int(info.get("invalid_eva_attempts", 0)),
        "missed_eva_opportunities": int(info.get("missed_eva_opportunities", 0)),
        "contacts": int(info.get("contacts", 0)),
        "contacts_per_100_distance": float(info.get("contacts", 0)) * 100.0 / max(1e-9, float(info.get("total_distance_cells", 0.0))),
        "total_distance_cells": float(info.get("total_distance_cells", 0.0)),
        "unique_cells": int(info.get("unique_cells", 0)),
        "path_efficiency": float(info.get("path_efficiency", 0.0)),
        "steering_agreement": float(np.mean(steering_matches)) if steering_matches else None,
        "event_agreement": float(np.mean(event_matches)) if event_matches else None,
        "target_agreement": float(np.mean(target_matches)) if target_matches else None,
        "invalid_target_selection_rate": float(invalid_target_selections) / max(1, len(steering_choices)),
        "target_invalidations_by_planner_failure": int(target_invalidations_by_planner_failure),
        "corr_angle_p_left": None,
        "corr_angle_p_right": None,
        "geodesic_euclidean_disagreement_rate": (
            geodesic_euclidean_disagreements / geodesic_euclidean_total if geodesic_euclidean_total else None
        ),
        "zero_kill": bool(info.get("total_kills", 0) == 0),
        "recovery": None,
        **movement,
        **_contact_event_stats(contacts_trace),
        "_eva_target_counts": eva_target_counts,
        "_teacher_events": teacher_events,
        "_policy_events": policy_events,
    }
