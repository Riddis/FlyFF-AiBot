"""Basic -> Beginner transition: disable recovery, switch to PPO.

Beginner never uses recovery -- balanced_training_vec_env_phase2 (imported
from simulator.navigation_ppo, unchanged) never wraps its training envs with
RecoveryController, so resume_ppo_chunk_phase2's rollout buffer is always
faithful (the policy's own sampled action is always what gets executed) and
its on-policy assumption is never at risk. See simulator.basic_environment's
module docstring for the full recovery/PPO design rationale this depends on.

A zero-shot raw (recovery-off) rollout right after Basic graduation is a
DIAGNOSTIC starting point for Beginner, never a Basic graduation
requirement -- see simulator.basic_milestone_evaluator's module docstring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np


class _EventOnlySpaceProbe(gym.Env):
    """Minimal probe env exposing only the event-only training contract's
    spaces (`Discrete(len(FarmingEvent))` over `Box(RAW_OBSERVATION_SIZE,)`)
    -- used solely to construct a fresh `ActorCriticPolicy` with the right
    shapes before `transfer_event_head_to_event_only_policy` overwrites its
    weights in place (`build_event_only_ppo_from_basic_checkpoint`). Never
    stepped for real: no simulator state, no real episode. Must genuinely
    subclass `gymnasium.Env` -- SB3's `PPO.__init__` rejects a duck-typed
    env that only exposes `observation_space`/`action_space`/`reset`/`step`
    without the real base class."""

    metadata: dict = {"render_modes": []}

    def __init__(self) -> None:
        from gymnasium import spaces

        from navigation.navigation_evidence import RAW_OBSERVATION_SIZE

        from farming.actions import FarmingEvent

        super().__init__()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(RAW_OBSERVATION_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(FarmingEvent))
        self._raw_size = RAW_OBSERVATION_SIZE

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        return np.zeros(self._raw_size, dtype=np.float32), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        return np.zeros(self._raw_size, dtype=np.float32), 0.0, True, False, {}


def build_event_only_ppo_from_basic_checkpoint(
    basic_checkpoint: str | Path,
    *,
    event_net_arch: list[int] | None = None,
    vf_net_arch: list[int] | None = None,
    seed: int = 0,
    device: str = "cpu",
    n_steps: int = 256,
    batch_size: int = 128,
    n_epochs: int = 4,
    learning_rate: float = 5e-5,
    clip_range: float = 0.10,
    target_kl: float = 0.015,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    ent_coef: float = 0.015,
) -> Any:
    """The Basic -> Beginner checkpoint bridge, end to end (docs/architecture/
    CURRICULUM_TRAINING_PIPELINE.md section 4/5): loads `basic_checkpoint`
    (a graduated `SplitSteeringNavigationPolicy`, `MultiDiscrete([3,3])`
    dual-head checkpoint), constructs a fresh event-only PPO
    (`Discrete(len(FarmingEvent))`, plain `ActorCriticPolicy`,
    project-standard conservative PPO hyperparameters -- same defaults as
    `simulator.basic_training.build_fresh_basic_policy`, for the same reason:
    so the returned checkpoint is immediately safe to hand to a PPO chunk
    without a separate hyperparameter-repair step), and transplants the
    source's event/value weights into it via `simulator.
    factorized_v193_training.transfer_event_head_to_event_only_policy`
    (proven bit-for-bit event-distribution-identical, see
    `tests/test_event_head_transplant.py`). The source's steering branch is
    discarded entirely -- it was never trained (see `build_fresh_basic_
    policy`'s docstring). `event_net_arch`/`vf_net_arch` must match the
    source checkpoint's own architecture or the transplant raises a shape
    mismatch (SplitSteeringNavigationPolicy's default `[64, 32]`/`[64, 32]`
    unless the source was built with something else).

    Returns the fresh event-only PPO model, not yet saved -- the caller
    saves it (with provenance) to establish the canonical Beginner starting
    checkpoint."""

    from stable_baselines3 import PPO

    from .factorized_v193_training import transfer_event_head_to_event_only_policy

    source = PPO.load(str(basic_checkpoint), device=device)
    probe_env = _EventOnlySpaceProbe()
    model = PPO(
        "MlpPolicy", probe_env,
        policy_kwargs={"net_arch": dict(pi=event_net_arch or [64, 32], vf=vf_net_arch or [64, 32])},
        seed=int(seed), device=device,
        # Same conservative project-standard hyperparameters as
        # build_fresh_basic_policy -- see that function's docstring.
        n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, learning_rate=learning_rate,
        clip_range=clip_range, target_kl=target_kl, gamma=gamma, gae_lambda=gae_lambda, ent_coef=ent_coef,
    )
    transfer_event_head_to_event_only_policy(source.policy, model.policy)
    return model


def save_event_only_checkpoint_with_provenance(
    model: Any,
    checkpoint_path: str | Path,
    *,
    basic_checkpoint: str | Path,
    seed: int,
    canonical_stage: str = "beginner",
) -> Path:
    """Saves the Basic -> Beginner event-only bridge checkpoint
    (`build_event_only_ppo_from_basic_checkpoint`'s output) together with
    its provenance manifest -- the one artifact in the event-only lineage
    that has no natural PPO-chunk/rehearsal call site of its own to write
    one (`continue_event_only_ppo_chunk`/`rehearse_event_only_on_basic_data`
    both write their own). Without this, the very first event-only
    checkpoint would be the one link in the chain untraceable back to which
    Basic checkpoint's event weights it carries."""

    from .navigation_subpolicy import event_only_architecture_contract
    from .run_provenance import build_run_manifest, write_run_manifest

    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(checkpoint))
    manifest = build_run_manifest(
        stage=canonical_stage, milestone="event_only_bridge", seeds=seed,
        config={"action_contract": "event_only", "source": "transfer_event_head_to_event_only_policy"},
        architecture_contract=event_only_architecture_contract(),
        starting_checkpoint=str(Path(basic_checkpoint).resolve()), output_checkpoint=str(checkpoint.resolve()),
    )
    write_run_manifest(checkpoint, manifest)
    return checkpoint


def continue_event_only_ppo_chunk(
    checkpoint: str | Path,
    output: str | Path,
    *,
    curriculum: str | Path,
    timesteps: int,
    stage: str = "early",
    seed: int = 0,
    episode_seconds: float = 150.0,
    max_actions: int = 1000,
    device: str = "cpu",
    progress_every_seconds: float = 15.0,
    canonical_stage: str = "beginner",
) -> dict[str, Any]:
    """One bounded event-only PPO chunk on an already-event-only checkpoint
    (the output of `build_event_only_ppo_from_basic_checkpoint`, or a prior
    round's own output), recovery off throughout (structural, matching
    `graduate_basic_to_beginner`'s reasoning), steering externally driven by
    `simulator.navigation_ppo.balanced_training_vec_env_event_only`'s
    `FrozenNavigationWrapper` composition -- never sampled by or logged from
    this policy. This is the event-only counterpart of
    `graduate_basic_to_beginner`, used by Beginner/Intermediate/Advanced
    alike (only the curriculum/checkpoint lineage differs -- same
    `canonical_stage` convention as that function)."""

    from .navigation_ppo import resume_ppo_chunk_event_only
    from .progress_reporting import SB3ProgressCallback
    from .run_provenance import build_run_manifest, write_run_manifest

    conservative_ppo_hparams = {
        "n_steps": 256, "batch_size": 128, "n_epochs": 4, "learning_rate": 5e-5,
        "clip_range": 0.10, "target_kl": 0.015, "gamma": 0.995, "gae_lambda": 0.95, "ent_coef": 0.015,
    }
    result = resume_ppo_chunk_event_only(
        checkpoint=checkpoint, curriculum=curriculum, output=output, timesteps=timesteps,
        stage=stage, seed=seed, episode_seconds=episode_seconds, max_actions=max_actions, device=device,
        callback=SB3ProgressCallback(int(timesteps), label=f"{canonical_stage}_ppo_event_only", min_interval_seconds=progress_every_seconds),
    )
    from .navigation_subpolicy import event_only_architecture_contract

    manifest = build_run_manifest(
        stage=canonical_stage, milestone="ppo_chunk", seeds=seed,
        config={"timesteps": timesteps, "stage": stage, "episode_seconds": episode_seconds,
                "max_actions": max_actions, "action_contract": "event_only", **conservative_ppo_hparams},
        curriculum_path=str(curriculum), recovery_config={"enabled": False, "reason": "structural -- see module docstring"},
        architecture_contract=event_only_architecture_contract(),
        starting_checkpoint=str(Path(checkpoint).resolve()), output_checkpoint=result["checkpoint_out"],
    )
    write_run_manifest(result["checkpoint_out"], manifest)
    return result


def _dual_head_event_forward(net: Any, observation_full: Any) -> int:
    """`event_forward` for `navigation_subpolicy.run_composed_episode` when
    `event_policy` is Basic's own dual-head `SplitSteeringNavigationPolicy`
    (not a plain event-only `ActorCriticPolicy`): reads only the event
    branch on the FULL observation (event_net has always consumed it,
    unsliced) -- the steering branch is never read, matching the recovered
    architecture's ownership split (steering belongs to `navigation_
    steering`, not to this net, even for Basic's own zero-shot diagnostic)."""
    import numpy as np
    import torch

    with torch.no_grad():
        obs_t = torch.as_tensor(
            np.asarray(observation_full, dtype=np.float32)[None, :], dtype=torch.float32, device=net.device,
        )
        event_probs = net.get_distribution(obs_t).distribution[1].probs[0].cpu().numpy()
    return int(event_probs.argmax())


def _aggregate_raw_diagnostic(checkpoint: str | Path, per_layout_results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    per_layout: dict[str, Any] = {}
    for layout_name, results in per_layout_results.items():
        per_layout[layout_name] = {
            "n_episodes": len(results),
            "physical_stagnation_episodes": sum(1 for r in results if r["physical_stagnation"]),
            "mean_contacts_per_100_distance": float(
                sum(r["contacts_per_100_distance"] for r in results) / max(1, len(results))
            ),
        }
    return {
        "role": "beginner_starting_point_diagnostic",
        "checkpoint": str(Path(checkpoint).resolve()),
        "per_layout": per_layout,
        "notes": "Diagnostic only -- not a Basic graduation requirement, not a Beginner graduation result.",
    }


def zero_shot_raw_diagnostic(
    checkpoint: str | Path,
    *,
    heldout_manifest_path: str,
    seeds: list[int],
    episode_seconds: float = 150.0,
    max_actions: int = 1000,
) -> dict[str, Any]:
    """Raw-policy (recovery disabled), Beginner-style evaluation of a
    just-graduated Basic checkpoint, against the existing immutable
    held-out pool. Purely informational: how much of Beginner's job is
    already done vs. still to learn. Never a pass/fail gate on Basic.

    Steering comes from `FrozenNavigationSteering`, exactly as it does in
    Basic's own assisted rollout (`simulator.basic_environment.
    _roll_basic_episode`) -- `checkpoint`'s own steering head is never read
    (docs/architecture/CURRICULUM_TRAINING_PIPELINE.md section 4/7: the
    direct-bearing/net's-own-steering path this replaced is superseded, not
    merely an alternative). Sequential/in-process -- see
    `zero_shot_raw_diagnostic_parallel` for a multi-process equivalent."""

    from stable_baselines3 import PPO

    from .curriculum_manifests import load_heldout_manifest
    from .navigation_subpolicy import FrozenNavigationSteering, run_composed_episode

    model = PPO.load(str(checkpoint), device="cpu")
    net = model.policy
    navigation_steering = FrozenNavigationSteering.load_frozen(device="cpu")
    manifest = load_heldout_manifest(heldout_manifest_path)

    per_layout_results: dict[str, list[dict[str, Any]]] = {}
    for layout_name in manifest.layouts:
        per_layout_results[layout_name] = [
            run_composed_episode(
                manifest.curriculum_path, layout_name, event_policy=net, navigation_steering=navigation_steering,
                seed=seed, episode_seconds=episode_seconds, max_actions=max_actions, stage=manifest.stage,
                event_forward=_dual_head_event_forward,
            )
            for seed in seeds
        ]
    return _aggregate_raw_diagnostic(checkpoint, per_layout_results)


_PARALLEL_WORKER_NET: Any = None
_PARALLEL_WORKER_NAVIGATION_STEERING: Any = None


def _init_raw_diagnostic_worker(checkpoint_path: str) -> None:
    global _PARALLEL_WORKER_NET, _PARALLEL_WORKER_NAVIGATION_STEERING
    from stable_baselines3 import PPO
    from .navigation_subpolicy import FrozenNavigationSteering
    _PARALLEL_WORKER_NET = PPO.load(checkpoint_path, device="cpu").policy
    _PARALLEL_WORKER_NAVIGATION_STEERING = FrozenNavigationSteering.load_frozen(device="cpu")


def _raw_diagnostic_worker_task(curriculum_path: str, layout_name: str, seed: int, episode_seconds: float, max_actions: int, stage: str = "early") -> dict[str, Any]:
    from .navigation_subpolicy import run_composed_episode
    return run_composed_episode(
        curriculum_path, layout_name, event_policy=_PARALLEL_WORKER_NET,
        navigation_steering=_PARALLEL_WORKER_NAVIGATION_STEERING,
        seed=seed, episode_seconds=episode_seconds, max_actions=max_actions, stage=stage,
        event_forward=_dual_head_event_forward,
    )


def zero_shot_raw_diagnostic_parallel(
    checkpoint: str | Path,
    *,
    heldout_manifest_path: str,
    seeds: list[int],
    episode_seconds: float = 150.0,
    max_actions: int = 1000,
    n_workers: int = 4,
) -> dict[str, Any]:
    """Same report as `zero_shot_raw_diagnostic`, computed by farming
    episodes out across `n_workers` OS processes instead of one sequential
    loop -- see `evaluate_basic_milestone_parallel`'s docstring for the
    same compute-for-wallclock tradeoff rationale."""

    from concurrent.futures import ProcessPoolExecutor

    from .curriculum_manifests import load_heldout_manifest

    manifest = load_heldout_manifest(heldout_manifest_path)
    tasks = [(manifest.curriculum_path, layout_name, seed, episode_seconds, max_actions, manifest.stage)
              for layout_name in manifest.layouts for seed in seeds]
    with ProcessPoolExecutor(
        max_workers=max(1, n_workers), initializer=_init_raw_diagnostic_worker, initargs=(str(checkpoint),),
    ) as pool:
        flat_results = list(pool.map(_raw_diagnostic_worker_task, *zip(*tasks)))

    per_layout_results: dict[str, list[dict[str, Any]]] = {name: [] for name in manifest.layouts}
    for task, result in zip(tasks, flat_results):
        per_layout_results[task[1]].append(result)
    return _aggregate_raw_diagnostic(checkpoint, per_layout_results)


def run_episode_925(
    curriculum_path: str | Path,
    layout_name: str,
    *,
    net: Any,
    seed: int,
    episode_seconds: float,
    max_actions: int,
    recovery: Any = None,
    stage: str = "early",
) -> dict[str, Any]:
    """925-dim-aware counterpart to milestone_evaluator.run_episode.

    SUPERSEDED for canonical Beginner/Intermediate/Advanced evaluation as of
    the frozen-navigation-sub-policy recovery (docs/architecture/
    CURRICULUM_TRAINING_PIPELINE.md section 4/10): this function drives
    `net`'s OWN steering head directly, which is exactly the direct-bearing
    path those stages no longer train (their checkpoints are event-only and
    have no steering head at all). `simulator.tools.RUN_CANONICAL_BEGINNER.
    py`/`RUN_CANONICAL_INTERMEDIATE.py`/`RUN_CANONICAL_ADVANCED.py` use
    `milestone_evaluator.evaluate_heldout`/`evaluate_challenge` with
    `navigation_steering=`/`use_frozen_navigation=True` instead (dispatching
    through `navigation_subpolicy.run_composed_episode`). Kept, not deleted,
    as a still-correct general-purpose evaluator for any
    SplitSteeringNavigationPolicy-shaped dual-head checkpoint that
    genuinely owns its own steering (e.g. re-evaluating a pre-recovery
    historical checkpoint) -- not what current canonical training produces.

    milestone_evaluator.run_episode drives iter_variant_environments' raw
    923-value env unwrapped -- correct for the standard production
    contract, but the wrong input width for a SplitSteeringNavigationPolicy
    (which expects the 925-value NavigationHistoryWrapper-augmented
    observation Basic/Beginner training already use; see
    _run_925_episode_raw's docstring for the identical reasoning applied to
    the simpler zero-shot diagnostic). Mirrors run_episode's exact
    tick-by-tick logic (angle tracking, geodesic-vs-euclidean disagreement,
    recovery integration, movement classification) verbatim, changing only
    where the observation comes from: env.reset()/env.step() on the WRAPPED
    env for the 925-value observation fed to the policy, base_env for every
    raw-simulator-API call (player_x/z, heading, map, _visible_candidates,
    _geodesic_field, eva_target_count) -- the same keep-both-references
    pattern basic_environment._roll_basic_episode already established for
    exactly this reason.
    """
    import numpy as np

    from .milestone_evaluator import _contact_event_stats, _is_durable_recovery, _policy_forward
    from .movement_classification import classify_episode_movement
    from .navigation_history import NavigationHistoryWrapper
    from .scripted_policies import scripted_command
    from .synthetic import iter_variant_environments

    entry, base_env = next(iter(iter_variant_environments(
        str(curriculum_path), stage=stage, seed=seed, episode_steps=max_actions,
        episode_seconds=episode_seconds, variant_name=layout_name,
    )))
    env = NavigationHistoryWrapper(base_env)
    observation, _ = env.reset(seed=seed)

    steering_choices: list[int] = []
    unique_cells_trace: list[int] = []
    total_distance_trace: list[float] = []
    contacts_trace: list[int] = []
    steering_matches: list[bool] = []
    event_matches: list[bool] = []
    angles: list[float] = []
    left_probs: list[float] = []
    right_probs: list[float] = []
    eva_target_counts: list[int] = []
    teacher_events: list[int] = []
    policy_events: list[int] = []
    geodesic_euclidean_disagreements = 0
    geodesic_euclidean_total = 0
    recovery_kills_during_intervention = 0
    info: dict[str, Any] = {}
    previous_distance = 0.0
    previous_contacts = 0

    for _ in range(int(max_actions)):
        angle = base_env.best_group_relative_angle()
        teacher_command = scripted_command("obstacle_aware", base_env)
        candidates = base_env._visible_candidates()
        if candidates:
            player_cell = base_env.map.native_to_layout_cell(base_env.player_x, base_env.player_z)
            geodesic_field = base_env._geodesic_field(player_cell)
            _potential, best_actor_id = base_env._group_approach_potential(candidates, geodesic_field)
            if best_actor_id is not None:
                geodesic_euclidean_total += 1
                if candidates[0][1].actor_id != best_actor_id:
                    geodesic_euclidean_disagreements += 1

        steering, event, steering_probs = _policy_forward(net, np.asarray(observation, dtype=np.float32))

        was_recovering = recovery is not None and recovery.state.value == "recovering"
        if recovery is not None:
            steering, event = recovery.step(
                tick=len(steering_choices),
                player_x=base_env.player_x, player_z=base_env.player_z, heading=base_env.heading,
                displacement_this_tick=info.get("total_distance_cells", 0.0) - previous_distance if info else 0.0,
                contact_this_tick=bool(info.get("contacts", 0) - previous_contacts) if info else False,
                map_model=base_env.map, policy_steering=steering, policy_event=event,
            )

        steering_choices.append(steering)
        steering_matches.append(steering == int(teacher_command.steering))
        event_matches.append(event == int(teacher_command.event))
        eva_target_counts.append(int(base_env.eva_target_count()))
        teacher_events.append(int(teacher_command.event))
        policy_events.append(event)
        if angle is not None:
            angles.append(angle)
            left_probs.append(float(steering_probs[1]))
            right_probs.append(float(steering_probs[2]))

        kills_before = int(info.get("total_kills", 0)) if info else 0
        previous_distance = float(info.get("total_distance_cells", 0.0)) if info else 0.0
        previous_contacts = int(info.get("contacts", 0)) if info else 0
        observation, _reward, terminated, truncated, info = env.step(np.asarray([steering, event], dtype=np.int64))
        if was_recovering:
            recovery_kills_during_intervention += int(info["total_kills"]) - kills_before
        unique_cells_trace.append(int(info["unique_cells"]))
        total_distance_trace.append(float(info["total_distance_cells"]))
        contacts_trace.append(int(info["contacts"]))
        if terminated or truncated:
            break

    env.close()

    movement = classify_episode_movement(
        steering_choices=steering_choices, unique_cells_trace=unique_cells_trace, total_distance_trace=total_distance_trace,
    )

    recovery_summary: dict[str, Any] | None = None
    if recovery is not None:
        durable_count = sum(
            1 for r in recovery.interventions
            if r.outcome == "recovered" and _is_durable_recovery(
                r.end_tick, unique_cells_trace=unique_cells_trace, contacts_trace=contacts_trace,
                total_distance_trace=total_distance_trace,
            )
        )
        recovery_summary = {
            "final_state": recovery.state.value,
            "intervention_count": len(recovery.interventions),
            "recovered_count": sum(1 for r in recovery.interventions if r.outcome == "recovered"),
            "durably_recovered_count": durable_count,
            "gave_up_count": sum(1 for r in recovery.interventions if r.outcome == "gave_up"),
            "intervention_durations": [
                (r.end_tick - r.trigger_tick) for r in recovery.interventions if r.end_tick is not None
            ],
            "kills_during_intervention": int(recovery_kills_during_intervention),
        }

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
        "corr_angle_p_left": float(np.corrcoef(angles, left_probs)[0, 1]) if len(angles) > 5 else None,
        "corr_angle_p_right": float(np.corrcoef(angles, right_probs)[0, 1]) if len(angles) > 5 else None,
        "geodesic_euclidean_disagreement_rate": (
            geodesic_euclidean_disagreements / geodesic_euclidean_total if geodesic_euclidean_total else None
        ),
        "zero_kill": bool(info.get("total_kills", 0) == 0),
        "recovery": recovery_summary,
        **movement,
        **_contact_event_stats(contacts_trace),
        "_eva_target_counts": eva_target_counts,
        "_teacher_events": teacher_events,
        "_policy_events": policy_events,
    }


def evaluate_heldout_925(
    model: Any, manifest: Any, *, seeds: list[int], episode_seconds: float, max_actions: int, use_recovery: bool = False,
) -> dict[str, Any]:
    """925-dim-aware counterpart to milestone_evaluator.evaluate_heldout --
    see run_episode_925's docstring for why the original cannot be reused
    directly against a SplitSteeringNavigationPolicy checkpoint. Reuses
    milestone_evaluator's own teacher-episode and aggregation logic
    unchanged (neither touches observation width)."""
    import numpy as np

    from .milestone_evaluator import _new_recovery_controller, _run_teacher_episode, _summarize_episodes

    net = model.policy
    per_layout: dict[str, Any] = {}
    for layout_name in manifest.layouts:
        teacher_results = [
            _run_teacher_episode(manifest.curriculum_path, layout_name, seed=seed, episode_seconds=episode_seconds, max_actions=max_actions, stage=manifest.stage)
            for seed in seeds
        ]
        teacher_median_kph = float(np.median([r["kills_per_simulated_hour"] for r in teacher_results])) if teacher_results else None
        policy_results = [
            run_episode_925(
                manifest.curriculum_path, layout_name, net=net, seed=seed, episode_seconds=episode_seconds,
                max_actions=max_actions, recovery=(_new_recovery_controller() if use_recovery else None), stage=manifest.stage,
            )
            for seed in seeds
        ]
        per_layout[layout_name] = _summarize_episodes(layout_name, policy_results, teacher_median_kph)
    return {"role": "heldout", "stage": manifest.stage, "assisted": use_recovery, "layouts": per_layout}


def evaluate_challenge_925(
    model: Any, manifest: Any, *, family_seeds: list[int], episode_seconds: float, max_actions: int, use_recovery: bool = False,
) -> dict[str, Any]:
    """925-dim-aware counterpart to milestone_evaluator.evaluate_challenge."""
    import numpy as np

    from .milestone_evaluator import _new_recovery_controller, _run_teacher_episode, _summarize_episodes

    net = model.policy
    fixed_results: dict[str, Any] = {}
    for scenario in manifest.fixed_regression_scenarios:
        teacher = _run_teacher_episode(
            scenario.curriculum_path, scenario.layout, seed=scenario.seed,
            episode_seconds=scenario.episode_seconds, max_actions=scenario.max_actions, stage=manifest.stage,
        )
        policy = run_episode_925(
            scenario.curriculum_path, scenario.layout, net=net, seed=scenario.seed,
            episode_seconds=scenario.episode_seconds, max_actions=scenario.max_actions,
            recovery=(_new_recovery_controller() if use_recovery else None), stage=manifest.stage,
        )
        policy["teacher_ratio"] = (
            policy["kills_per_simulated_hour"] / teacher["kills_per_simulated_hour"] if teacher["kills_per_simulated_hour"] else None
        )
        policy["expected_failure_signature"] = scenario.expected_failure_signature
        fixed_results[scenario.id] = {k: v for k, v in policy.items() if not k.startswith("_")}

    per_layout: dict[str, Any] = {}
    for layout_name in manifest.challenge_family_layouts:
        teacher_results = [
            _run_teacher_episode(manifest.challenge_family_curriculum_path, layout_name, seed=seed, episode_seconds=episode_seconds, max_actions=max_actions, stage=manifest.stage)
            for seed in family_seeds
        ]
        teacher_median_kph = float(np.median([r["kills_per_simulated_hour"] for r in teacher_results])) if teacher_results else None
        policy_results = [
            run_episode_925(
                manifest.challenge_family_curriculum_path, layout_name, net=net, seed=seed, episode_seconds=episode_seconds,
                max_actions=max_actions, recovery=(_new_recovery_controller() if use_recovery else None), stage=manifest.stage,
            )
            for seed in family_seeds
        ]
        per_layout[layout_name] = _summarize_episodes(layout_name, policy_results, teacher_median_kph)

    return {
        "role": "challenge", "stage": manifest.stage, "assisted": use_recovery,
        "fixed_regression_scenarios": fixed_results, "challenge_family": per_layout,
    }


_PARALLEL_925_EVAL_NET: Any = None


def _init_925_eval_worker(checkpoint_path: str) -> None:
    global _PARALLEL_925_EVAL_NET
    from stable_baselines3 import PPO
    _PARALLEL_925_EVAL_NET = PPO.load(checkpoint_path, device="cpu").policy


def _heldout_925_task(curriculum_path: str, layout_name: str, seed: int, episode_seconds: float, max_actions: int, use_recovery: bool, stage: str = "early") -> tuple[str, dict[str, Any], dict[str, Any]]:
    from .milestone_evaluator import _new_recovery_controller, _run_teacher_episode
    teacher = _run_teacher_episode(curriculum_path, layout_name, seed=seed, episode_seconds=episode_seconds, max_actions=max_actions, stage=stage)
    policy = run_episode_925(
        curriculum_path, layout_name, net=_PARALLEL_925_EVAL_NET, seed=seed, episode_seconds=episode_seconds,
        max_actions=max_actions, recovery=(_new_recovery_controller() if use_recovery else None), stage=stage,
    )
    return layout_name, teacher, policy


def evaluate_heldout_925_parallel(
    checkpoint_path: str | Path, manifest: Any, *, seeds: list[int], episode_seconds: float, max_actions: int,
    use_recovery: bool = False, n_workers: int = 4,
) -> dict[str, Any]:
    """Same report as evaluate_heldout_925, computed by farming (teacher,
    policy) episode pairs out across n_workers OS processes -- see
    milestone_evaluator.evaluate_heldout_parallel's docstring for the same
    compute-for-wallclock rationale, needed even more here given this
    project's real evaluation scale (episode_seconds=150, max_actions=1000)."""
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor

    from .milestone_evaluator import _summarize_episodes

    tasks = [
        (manifest.curriculum_path, layout_name, seed, episode_seconds, max_actions, use_recovery, manifest.stage)
        for layout_name in manifest.layouts for seed in seeds
    ]
    with ProcessPoolExecutor(
        max_workers=max(1, n_workers), initializer=_init_925_eval_worker, initargs=(str(checkpoint_path),),
    ) as pool:
        flat = list(pool.map(_heldout_925_task, *zip(*tasks))) if tasks else []

    teachers_by_layout: dict[str, list[dict[str, Any]]] = {name: [] for name in manifest.layouts}
    policies_by_layout: dict[str, list[dict[str, Any]]] = {name: [] for name in manifest.layouts}
    for layout_name, teacher, policy in flat:
        teachers_by_layout[layout_name].append(teacher)
        policies_by_layout[layout_name].append(policy)

    per_layout = {}
    for layout_name in manifest.layouts:
        teacher_results = teachers_by_layout[layout_name]
        teacher_median_kph = float(np.median([r["kills_per_simulated_hour"] for r in teacher_results])) if teacher_results else None
        per_layout[layout_name] = _summarize_episodes(layout_name, policies_by_layout[layout_name], teacher_median_kph)
    return {"role": "heldout", "stage": manifest.stage, "assisted": use_recovery, "layouts": per_layout}


def _challenge_925_fixed_task(
    curriculum_path: str, layout: str, seed: int, episode_seconds: float, max_actions: int,
    expected_failure_signature: str, use_recovery: bool, stage: str = "early",
) -> dict[str, Any]:
    from .milestone_evaluator import _new_recovery_controller, _run_teacher_episode
    teacher = _run_teacher_episode(curriculum_path, layout, seed=seed, episode_seconds=episode_seconds, max_actions=max_actions, stage=stage)
    policy = run_episode_925(
        curriculum_path, layout, net=_PARALLEL_925_EVAL_NET, seed=seed, episode_seconds=episode_seconds,
        max_actions=max_actions, recovery=(_new_recovery_controller() if use_recovery else None), stage=stage,
    )
    policy["teacher_ratio"] = policy["kills_per_simulated_hour"] / teacher["kills_per_simulated_hour"] if teacher["kills_per_simulated_hour"] else None
    policy["expected_failure_signature"] = expected_failure_signature
    return {k: v for k, v in policy.items() if not k.startswith("_")}


def evaluate_challenge_925_parallel(
    checkpoint_path: str | Path, manifest: Any, *, family_seeds: list[int], episode_seconds: float, max_actions: int,
    use_recovery: bool = False, n_workers: int = 4,
) -> dict[str, Any]:
    """Same report as evaluate_challenge_925, parallelized the same way as
    evaluate_heldout_925_parallel."""
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor

    from .milestone_evaluator import _summarize_episodes

    fixed_tasks = [
        (s.curriculum_path, s.layout, s.seed, s.episode_seconds, s.max_actions, s.expected_failure_signature, use_recovery, manifest.stage)
        for s in manifest.fixed_regression_scenarios
    ]
    family_tasks = [
        (manifest.challenge_family_curriculum_path, layout_name, seed, episode_seconds, max_actions, use_recovery, manifest.stage)
        for layout_name in manifest.challenge_family_layouts for seed in family_seeds
    ]

    fixed_results: dict[str, Any] = {}
    per_layout: dict[str, Any] = {}
    with ProcessPoolExecutor(
        max_workers=max(1, n_workers), initializer=_init_925_eval_worker, initargs=(str(checkpoint_path),),
    ) as pool:
        if fixed_tasks:
            fixed_flat = list(pool.map(_challenge_925_fixed_task, *zip(*fixed_tasks)))
            for scenario, result in zip(manifest.fixed_regression_scenarios, fixed_flat):
                fixed_results[scenario.id] = result
        if family_tasks:
            family_flat = list(pool.map(_heldout_925_task, *zip(*family_tasks)))
            teachers_by_layout: dict[str, list[dict[str, Any]]] = {name: [] for name in manifest.challenge_family_layouts}
            policies_by_layout: dict[str, list[dict[str, Any]]] = {name: [] for name in manifest.challenge_family_layouts}
            for layout_name, teacher, policy in family_flat:
                teachers_by_layout[layout_name].append(teacher)
                policies_by_layout[layout_name].append(policy)
            for layout_name in manifest.challenge_family_layouts:
                teacher_results = teachers_by_layout[layout_name]
                teacher_median_kph = float(np.median([r["kills_per_simulated_hour"] for r in teacher_results])) if teacher_results else None
                per_layout[layout_name] = _summarize_episodes(layout_name, policies_by_layout[layout_name], teacher_median_kph)

    return {
        "role": "challenge", "stage": manifest.stage, "assisted": use_recovery,
        "fixed_regression_scenarios": fixed_results, "challenge_family": per_layout,
    }


def graduate_basic_to_beginner(
    basic_checkpoint: str | Path,
    output: str | Path,
    *,
    curriculum: str | Path,
    timesteps: int,
    stage: str = "early",
    seed: int = 0,
    episode_seconds: float = 150.0,
    max_actions: int = 1000,
    device: str = "cpu",
    progress_every_seconds: float = 15.0,
    canonical_stage: str = "beginner",
) -> dict[str, Any]:
    """One bounded PPO chunk on the Basic-graduated checkpoint, recovery off
    throughout (structural, not configurable -- see module docstring),
    conservative project-standard hyperparameters (navigation_ppo.
    resume_ppo_chunk_phase2's own defaults, passed through unchanged).
    Mirrors resume_ppo_chunk_phase2's own shape: one bounded chunk, save,
    never loops on its own, no rehearsal folded in -- call
    rehearse_beginner_on_basic_data separately/periodically between chunks
    if forgetting is observed.

    Despite the function's name (kept for now -- Intermediate/Advanced reuse
    this same PPO-chunk mechanics unchanged, only the curriculum/checkpoint
    lineage differs), ``canonical_stage`` is the real pipeline stage this
    call represents ("beginner"/"intermediate"/"advanced") and is recorded
    as such in provenance -- NOT derived from ``stage`` (the curriculum's own
    internal stage string, e.g. "early" for Beginner, which is a different
    axis and historically got hardcoded to "beginner" in every provenance
    file regardless of caller, confirmed against real Intermediate/Advanced
    checkpoint provenance during the 2026-08-08 review)."""

    from .navigation_ppo import resume_ppo_chunk_phase2
    from .progress_reporting import SB3ProgressCallback
    from .run_provenance import build_run_manifest, write_run_manifest

    conservative_ppo_hparams = {
        "n_steps": 256, "batch_size": 128, "n_epochs": 4, "learning_rate": 5e-5,
        "clip_range": 0.10, "target_kl": 0.015, "gamma": 0.995, "gae_lambda": 0.95, "ent_coef": 0.015,
    }
    result = resume_ppo_chunk_phase2(
        checkpoint=basic_checkpoint, curriculum=curriculum, output=output, timesteps=timesteps,
        stage=stage, seed=seed, episode_seconds=episode_seconds, max_actions=max_actions, device=device,
        callback=SB3ProgressCallback(int(timesteps), label=f"{canonical_stage}_ppo", min_interval_seconds=progress_every_seconds),
    )
    manifest = build_run_manifest(
        stage=canonical_stage, milestone="ppo_chunk", seeds=seed,
        config={"timesteps": timesteps, "stage": stage, "episode_seconds": episode_seconds,
                "max_actions": max_actions, **conservative_ppo_hparams},
        curriculum_path=str(curriculum), recovery_config={"enabled": False, "reason": "structural -- see module docstring"},
        starting_checkpoint=str(Path(basic_checkpoint).resolve()), output_checkpoint=result["checkpoint_out"],
    )
    write_run_manifest(result["checkpoint_out"], manifest)
    return result


def rehearse_beginner_on_basic_data(
    checkpoint: str | Path,
    output: str | Path,
    *,
    basic_dataset_paths: list[str | Path],
    epochs: int = 2,
    learning_rate: float = 1e-5,
    batch_size: int = 128,
    seed: int = 0,
    canonical_stage: str = "beginner",
) -> dict[str, Any]:
    """Periodic BC rehearsal on Basic-stage data (human bootstrap +
    recovery-assisted DAgger, concatenated) to guard against Beginner's PPO
    phase forgetting Basic-stage event/EVA competence. Reuses
    basic_training.bootstrap_policy_from_human_recordings' training loop
    (session-stratified split, masked steering loss) unchanged -- it does
    not care whether "session" boundaries came from real recordings or
    simulator episodes, only that adjacent indices within one session are
    temporally correlated and should not straddle train/val.

    Deliberately event-only (bootstrap_policy_from_human_recordings'
    default train_heads): the combined dataset mixes human data (steering
    not well-correlated with the current representation, see that
    function's module docstring) with DAgger data (steering IS
    well-correlated) in one pool, with no per-sample source tag this
    function threads through -- excluding steering from rehearsal entirely
    is the conservative choice until that's examined properly. The user has
    flagged comparing pre/post-rehearsal navigation metrics before the
    first real Beginner continuation specifically because of this; not
    resolved here, intentionally deferred.
    """

    from stable_baselines3 import PPO

    from .basic_training import bootstrap_policy_from_human_recordings
    from .run_provenance import build_run_manifest, write_run_manifest

    combined_path = Path(output).with_suffix(".rehearsal_combined.npz")
    _concatenate_basic_datasets(basic_dataset_paths, combined_path)

    model = PPO.load(str(checkpoint), device="cpu")
    result = bootstrap_policy_from_human_recordings(
        model, combined_path, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, seed=seed,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))

    manifest = build_run_manifest(
        stage=canonical_stage, milestone="rehearsal", seeds=seed,
        config={"epochs": epochs, "learning_rate": learning_rate, "batch_size": batch_size,
                "basic_dataset_paths": [str(p) for p in basic_dataset_paths]},
        starting_checkpoint=str(Path(checkpoint).resolve()), output_checkpoint=str(output_path.resolve()),
    )
    write_run_manifest(output_path, manifest)
    return result


def rehearse_event_only_on_basic_data(
    checkpoint: str | Path,
    output: str | Path,
    *,
    basic_dataset_paths: list[str | Path],
    epochs: int = 2,
    learning_rate: float = 1e-5,
    batch_size: int = 128,
    seed: int = 0,
    canonical_stage: str = "beginner",
) -> dict[str, Any]:
    """Event-only counterpart of `rehearse_beginner_on_basic_data`, for a
    checkpoint produced by `build_event_only_ppo_from_basic_checkpoint`/
    `continue_event_only_ppo_chunk` (plain `ActorCriticPolicy`,
    `Discrete(len(FarmingEvent))`) -- that function's own training loop
    hard-requires a `SplitSteeringNavigationPolicy` (asserts `policy.
    mlp_extractor.steering_net` exists), so it cannot be reused unchanged
    here. Trains only the event action on the RAW_OBSERVATION_SIZE slice of
    the combined Basic-stage dataset (human bootstrap + recovery-assisted
    DAgger) -- event_net has never read the navigation sidecar (see
    `FrozenNavigationWrapper`'s docstring), and this policy has no steering
    head to train (steering labels in the combined dataset are inert here,
    matching the recovered architecture's ownership split: steering belongs
    to the frozen navigation checkpoint, never to a curriculum checkpoint)."""

    import torch
    import torch.nn.functional as F
    from stable_baselines3 import PPO

    from navigation.navigation_evidence import RAW_OBSERVATION_SIZE

    from .basic_training import _session_stratified_split
    from .factorized_v193_training import _sqrt_inverse_class_weights
    from .run_provenance import build_run_manifest, write_run_manifest

    combined_path = Path(output).with_suffix(".rehearsal_combined.npz")
    _concatenate_basic_datasets(basic_dataset_paths, combined_path)

    with np.load(combined_path, allow_pickle=False) as data:
        observations = np.asarray(data["observations"], dtype=np.float32)[:, :RAW_OBSERVATION_SIZE]
        actions = np.asarray(data["actions"], dtype=np.int64)
        session_index = np.asarray(data["session_index"], dtype=np.int64)

    train_idx, val_idx = _session_stratified_split(
        session_index, validation_fraction=0.15, seed=seed, event_labels=actions[:, 1],
    )
    if len(val_idx) == 0 or len(train_idx) == 0:
        raise ValueError("event rehearsal split produced an empty train or validation slice")

    from farming.actions import FarmingEvent

    model = PPO.load(str(checkpoint), device="cpu")
    policy = model.policy
    if int(getattr(policy.action_net, "out_features", -1)) != len(FarmingEvent):
        raise ValueError(
            f"policy.action_net has {getattr(policy.action_net, 'out_features', None)} outputs, expected "
            f"{len(FarmingEvent)} -- is model.policy a fresh event-only ActorCriticPolicy "
            "(build_event_only_ppo_from_basic_checkpoint's output), not a SplitSteeringNavigationPolicy?"
        )

    event_weights = torch.as_tensor(
        _sqrt_inverse_class_weights(actions[:, 1], train_idx, 3), dtype=torch.float32, device=policy.device,
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=float(learning_rate))
    policy.train()
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        order = rng.permutation(train_idx)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(order), int(batch_size)):
            batch_idx = order[start : start + int(batch_size)]
            obs = torch.as_tensor(observations[batch_idx], device=policy.device)
            labels = torch.as_tensor(actions[batch_idx, 1], device=policy.device, dtype=torch.long)
            distribution = policy.get_distribution(obs).distribution
            loss = F.cross_entropy(distribution.logits, labels, weight=event_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        history.append({"epoch": epoch, "mean_event_loss": epoch_loss / max(1, n_batches)})
    policy.eval()

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))

    from .navigation_subpolicy import event_only_architecture_contract

    manifest = build_run_manifest(
        stage=canonical_stage, milestone="rehearsal", seeds=seed,
        config={"epochs": epochs, "learning_rate": learning_rate, "batch_size": batch_size,
                "basic_dataset_paths": [str(p) for p in basic_dataset_paths], "action_contract": "event_only"},
        architecture_contract=event_only_architecture_contract(),
        starting_checkpoint=str(Path(checkpoint).resolve()), output_checkpoint=str(output_path.resolve()),
    )
    write_run_manifest(output_path, manifest)
    return {
        "train_samples": int(len(train_idx)), "validation_samples": int(len(val_idx)), "history": history,
    }


def _concatenate_basic_datasets(dataset_paths: list[str | Path], output_path: Path) -> Path:
    import numpy as np

    observations, actions, steering_valid, session_index = [], [], [], []
    session_offset = 0
    for path in dataset_paths:
        with np.load(Path(path), allow_pickle=False) as data:
            observations.append(np.asarray(data["observations"], dtype=np.float32))
            actions.append(np.asarray(data["actions"], dtype=np.int64))
            steering_valid.append(np.asarray(data["steering_label_valid"], dtype=np.bool_))
            sessions = np.asarray(data["session_index"], dtype=np.int64)
            session_index.append(sessions + session_offset)
            session_offset += int(sessions.max()) + 1 if len(sessions) else 0
    np.savez_compressed(
        output_path,
        observations=np.concatenate(observations, axis=0),
        actions=np.concatenate(actions, axis=0),
        steering_label_valid=np.concatenate(steering_valid, axis=0),
        session_index=np.concatenate(session_index, axis=0),
    )
    return output_path
