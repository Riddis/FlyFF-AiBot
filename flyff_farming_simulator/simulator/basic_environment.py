"""Basic-stage recovery-assisted rollout collection.

THE CENTRAL DESIGN DECISION, stated precisely: Basic NEVER runs PPO. There
is no rollout buffer, no advantage estimation, no stored log-prob, anywhere
in this module. Recovery-assisted trajectories are used ONLY to produce
supervised (observation, teacher-or-policy label) pairs for BC/DAgger
training (simulator.basic_training.bootstrap_policy_from_human_recordings
and this module's collect_basic_dagger_dataset), mined and trained the same
way simulator.navigation_dataset/factorized_v193_training already do.

Why not PPO-with-recovery: PPO's clipped surrogate objective is built on
ratio = exp(log pi_theta_new(a|s) - log pi_theta_old(a|s)), where a is the
SAME action whose execution produced the stored reward/next-state used to
compute its advantage. The rollout buffer stores (obs, action, reward, done,
value, log_prob) with log_prob = log pi_theta_old(action | obs) computed at
the moment the policy sampled that action. If recovery overrides the
environment's actual applied action after the policy has already sampled
and logged one, two things can happen, both broken:

  (a) The buffer keeps the POLICY's sampled action/log_prob, but the
      reward/next_observation are consequences of the RECOVERY action that
      actually executed. The stored tuple is no longer internally
      consistent -- credit for a transition the policy never caused gets
      assigned to an action it never took. This is not a subtle numerical
      issue: the policy-gradient/REINFORCE estimator E[grad log pi(a|s) *
      A(s,a)] is only valid when `a` is the action that actually produced
      the return being credited to it. Substituting the executed action
      while keeping the sampled one's bookkeeping breaks that correspondence
      outright.

  (b) The alternative -- store the RECOVERY action with a log_prob
      RECOMPUTED for it under the current policy (log pi_theta(recovery_
      action | obs), well-defined even though the policy never sampled it)
      -- is at least internally consistent, but changes what the update
      actually does: PPO would now reinforce the policy toward whatever
      recovery decided, using the environment's ordinary reward, for every
      intervened tick. That's no longer "the policy learning from its own
      exploration" -- it's imitation-via-RL-reward of the recovery
      controller, mixed into the same buffer as genuine on-policy
      experience, with no clean way to separate or weight the two. Sound in
      principle if that reward shaping happens to teach the right thing;
      much harder to reason about or debug than keeping the two learning
      modalities separate, and this codebase has no existing precedent or
      testing for it.

Recovery has never touched any PPO training path in this codebase (verified
directly: RecoveryController appears only in milestone_evaluator.py and
stagnation_diagnostics.py -- evaluation/diagnostic contexts -- never in
navigation_ppo.py, factorized_v193_cli.py, or factorized_cli.py's training
environment construction). This module keeps that property: Basic's
recovery-assisted rollouts feed BC/DAgger only. Beginner (simulator.
navigation_ppo.resume_ppo_chunk_phase2) runs PPO with recovery off entirely
-- by construction, since balanced_training_vec_env_phase2 never wraps with
RecoveryController -- so its on-policy assumption is never at risk.

Recovery's role during collection: prevent a single bad approach from
wasting an entire long episode (matching RecoveryController's existing
no_progress_ticks_required=15-tick trigger, ~3s at the project's 0.2s/tick
cadence -- not instant, so the policy still experiences the bad-approach ->
developing-contact window before anything intervenes) and save whatever
progress/context accumulated, not to be the supervised target itself: every
mined tick is labeled by the scripted obstacle-aware teacher (matching
simulator.navigation_dataset's existing precedent), never by recovery's own
chosen action, unless a specific failure mode is later found that the
teacher genuinely cannot label -- no such case is known now.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .local_navigation_features import derive_physical_clearance_features
from .navigation_dataset import (
    CATEGORY_PRECEDENCE,
    MiningConfig,
    _classify_tick,
    _group_into_events,
)
from .navigation_history import NavigationHistoryWrapper
from .progress_reporting import ProgressPrinter
from .recovery_controller import RecoveryController
from .scripted_policies import scripted_command
from .synthetic import iter_variant_environments


@dataclass
class _BasicTickRecord:
    tick: int
    observation_925: np.ndarray
    contact: bool
    min_clearance: float
    displacement: float
    policy_steering: int
    policy_event: int
    teacher_steering: int
    teacher_event: int
    recovering: bool


def _policy_forward(net: Any, observation_925: np.ndarray) -> tuple[int, int]:
    # Unlike navigation_dataset._policy_forward (which drives rollout with
    # the OLD 923-input 15k policy even inside a 925-wrapped env, purely to
    # RECORD 925-dim data for a later fine-tune of a different policy), this
    # module's `net` IS the 925-input SplitSteeringNavigationPolicy under
    # training -- it must see its own sidecar values, never sliced to 923.
    with torch.no_grad():
        obs_t = torch.as_tensor(observation_925[None, :], dtype=torch.float32, device=net.device)
        dist = net.get_distribution(obs_t).distribution
        s = dist[0].probs[0].cpu().numpy()
        e = dist[1].probs[0].cpu().numpy()
    return int(s.argmax()), int(e.argmax())


def _roll_basic_episode(
    curriculum_path: str, layout_name: str, *, seed: int, model: Any, episode_seconds: float, max_actions: int,
    history_window: int, expected_clear_path_displacement: float,
) -> tuple[list[_BasicTickRecord], dict[str, Any]]:
    """One recovery-assisted episode. Returns per-tick records (for DAgger
    mining) and a small intervention summary (for the Basic milestone
    evaluator) -- recovery's chosen action is applied to the environment
    when active, but every record's label still comes from the teacher, and
    nothing here is a PPO transition."""

    net = model.policy
    entry, base_env = next(iter(iter_variant_environments(
        curriculum_path, stage="early", seed=seed, episode_steps=max_actions,
        episode_seconds=episode_seconds, variant_name=layout_name,
    )))
    env = NavigationHistoryWrapper(
        base_env, window=history_window, expected_clear_path_displacement=expected_clear_path_displacement,
    )
    observation, _info = env.reset(seed=seed)
    recovery = RecoveryController()

    records: list[_BasicTickRecord] = []
    info: dict[str, Any] = {}
    previous_distance = 0.0
    previous_contacts = 0
    intervention_ticks = 0
    for tick in range(int(max_actions)):
        raw = np.asarray(observation, dtype=np.float32)[:923]
        teacher_command = scripted_command("obstacle_aware", base_env)
        # policy_steering/policy_event are the policy's OWN proposed action
        # for the CURRENT observation -- this, not whatever recovery decides
        # to actually execute, is what gets DAgger-labeled below. Recovery
        # only changes what physically happens next (which affects the NEXT
        # observation), never what this tick's supervised example is.
        policy_steering, policy_event = _policy_forward(net, np.asarray(observation, dtype=np.float32))
        clearance = derive_physical_clearance_features(raw)

        # Exact call contract/order as milestone_evaluator.run_episode:
        # recovery.step() is queried BEFORE env.step(), using evidence from
        # the PREVIOUS env.step() (captured in `info`, empty on tick 0), to
        # decide whether to override this tick's applied action.
        applied_steering, applied_event = recovery.step(
            tick=tick, player_x=base_env.player_x, player_z=base_env.player_z, heading=base_env.heading,
            displacement_this_tick=info.get("total_distance_cells", 0.0) - previous_distance if info else 0.0,
            contact_this_tick=bool(info.get("contacts", 0) - previous_contacts) if info else False,
            map_model=base_env.map, policy_steering=policy_steering, policy_event=policy_event,
        )
        recovering_now = recovery.state.value == "recovering"
        if recovering_now:
            intervention_ticks += 1
        previous_distance = float(info.get("total_distance_cells", 0.0)) if info else 0.0
        previous_contacts = int(info.get("contacts", 0)) if info else 0

        next_observation, _r, term, trunc, info = env.step(
            np.asarray([applied_steering, applied_event], dtype=np.int64)
        )

        total_distance = float(info.get("total_distance_cells", previous_distance))
        total_contacts = int(info.get("contacts", previous_contacts))
        displacement_this_tick = total_distance - previous_distance
        contact_this_tick = total_contacts > previous_contacts

        records.append(_BasicTickRecord(
            tick=tick, observation_925=np.asarray(observation, dtype=np.float32),
            contact=contact_this_tick, min_clearance=float(np.min(clearance)),
            displacement=displacement_this_tick, policy_steering=policy_steering, policy_event=policy_event,
            teacher_steering=int(teacher_command.steering), teacher_event=int(teacher_command.event),
            recovering=recovering_now,
        ))

        observation = next_observation
        if term or trunc:
            break

    env.close()
    summary = {
        "layout": layout_name, "seed": int(seed), "steps": len(records),
        "intervention_count": len(recovery.interventions),
        "intervention_ticks": intervention_ticks,
        "final_state": recovery.state.value,
    }
    return records, summary


_DAGGER_PARALLEL_WORKER_MODEL: Any = None


def _init_dagger_roll_worker(checkpoint_path: str) -> None:
    global _DAGGER_PARALLEL_WORKER_MODEL
    from stable_baselines3 import PPO
    _DAGGER_PARALLEL_WORKER_MODEL = PPO.load(checkpoint_path, device="cpu")


def _dagger_roll_worker_task(
    curriculum_path: str, layout_name: str, seed: int, episode_seconds: float, max_actions: int,
    history_window: int, expected_clear_path_displacement: float,
) -> tuple[str, int, list, dict[str, Any]]:
    records, summary = _roll_basic_episode(
        curriculum_path, layout_name, seed=seed, model=_DAGGER_PARALLEL_WORKER_MODEL,
        episode_seconds=episode_seconds, max_actions=max_actions,
        history_window=history_window, expected_clear_path_displacement=expected_clear_path_displacement,
    )
    return layout_name, seed, records, summary


def collect_basic_dagger_dataset(
    curriculum_path: str,
    layout_names: list[str],
    *,
    seeds: list[int],
    model: Any,
    episode_seconds: float,
    max_actions: int,
    config: MiningConfig = MiningConfig(),
    progress_every_seconds: float = 15.0,
    checkpoint_path: str | None = None,
    n_workers: int = 1,
) -> dict[str, Any]:
    """Recovery-assisted counterpart to
    simulator.navigation_dataset.mine_navigation_dataset: same 4-category
    classification/event-grouping/sampling-cap logic (imported, not
    reimplemented, so both stay consistent), same teacher-labeling default,
    but rolled with recovery enabled so a bad approach cannot waste an
    entire long episode. Every mined observation/label pair is still a plain
    supervised example -- see this module's docstring for why recovery never
    enters a PPO buffer. `layout_names`/`curriculum_path` must be
    TRAINING-side, matching mine_navigation_dataset's own requirement.

    If `checkpoint_path` is given and `n_workers` > 1, every (layout, seed)
    episode is rolled up front across `n_workers` OS processes (each
    loading its own copy of the checkpoint) instead of one sequential loop,
    and the mining/capping logic below then runs unchanged over the
    results. This deliberately rolls some episodes that the sequential
    per-layout event cap (`max_events_per_layout_seed`) would otherwise
    have skipped -- wasted compute traded for wall-clock time, per the
    project's "prefer wasting compute over wallclock time" preference --
    but produces byte-for-byte the same mined dataset as the sequential
    path, since the capping/sampling decisions themselves are untouched and
    still applied in the same layout/seed order over `model`'s (or the
    loaded checkpoint's, which must match `model`) policy outputs."""

    precomputed_episodes: dict[tuple[str, int], tuple[list, dict[str, Any]]] = {}
    if checkpoint_path is not None and n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        tasks = [
            (curriculum_path, layout_name, seed, episode_seconds, max_actions,
             config.history_window, config.expected_clear_path_displacement)
            for layout_name in layout_names for seed in seeds
        ]
        with ProcessPoolExecutor(
            max_workers=n_workers, initializer=_init_dagger_roll_worker, initargs=(checkpoint_path,),
        ) as pool:
            for layout_name, seed, records, summary in pool.map(_dagger_roll_worker_task, *zip(*tasks)):
                precomputed_episodes[(layout_name, seed)] = (records, summary)

    observations: list[np.ndarray] = []
    actions: list[tuple[int, int]] = []
    categories: list[str] = []
    layout_ids: list[int] = []
    episode_ids: list[int] = []
    teacher_agrees_flags: list[bool] = []
    category_counts: dict[str, int] = {c: 0 for c in CATEGORY_PRECEDENCE}
    episode_summaries: list[dict[str, Any]] = []
    episode_counter = 0

    total_episodes = len(layout_names) * len(seeds)
    progress = ProgressPrinter(total_episodes, label="basic_dagger_collection", min_interval_seconds=progress_every_seconds)
    episodes_done = 0

    for layout_id, layout_name in enumerate(layout_names):
        layout_event_count = 0
        for seed in seeds:
            episodes_done += 1
            if layout_event_count >= config.max_events_per_layout_seed:
                progress.update(episodes_done, extra=f"layout={layout_name} (event cap reached, skipped)")
                continue
            if (layout_name, seed) in precomputed_episodes:
                records, summary = precomputed_episodes[(layout_name, seed)]
            else:
                records, summary = _roll_basic_episode(
                    curriculum_path, layout_name, seed=seed, model=model,
                    episode_seconds=episode_seconds, max_actions=max_actions,
                    history_window=config.history_window,
                    expected_clear_path_displacement=config.expected_clear_path_displacement,
                )
            episode_summaries.append({**summary, "layout_index": layout_id, "episode_index": episode_counter})
            progress.update(
                episodes_done,
                extra=f"layout={layout_name} interventions={summary['intervention_count']} steps={summary['steps']}",
            )

            tick_categories = [_classify_tick(records, i, config) for i in range(len(records))]
            events = _group_into_events(tick_categories)

            episode_event_count = 0
            for category, start, end in events:
                if episode_event_count >= config.max_events_per_episode:
                    break
                if layout_event_count >= config.max_events_per_layout_seed:
                    break
                span = list(range(start, end + 1))
                sample_indices = span[-config.max_samples_per_event :]
                for i in sample_indices:
                    record = records[i]
                    if category in ("persistent_wedge", "collision_onset", "ordinary"):
                        label = (record.teacher_steering, record.teacher_event)
                        agrees = (record.teacher_steering, record.teacher_event) == (
                            record.policy_steering, record.policy_event,
                        )
                    else:  # safe_proximity
                        label = (record.policy_steering, record.policy_event)
                        agrees = (record.teacher_steering, record.teacher_event) == (
                            record.policy_steering, record.policy_event,
                        )
                    observations.append(record.observation_925)
                    actions.append(label)
                    categories.append(category)
                    layout_ids.append(layout_id)
                    episode_ids.append(episode_counter)
                    teacher_agrees_flags.append(agrees)
                    category_counts[category] += 1
                episode_event_count += 1
                layout_event_count += 1
            episode_counter += 1
    progress.finish()

    return {
        "observations": np.asarray(observations, dtype=np.float32) if observations else np.zeros((0, 925), dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64) if actions else np.zeros((0, 2), dtype=np.int64),
        "categories": categories,
        "layout_index": np.asarray(layout_ids, dtype=np.int64),
        "episode_index": np.asarray(episode_ids, dtype=np.int64),
        "teacher_agrees": np.asarray(teacher_agrees_flags, dtype=bool),
        "category_counts": category_counts,
        "layout_names": list(layout_names),
        "episode_summaries": episode_summaries,
    }


def save_basic_dagger_dataset(mined: dict[str, Any], output_path: str) -> str:
    """Persist collect_basic_dagger_dataset's return value in the same
    schema basic_training.bootstrap_policy_from_human_recordings /
    beginner_transition._concatenate_basic_datasets expect (steering_label_
    valid, session_index), so Basic-stage human and simulator data can be
    trained on and rehearsed with the same function. Every mined sample has
    a fully-determined steering label (unlike human click-to-move sessions),
    so steering_label_valid is all-True; episode_index doubles as
    session_index (same "do not split temporally-adjacent samples across
    train/val" semantic)."""

    from pathlib import Path

    observations = mined["observations"]
    actions = mined["actions"]
    n = observations.shape[0]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        observations=observations,
        actions=actions,
        steering_label_valid=np.ones((n,), dtype=np.bool_),
        session_index=mined["episode_index"],
    )
    return str(output)
