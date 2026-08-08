"""Basic-stage entrypoint: fresh-initialization current-architecture policy,
two-source bootstrap, and the canonical checkpoint naming scheme.

STEERING is bootstrapped from the scripted simulator teacher, not human
recordings. EVENT/EVA is bootstrapped from human recordings. This mirrors a
finding already established earlier in this project: when the steering
branch was restricted to the compact target-geometry representation, human
steering-label accuracy fell to ~31.6% while scripted-teacher steering,
target-angle correlations, and actual simulated farming behavior all stayed
healthy. Re-confirmed directly on the real canonical human dataset here (all
6 original geometry features correlate <=0.05 with the recorded human
steering key across 2684 valid samples) -- the compact steering
representation was deliberately narrowed to kill an earlier shortcut-
learning bug, and it does not encode enough of a human's situational
awareness to explain their exact recorded key presses on a frame-by-frame
basis, even though it explains scripted-teacher and simulated-policy
steering well. This is an accepted property of the representation, not a
recording or simulator bug -- human steering BC is not used to initialize
the fresh steering branch.

Basic never runs PPO (see simulator/basic_environment.py's module docstring
for the full recovery/PPO design rationale). Everything here is supervised
learning: cross-entropy on (observation, label) pairs, with no rollout
buffer, no advantage estimation, no log-prob bookkeeping -- so there is no
on-policy consistency requirement to violate regardless of what recovery
does to the executed trajectory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from farming.actions import FarmingEvent, SteeringAction

from .factorized_training import (
    _event_sampling_fractions,
    _train_balanced_factorized_heads,
    _training_values,
    factorized_stage_gate,
)
from .factorized_v193_training import (
    _apply_prior_bias_correction,
    _class_priors,
    _human_session_stratified_split,
    _layout_stratified_episode_split,
    _natural_human_event_target,
    _prediction_diagnostics,
    _sqrt_inverse_class_weights,
    _train_natural_prior_epoch,
)
from .navigation_history import (
    CALIBRATED_EXPECTED_CLEAR_PATH_DISPLACEMENT,
    CALIBRATED_HISTORY_WINDOW,
    POLICY_INPUT_SIZE,
    RAW_OBSERVATION_SIZE,
    NavigationStepEvidence,
    sidecar_values_from_history,
)
from .progress_reporting import ProgressPrinter
from .split_branch_policy import STEERING_NAVIGATION_FEATURE_SIZE

CANONICAL_STAGE_PREFIX = "canonical"


def canonical_checkpoint_name(stage: str, milestone: str) -> str:
    """Name checkpoints/configs by where they belong in the curriculum, not
    by the sequence of experiments that produced them. E.g.
    canonical_checkpoint_name("basic", "bootstrap") ->
    "canonical_basic_bootstrap"; canonical_checkpoint_name("beginner",
    "ppo_010k") -> "canonical_beginner_ppo_010k"."""

    stage_key = stage.strip().lower()
    from .curriculum_stages import CANONICAL_STAGES

    if stage_key not in CANONICAL_STAGES:
        raise KeyError(f"Unknown canonical stage {stage!r}; expected one of {sorted(CANONICAL_STAGES)}")
    milestone_key = milestone.strip().lower().replace(" ", "_")
    return f"{CANONICAL_STAGE_PREFIX}_{stage_key}_{milestone_key}"


def save_checkpoint_with_provenance(
    model: Any,
    checkpoint_path: str | Path,
    *,
    stage: str,
    milestone: str,
    seeds: list[int] | int,
    config: dict[str, Any],
    curriculum_path: str | None = None,
    heldout_manifest_path: str | None = None,
    recording_paths: list[str] | None = None,
    recovery_config: dict[str, Any] | None = None,
    dagger_config: dict[str, Any] | None = None,
    starting_checkpoint: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save a checkpoint AND its provenance manifest together, so a
    checkpoint on disk is never missing the record of what produced it.
    Prefer this over a bare model.save() for every canonical Basic/Beginner
    checkpoint. `starting_checkpoint=None` (the default) is the explicit
    record that this run began from a fresh current-architecture
    initialization, never a historical checkpoint -- pass the real path for
    any run that legitimately continues from one (e.g. Beginner from a
    Basic checkpoint)."""

    from .run_provenance import build_run_manifest, write_run_manifest

    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(checkpoint))
    manifest = build_run_manifest(
        stage=stage, milestone=milestone, seeds=seeds, config=config,
        curriculum_path=curriculum_path, heldout_manifest_path=heldout_manifest_path,
        recording_paths=recording_paths, recovery_config=recovery_config, dagger_config=dagger_config,
        starting_checkpoint=starting_checkpoint, output_checkpoint=str(checkpoint.resolve()), extra=extra,
    )
    write_run_manifest(checkpoint, manifest)
    return checkpoint


def reconstruct_session_sidecars(
    displacement_cells: np.ndarray,
    events: np.ndarray,
    *,
    contact: np.ndarray | None = None,
    history_window: int = CALIBRATED_HISTORY_WINDOW,
    expected_clear_path_displacement: float = CALIBRATED_EXPECTED_CLEAR_PATH_DISPLACEMENT,
) -> np.ndarray:
    """Reconstruct [recent_progress, recent_contact] for one session's
    samples, already in strict temporal order, causally aligned with
    NavigationHistoryWrapper's actual online semantics -- see
    tests/test_temporal_sidecar_parity.py for the parity proof this is
    checked against directly (drive the real wrapper against a controlled
    fake env, compare index by index).

    The alignment is easy to get wrong in two independent ways, both fixed
    here (an earlier version of this function had both bugs):

    1. ORDER of append vs compute. Online, NavigationHistoryWrapper.step()
       appends the tick's evidence to history BEFORE computing the sidecar
       for the resulting observation -- so observation_k's sidecar (k>=1)
       already reflects the very transition that just produced it (that
       transition already happened by the time the policy sees
       observation_k; it is history, not future). Computing the sidecar
       BEFORE appending the current sample's own evidence -- as an earlier,
       buggy version of this function did -- makes every reconstructed
       sidecar one tick STALER than its online counterpart: not a future
       leak, but a real train/rollout mismatch (the policy would be trained
       to expect systematically less-current information than it actually
       receives live).

    2. WHICH transition an evidence entry describes. Sample i's
       displacement_cells[i] is the (i-1)->i transition (real inter-frame
       displacement, see demonstrations.py). But `eva_attempted` for that
       SAME (i-1)->i transition must come from the action recorded AT
       sample i-1 (the action that was being taken FROM state i-1, which is
       what drove the i-1->i transition) -- NOT from sample i's own action
       (which drives the NEXT transition, i->(i+1)). Pairing
       displacement_cells[i] with events[i] instead of events[i-1] -- as an
       earlier, buggy version of this function did -- silently merges
       evidence from two different transitions into one NavigationStepEvidence.

    Sample 0 (first in session) always gets a zero sidecar from empty
    history, exactly matching what NavigationHistoryWrapper.reset() does --
    there is no completed transition yet to summarize.

    `contact`, if given, is ground-truth per-sample contact evidence
    (contact[idx] = whether the (idx-1)->idx transition made contact),
    aligned the same causal way as displacement. Human recordings have no
    such ground truth (see demonstrations.export_demonstrations' docstring),
    so build_human_bootstrap_dataset never passes it -- the default (None,
    treated as all-False) is what produces that documented neutral
    recent_contact placeholder. This parameter exists so the alignment
    algorithm itself can be verified against known contact ground truth
    (tests/test_temporal_sidecar_parity.py), independent of that
    human-recording-specific limitation.
    """

    n = displacement_cells.shape[0]
    sidecars = np.zeros((n, 2), dtype=np.float32)
    history: list[NavigationStepEvidence] = []
    for idx in range(n):
        if idx > 0:
            history.append(
                NavigationStepEvidence(
                    displacement_cells=float(displacement_cells[idx]),
                    contact=False if contact is None else bool(contact[idx]),
                    eva_attempted=bool(events[idx - 1] == int(FarmingEvent.CAST_EVA)),
                )
            )
        sidecars[idx] = sidecar_values_from_history(
            history[-history_window:], expected_clear_path_displacement=expected_clear_path_displacement,
        )
    return sidecars


def build_human_bootstrap_dataset(
    demonstration_dataset_path: str | Path,
    output_path: str | Path,
    *,
    history_window: int = CALIBRATED_HISTORY_WINDOW,
    expected_clear_path_displacement: float = CALIBRATED_EXPECTED_CLEAR_PATH_DISPLACEMENT,
) -> Path:
    """Convert a simulator.demonstrations.export_demonstrations output
    (923-value observations, session_index/elapsed_ms/displacement_cells in
    per-session sequential order) into a 925-value (raw + sidecar) Basic
    bootstrap dataset.

    recent_progress is reconstructed faithfully from real inter-frame
    displacement (same windowing as NavigationHistoryWrapper, via the same
    shared sidecar_values_from_history function). recent_contact CANNOT be
    faithfully reconstructed -- recordings carry no collision ground truth
    (see demonstrations.export_demonstrations' docstring) -- so it is fixed
    at a documented neutral 0.0 for every sample here, not inferred from
    displacement shortfall (that would be invented data). This is an
    explicit, reported design choice, not an oversight: the steering branch
    sees a real, informative recent_progress signal and a constant,
    uninformative recent_contact channel during human-data bootstrap: the
    first real recent_contact signal a Basic-stage policy sees comes from
    simulator recovery-assisted rollouts (simulator.basic_environment),
    which have exact, ground-truth collision detection.
    """

    with np.load(Path(demonstration_dataset_path), allow_pickle=False) as data:
        observations = np.asarray(data["observations"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.int64)
        steering_label_valid = np.asarray(data["steering_label_valid"], dtype=np.bool_)
        event_label_valid = np.asarray(data["event_label_valid"], dtype=np.bool_)
        session_index = np.asarray(data["session_index"], dtype=np.int64)
        elapsed_ms = np.asarray(data["elapsed_ms"], dtype=np.int64)
        displacement_cells = np.asarray(data["displacement_cells"], dtype=np.float32)
        # Per-SESSION (not per-sample) role, e.g. "direct_keyboard" vs
        # "eva_only" -- propagated through so bootstrap_policy_from_human_
        # recordings can compute the event class prior from continuous
        # direct-keyboard sessions only, not skewed by eva_only's curated
        # near-100%-CAST_EVA clips (see that function's docstring).
        source_recording_role = (
            np.asarray(data["source_recording_role"], dtype=str) if "source_recording_role" in data else None
        )

    if observations.shape[1] != RAW_OBSERVATION_SIZE:
        raise ValueError(
            f"demonstration dataset observations must be {RAW_OBSERVATION_SIZE}-valued, "
            f"got {observations.shape[1]}"
        )

    n = observations.shape[0]
    sidecars = np.zeros((n, POLICY_INPUT_SIZE - RAW_OBSERVATION_SIZE), dtype=np.float32)

    for session in np.unique(session_index):
        order = np.flatnonzero(session_index == session)
        order = order[np.argsort(elapsed_ms[order], kind="stable")]
        sample_sidecars = reconstruct_session_sidecars(
            displacement_cells[order], actions[order, 1],
            history_window=history_window, expected_clear_path_displacement=expected_clear_path_displacement,
        )
        sidecars[order] = sample_sidecars

    policy_input = np.concatenate([observations, sidecars], axis=1)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = dict(
        observations=policy_input.astype(np.float32),
        actions=actions,
        steering_label_valid=steering_label_valid,
        event_label_valid=event_label_valid,
        session_index=session_index,
        recent_contact_is_neutral_placeholder=np.asarray([True]),
    )
    if source_recording_role is not None:
        save_kwargs["source_recording_role"] = source_recording_role
    np.savez_compressed(output, **save_kwargs)
    return output


def build_fresh_basic_policy(
    env: Any,
    *,
    steering_net_arch: list[int] | None = None,
    event_net_arch: list[int] | None = None,
    vf_net_arch: list[int] | None = None,
    seed: int = 0,
    device: str = "cpu",
) -> Any:
    """A genuinely fresh-initialization SplitSteeringNavigationPolicy: no
    zero-init transplant, no loaded weights of any kind -- the historical
    15k checkpoint and all of its descendants are benchmarks only, never a
    parent for this lineage. `env` must already be
    NavigationHistoryWrapper-wrapped (925-value observation space)."""

    from stable_baselines3 import PPO

    from .split_branch_policy import SplitSteeringNavigationPolicy

    return PPO(
        SplitSteeringNavigationPolicy,
        env,
        policy_kwargs={
            "steering_net_arch": steering_net_arch or [64, 32],
            "event_net_arch": event_net_arch or [64, 32],
            "vf_net_arch": vf_net_arch or [64, 32],
        },
        seed=int(seed),
        device=device,
        # Conservative project-standard hyperparameters (see
        # navigation_ppo.resume_ppo_chunk_phase2's docstring for why these
        # are always passed explicitly). Irrelevant to Basic's BC-only
        # bootstrap, but set correctly from construction so the same
        # checkpoint is immediately safe to hand to Beginner's PPO phase
        # without a separate hyperparameter-repair step.
        n_steps=256,
        batch_size=128,
        n_epochs=4,
        learning_rate=5e-5,
        clip_range=0.10,
        target_kl=0.015,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.015,
    )


def collect_simulator_teacher_dataset(
    curriculum_path: str,
    layout_names: list[str],
    *,
    samples: int,
    episode_seconds: float,
    max_actions: int,
    seed: int = 0,
    teacher_policy: str = "obstacle_aware",
    history_window: int = CALIBRATED_HISTORY_WINDOW,
    expected_clear_path_displacement: float = CALIBRATED_EXPECTED_CLEAR_PATH_DISPLACEMENT,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Roll the scripted simulator teacher through `layout_names`,
    collecting (925-value observation, teacher action) pairs -- the source
    of steering supervision for the fresh Basic policy (see module
    docstring for why, not human recordings). 925-value-aware sibling of
    factorized_v193_training.collect_teacher_dataset_v193 (which is fixed
    at 923 values, no NavigationHistoryWrapper).

    Rolling the TEACHER's own actions (not the untrained fresh policy's)
    gives a clean, well-distributed demonstration set for the very first
    bootstrap -- an untrained policy's own rollout would mostly wander
    degenerate/uninformative states. Basic's later DAgger rounds are what
    expand coverage to policy-visited states.
    """

    from .navigation_history import NavigationHistoryWrapper
    from .scripted_policies import scripted_command
    from .synthetic import iter_variant_environments

    environments = list(iter_variant_environments(
        curriculum_path, stage="early", seed=seed, episode_steps=max_actions, episode_seconds=episode_seconds,
    ))
    if layout_names:
        environments = [(entry, env) for entry, env in environments if entry.name in layout_names]
    if not environments:
        raise ValueError("No synthetic variants matched the requested teacher-collection layouts")

    observations: list[np.ndarray] = []
    actions: list[tuple[int, int]] = []
    episode_ids: list[int] = []
    layout_ids: list[int] = []
    episode_id = 0
    progress = ProgressPrinter(int(samples), label="teacher_dataset_collection", min_interval_seconds=15.0)
    try:
        while len(observations) < int(samples):
            for layout_id, (_entry, base_env) in enumerate(environments):
                env = NavigationHistoryWrapper(
                    base_env, window=history_window, expected_clear_path_displacement=expected_clear_path_displacement,
                )
                observation, _ = env.reset(seed=seed + episode_id * 1009 + layout_id * 37)
                for _ in range(int(max_actions)):
                    command = scripted_command(teacher_policy, base_env)
                    observations.append(np.asarray(observation, dtype=np.float32).copy())
                    actions.append(command.as_array())
                    episode_ids.append(episode_id)
                    layout_ids.append(layout_id)
                    observation, _, terminated, truncated, _ = env.step(np.asarray(command.as_array(), dtype=np.int64))
                    progress.update(min(len(observations), int(samples)))
                    if len(observations) >= int(samples) or terminated or truncated:
                        break
                episode_id += 1
                if len(observations) >= int(samples):
                    break
    finally:
        for _, env in environments:
            env.close()
    progress.finish()

    obs = np.asarray(observations[: int(samples)], dtype=np.float32)
    labels = np.asarray(actions[: int(samples)], dtype=np.int64)
    episode_index = np.asarray(episode_ids[: int(samples)], dtype=np.int64)
    layout_index = np.asarray(layout_ids[: int(samples)], dtype=np.int64)
    if obs.shape != (int(samples), POLICY_INPUT_SIZE) or labels.shape != (int(samples), 2):
        raise ValueError(f"Unexpected teacher arrays: observations={obs.shape}, actions={labels.shape}")

    train_indices, validation_indices, validation_episodes = _layout_stratified_episode_split(
        episode_index, layout_index, labels, validation_fraction=0.20, seed=seed,
    )
    layout_names_used = np.asarray([entry.name for entry, _env in environments], dtype=str)

    result = {
        "observations": obs, "actions": labels, "episode_index": episode_index, "layout_index": layout_index,
        "layout_names": layout_names_used, "train_indices": train_indices, "validation_indices": validation_indices,
        "validation_episodes": validation_episodes,
        "steering_counts": np.bincount(labels[:, 0], minlength=3).tolist(),
        "event_counts": np.bincount(labels[:, 1], minlength=3).tolist(),
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, observations=obs, actions=labels, episode_index=episode_index, layout_index=layout_index,
            layout_names=layout_names_used, train_indices=train_indices, validation_indices=validation_indices,
        )
        result["path"] = str(path.resolve())
    return result


def bootstrap_steering_from_teacher(
    model: Any,
    teacher_dataset: dict[str, Any] | str | Path,
    *,
    epochs: int = 20,
    learning_rate: float = 3.0e-4,
    batch_size: int = 128,
    seed: int = 0,
    progress_every_seconds: float = 15.0,
) -> dict[str, Any]:
    """Steering-only BC from the scripted-teacher dataset (freezes
    event_net/event_out/value_net, same freeze pattern as
    factorized_v193_training.fine_tune_steering_branch_v193 -- here applied
    to a fresh, not already-competent, steering branch). Class-balanced loss
    (this codebase's established fix for exactly the majority-class-collapse
    failure mode -- see factorized_v193_training._sqrt_inverse_class_weights).

    `teacher_dataset` may be the dict collect_simulator_teacher_dataset
    returns directly, or a path to what it saved.
    """

    import torch
    import torch.nn.functional as F

    if isinstance(teacher_dataset, (str, Path)):
        with np.load(Path(teacher_dataset), allow_pickle=False) as data:
            observations = np.asarray(data["observations"], dtype=np.float32)
            labels = np.asarray(data["actions"], dtype=np.int64)
            train_idx = np.asarray(data["train_indices"], dtype=np.int64)
            val_idx = np.asarray(data["validation_indices"], dtype=np.int64)
    else:
        observations = teacher_dataset["observations"]
        labels = teacher_dataset["actions"]
        train_idx = teacher_dataset["train_indices"]
        val_idx = teacher_dataset["validation_indices"]

    if observations.shape[1] != POLICY_INPUT_SIZE:
        raise ValueError(f"teacher dataset observations must be {POLICY_INPUT_SIZE}-valued, got {observations.shape[1]}")

    policy = model.policy
    steering_first_layer = getattr(getattr(policy, "mlp_extractor", None), "steering_net", [None])[0]
    if getattr(steering_first_layer, "in_features", None) != STEERING_NAVIGATION_FEATURE_SIZE:
        raise ValueError("policy.mlp_extractor.steering_net does not match STEERING_NAVIGATION_FEATURE_SIZE; is model.policy a SplitSteeringNavigationPolicy?")

    steering_weights = torch.as_tensor(
        _sqrt_inverse_class_weights(labels[:, 0], train_idx, 3), dtype=torch.float32, device=policy.device,
    )

    trainable_names = ("mlp_extractor.steering_net", "action_net.steering_out")
    trainable_params = []
    for name, param in policy.named_parameters():
        is_trainable = any(key in name for key in trainable_names)
        param.requires_grad = is_trainable
        if is_trainable:
            trainable_params.append(param)
    assert trainable_params, "unreachable: steering_input check above already validated the architecture"

    policy.eval()
    before = _prediction_diagnostics(policy, observations[val_idx], labels[val_idx], batch_size=batch_size)

    optimizer = torch.optim.Adam(trainable_params, lr=float(learning_rate))
    policy.train()
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    total_batches = int(epochs) * max(1, (len(train_idx) + batch_size - 1) // batch_size)
    progress = ProgressPrinter(total_batches, label="teacher_steering_bootstrap", min_interval_seconds=progress_every_seconds)
    batches_done = 0
    for epoch in range(1, int(epochs) + 1):
        order = rng.permutation(train_idx)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(order), int(batch_size)):
            batch_idx = order[start : start + int(batch_size)]
            obs = torch.as_tensor(observations[batch_idx], device=policy.device)
            steering_labels = torch.as_tensor(labels[batch_idx, 0], device=policy.device, dtype=torch.long)
            distribution = policy.get_distribution(obs).distribution
            loss = F.cross_entropy(distribution[0].logits, steering_labels, weight=steering_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
            batches_done += 1
            progress.update(batches_done, extra=f"epoch={epoch}/{epochs}")
        history.append({"epoch": epoch, "mean_steering_loss": epoch_loss / max(1, n_batches)})
    progress.finish()

    policy.eval()
    after = _prediction_diagnostics(policy, observations[val_idx], labels[val_idx], batch_size=batch_size)
    for _name, param in policy.named_parameters():
        param.requires_grad = True

    # Explicit target-angle correlation sign check (part of the required
    # pre-launch smoke test): recompute the 6 geometry features on the
    # validation observations and confirm sin(relative_angle) correlates
    # POSITIVELY with LEFT and NEGATIVELY with RIGHT -- the same check that
    # was healthy historically (+0.72..0.76 / -0.73..-0.76) and is exactly
    # what the human-data audit found broken for human labels.
    from .split_branch_policy import derive_geometry_features_torch

    with torch.no_grad():
        geom = derive_geometry_features_torch(torch.as_tensor(observations[val_idx, :923], dtype=torch.float32)).numpy()
    val_steering = labels[val_idx, 0]
    sin_angle = geom[:, 0]
    left_mask = val_steering == int(SteeringAction.LEFT)
    right_mask = val_steering == int(SteeringAction.RIGHT)
    angle_correlation = {
        "corr_sin_angle_vs_is_left": float(np.corrcoef(sin_angle, left_mask.astype(float))[0, 1]) if left_mask.any() else None,
        "corr_sin_angle_vs_is_right": float(np.corrcoef(sin_angle, right_mask.astype(float))[0, 1]) if right_mask.any() else None,
    }

    return {
        "train_samples": int(len(train_idx)), "validation_samples": int(len(val_idx)),
        "history": history, "before": before, "after": after, "angle_correlation": angle_correlation,
    }


def _session_stratified_split(
    session_index: np.ndarray, *, validation_fraction: float, seed: int,
    event_labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out entire recording sessions for validation -- never split
    temporally-adjacent frames from the same session across train/val.

    If `event_labels` is given, guarantees every event class present
    anywhere in the data also appears in validation -- without this, a
    single homogeneous session (e.g. an eva_only recording, which is
    ALWAYS 100% CAST_EVA by construction) can become the entire validation
    set by chance, making "validation accuracy" report a near-meaningless
    number (0 support for two of three classes) rather than a real
    generalization check. Mirrors factorized_v193_training.
    _layout_stratified_episode_split's own established required-class-
    coverage mechanism, adapted for sessions instead of episodes.
    """

    sessions = np.unique(session_index)
    rng = np.random.default_rng(seed)
    if len(sessions) < 2:
        # Single-session smoke datasets: fall back to a random row split so
        # small test fixtures still produce a non-empty validation slice.
        n = len(session_index)
        idx = rng.permutation(n)
        cut = max(1, int(round(n * validation_fraction)))
        return idx[cut:], idx[:cut]
    target_count = max(1, int(round(len(sessions) * validation_fraction)))
    validation_sessions = set(rng.choice(sessions, size=min(target_count, len(sessions) - 1), replace=False).tolist())

    if event_labels is not None:
        for value in np.unique(event_labels).tolist():
            current_mask = np.isin(session_index, list(validation_sessions))
            if np.any(event_labels[current_mask] == value):
                continue
            if len(validation_sessions) + 1 >= len(sessions):
                # Adding another session would consume every session into
                # validation, leaving nothing to train on. Leave this class
                # uncovered in validation instead -- a real data-scarcity
                # fact to surface via the reported diagnostics, not
                # something this split can manufacture by starving train.
                continue
            candidates = [
                int(s) for s in sessions
                if int(s) not in validation_sessions and np.any(event_labels[session_index == s] == value)
            ]
            if candidates:
                validation_sessions.add(candidates[0])
            # If no session anywhere contains this class, there is nothing
            # to add -- train will also lack it, which is a real data-
            # scarcity fact to surface via the reported diagnostics, not
            # something this split can manufacture.

    val_mask = np.isin(session_index, list(validation_sessions))
    return np.flatnonzero(~val_mask), np.flatnonzero(val_mask)


def _load_event_training_pool(dataset_paths: list[str | Path]) -> dict[str, np.ndarray]:
    """Concatenate the human bootstrap dataset with zero or more mined
    Basic-stage DAgger round datasets into one pool for event-head
    training. Session ids are offset per source so sessions never collide
    across files. A file with no `source_recording_role` field (every
    DAgger round dataset -- see basic_environment.save_basic_dagger_
    dataset) defaults its sessions to role "simulator_mined": these are
    real, correctly teacher/policy-labeled examples, but DAgger mining
    concentrates on interesting/difficult states (basic_environment.py's
    4-category classification), so they are exactly as unrepresentative of
    ordinary-play EVENT FREQUENCY as eva_only human sessions already are --
    they must never feed the natural-prior estimate (only human
    direct_keyboard sessions do that), even though they are excellent,
    balanced discrimination examples for the recognition phase."""

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    steering_valid: list[np.ndarray] = []
    session_index: list[np.ndarray] = []
    roles: list[np.ndarray] = []
    offset = 0
    for path in dataset_paths:
        with np.load(Path(path), allow_pickle=False) as data:
            obs = np.asarray(data["observations"], dtype=np.float32)
            act = np.asarray(data["actions"], dtype=np.int64)
            sv = np.asarray(data["steering_label_valid"], dtype=np.bool_)
            sessions = np.asarray(data["session_index"], dtype=np.int64)
            n_sessions = int(sessions.max()) + 1 if len(sessions) else 0
            file_roles = (
                np.asarray(data["source_recording_role"], dtype="<U20")
                if "source_recording_role" in data
                else np.full(n_sessions, "simulator_mined", dtype="<U20")
            )
        observations.append(obs)
        actions.append(act)
        steering_valid.append(sv)
        session_index.append(sessions + offset)
        roles.append(file_roles)
        offset += n_sessions
    return {
        "observations": np.concatenate(observations, axis=0),
        "actions": np.concatenate(actions, axis=0),
        "steering_label_valid": np.concatenate(steering_valid, axis=0),
        "session_index": np.concatenate(session_index, axis=0),
        "source_recording_role": np.concatenate(roles, axis=0),
    }


def _event_score(diagnostics: dict[str, Any]) -> float:
    """Higher is better: the WEAKER of NONE/CAST_EVA recall -- a collapse
    in either direction must be penalized, not hidden by averaging against
    a healthy value on the other class. This is the sole model-selection
    criterion for best-checkpoint restoration and early stopping. No
    prior-drift penalty here: training balance and deployment-calibrated
    probability are different concerns (see bootstrap_event_head's
    docstring) and mixing them into one selection score is what let an
    earlier version of this pipeline pick a calibration-phase checkpoint
    that had quietly collapsed back to always-NONE."""
    per_class = diagnostics["gate"]["heads"]["event"]["per_class"]
    recalls = {entry["value"]: entry["recall"] for entry in per_class if entry["support"] > 0}
    required = [recalls[v] for v in (int(FarmingEvent.NONE), int(FarmingEvent.CAST_EVA)) if v in recalls]
    return float(min(required)) if required else float("-inf")


def _event_gate_report(diagnostics: dict[str, Any], *, minimum_support: int, minimum_recall: float = 0.20) -> dict[str, Any]:
    """Event-head-ONLY pass/fail -- deliberately independent of
    factorized_stage_gate's combined steering+event `passed`/`reasons`.
    That combined gate is exactly the "old combined-training-regime
    plumbing" this project's own review caught: it would fail a perfectly
    good event bootstrap because an intentionally untouched, randomly-
    initialized steering head has near-zero recall by construction. A
    class only has its recall floor enforced once validation support
    reaches `minimum_support` -- below that (JUMP has had as few as 3 real
    human examples in this project's own data), the class is reported as
    underdetermined, never allowed to silently pass OR fail the gate on
    essentially no evidence."""
    per_class = diagnostics["gate"]["heads"]["event"]["per_class"]
    reasons: list[str] = []
    underdetermined: list[int] = []
    for entry in per_class:
        value, support, recall = entry["value"], entry["support"], entry["recall"]
        if support == 0:
            continue
        if support < minimum_support:
            underdetermined.append(value)
            continue
        if recall < minimum_recall:
            reasons.append(f"event class {value} recall {recall:.3f} is below {minimum_recall:.3f} (support={support})")
    return {"passed": not reasons, "reasons": reasons, "underdetermined_classes": underdetermined}


def _macro_f1(diagnostics: dict[str, Any]) -> float:
    """Unweighted mean F1 across event classes with real validation support
    -- reported alongside accuracy/recall/precision so aggregate accuracy
    is never the only number available (see bootstrap_event_head's
    docstring for why aggregate accuracy alone previously hid a collapse)."""
    per_class = diagnostics["gate"]["heads"]["event"]["per_class"]
    scores = []
    for entry in per_class:
        if entry["support"] == 0:
            continue
        p, r = entry["precision"], entry["recall"]
        scores.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def _event_probability_by_true_class(
    policy: Any, observations: np.ndarray, true_event_labels: np.ndarray, *, batch_size: int,
) -> dict[str, Any]:
    """Mean predicted P(event=c) conditioned on the TRUE label -- e.g.
    result["CAST_EVA"]["mean_predicted_probability"][1] is P(EVA | true
    EVA), result["NONE"]["mean_predicted_probability"][1] is P(EVA | true
    NONE). A healthy head needs the former well above the latter; a head
    that only learned the marginal rate has both close to the same value
    (matching this project's real observed failure: ~0.45 for both)."""
    import torch

    policy.eval()
    probs_batches = []
    with torch.no_grad():
        for start in range(0, len(observations), int(batch_size)):
            obs = torch.as_tensor(observations[start : start + int(batch_size)], device=policy.device)
            probs_batches.append(policy.get_distribution(obs).distribution[1].probs.cpu().numpy())
    probs = np.concatenate(probs_batches, axis=0) if probs_batches else np.zeros((0, 3))
    result: dict[str, Any] = {}
    for value, name in ((int(FarmingEvent.NONE), "NONE"), (int(FarmingEvent.CAST_EVA), "CAST_EVA"), (int(FarmingEvent.JUMP), "JUMP")):
        mask = true_event_labels == value
        if not np.any(mask):
            result[name] = {"support": 0, "mean_predicted_probability": [None, None, None]}
        else:
            result[name] = {"support": int(mask.sum()), "mean_predicted_probability": probs[mask].mean(axis=0).tolist()}
    return result


def bootstrap_event_head(
    model: Any,
    dataset_paths: list[str | Path] | str | Path,
    *,
    max_epochs: int = 60,
    learning_rate: float = 2.0e-3,
    batch_size: int = 128,
    validation_fraction: float = 0.2,
    seed: int = 0,
    patience: int = 10,
    minimum_class_support_for_gate: int = 20,
    use_balanced_resampling: bool = False,
    calibration_epochs: int = 0,
    calibration_learning_rate: float = 1.0e-4,
    progress_every_seconds: float = 10.0,
) -> dict[str, Any]:
    """Event-head-only BC from fresh (or continuing) weights: class-weighted
    cross-entropy (factorized_v193_training._sqrt_inverse_class_weights,
    computed from the natural direct_keyboard-only prior), validated every
    epoch, with best-checkpoint restoration and early stopping on a
    class-balanced held-out score (see _event_score) -- not a fixed epoch
    count.

    Design history, so this isn't silently re-broken the same way twice:
    the first version of this function used a two-phase class-balanced-
    resampling "recognition" + natural-prior "calibration" pipeline (ported
    from factorized_v193_training.train_hybrid_factorized_teacher_v193),
    reasoning that plain loss-reweighting could not escape a "learned only
    the marginal rate" collapse (CAST_EVA recall stuck at 0.0 across 4
    rounds of the first canonical Basic run despite mean predicted P(EVA)
    tracking the true rate). A controlled ablation disproved that
    reasoning: plain class-weighted single-phase BC, given the SAME
    optimizer-step budget as the two-phase version, reached comparable
    CAST_EVA recall from the same fresh initialization. The actual root
    cause was training INTENSITY -- 12 epochs @ 3e-4 (copied unmodified
    from train_hybrid_factorized_teacher_v193's own defaults) was tuned for
    a regime with a much larger interleaved scripted+human pool; a
    human-only bootstrap of ~1750 train samples from a genuinely fresh,
    near-input-independent initialization (measured directly: event logit
    std ~0.005 across a real batch before any training) needs substantially
    more optimizer steps to escape that starting point, resampled or not.
    This function defaults to the simpler single-phase path
    (use_balanced_resampling=False) accordingly.

    Optimizer steps and examples seen are reported explicitly (`history`
    entries carry `cumulative_steps`/`cumulative_examples`, and the
    top-level result carries the totals) precisely because "12 epochs" or
    "60 epochs" means something different on every different pool size --
    epoch count alone is not a portable measure of how much training
    actually happened, which is what caused the original mistake.

    Pass/fail is event-head-ONLY (see _event_gate_report) -- never
    conflated with the untouched, randomly-initialized steering head's own
    recall, which the shared factorized_stage_gate's combined `passed`
    would otherwise fail this bootstrap on for a reason that has nothing
    to do with the event head. Also support-aware: a class only has its
    recall floor enforced once validation support reaches
    `minimum_class_support_for_gate`; below that (typically JUMP,
    chronically ~3 real human examples) the class is reported as
    underdetermined, never allowed to silently pass or fail the gate on
    essentially no evidence.

    `use_balanced_resampling=True` opts into the original class-balanced
    per-epoch resampling instead of plain weighted cross-entropy. Kept
    available, not deleted -- a future, more severely imbalanced dataset
    might genuinely need it even though today's event bootstrap does not;
    the ablation that motivated the simpler default did not test every
    possible class-imbalance regime, only the one this project currently
    has.

    `calibration_epochs > 0` opts into an OPTIONAL post-hoc natural-prior
    calibration pass (low learning rate, no resampling) intended to pull
    output probabilities toward the natural direct_keyboard rate without
    losing discrimination. It is kept only if it does not regress the
    event score achieved by the main training phase -- an earlier
    calibration configuration was observed turning a discriminating model
    back into an always-NONE predictor; calibration must never be allowed
    to trade discrimination for a better-matched marginal frequency.

    `dataset_paths` may be a single path (e.g. just the human bootstrap
    dataset) or a list of paths (e.g. human bootstrap + all Basic-stage
    mined DAgger round datasets accumulated so far) -- see
    _load_event_training_pool for how sources are combined and how
    DAgger-mined sessions are kept out of the natural-prior estimate.
    """

    import copy

    import torch

    if isinstance(dataset_paths, (str, Path)):
        dataset_paths = [dataset_paths]
    pool = _load_event_training_pool(list(dataset_paths))
    observations, actions = pool["observations"], pool["actions"]
    steering_label_valid = pool["steering_label_valid"]
    session_index = pool["session_index"]
    session_roles = pool["source_recording_role"]

    if observations.shape[1] != POLICY_INPUT_SIZE:
        raise ValueError(f"dataset observations must be {POLICY_INPUT_SIZE}-valued, got {observations.shape[1]}")

    policy = model.policy
    event_out = getattr(getattr(policy, "action_net", None), "event_out", None)
    if event_out is None or int(getattr(event_out, "out_features", -1)) != 3:
        raise ValueError(
            "policy.action_net.event_out is missing or the wrong shape; is model.policy a SplitSteeringNavigationPolicy?"
        )

    event_valid = np.ones(len(session_index), dtype=np.bool_)
    train_idx, val_idx = _human_session_stratified_split(
        session_index, steering_label_valid, event_valid, actions,
        validation_fraction=validation_fraction, seed=seed,
    )
    if len(val_idx) == 0 or len(train_idx) == 0:
        raise ValueError("event training pool split produced an empty train or validation slice")

    continuous_session_ids = np.flatnonzero(session_roles == "direct_keyboard")
    continuous_mask = np.isin(session_index, continuous_session_ids)

    natural_indices = train_idx[continuous_mask[train_idx]]
    fallback_target = _class_priors(actions[:, 1], train_idx, 3)
    natural_event_target = _class_priors(actions[:, 1], natural_indices, 3) if len(natural_indices) else fallback_target
    natural_event_weight_indices = natural_indices if len(natural_indices) else train_idx
    natural_event_weights = _sqrt_inverse_class_weights(actions[:, 1], natural_event_weight_indices, 3)

    event_values = _training_values(
        actions[:, 1], train_idx, required=(int(FarmingEvent.NONE), int(FarmingEvent.CAST_EVA)),
        optional=(int(FarmingEvent.JUMP),), minimum_optional_support=8,
    )
    event_fractions = _event_sampling_fractions(event_values)

    trainable_names = ("mlp_extractor.event_net", "action_net.event_out")
    for name, param in policy.named_parameters():
        param.requires_grad = any(key in name for key in trainable_names)
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    assert trainable_params, "unreachable: event_out shape check above already validated the architecture"

    rng = np.random.default_rng(seed)
    policy.eval()
    before = _prediction_diagnostics(
        policy, observations[val_idx], actions[val_idx], batch_size=batch_size,
        steering_valid_mask=steering_label_valid[val_idx],
    )

    def _validate() -> dict[str, Any]:
        return _prediction_diagnostics(
            policy, observations[val_idx], actions[val_idx], batch_size=batch_size,
            steering_valid_mask=steering_label_valid[val_idx],
        )

    steering_forced_invalid = np.zeros(len(observations), dtype=np.bool_)
    dummy_steering_weights = np.ones(3, dtype=np.float32)

    # --- main phase: weighted CE by default, or class-balanced resampling
    # if explicitly requested -- validated and best-checkpoint-tracked
    # every epoch, stopped early once `patience` epochs pass with no
    # improvement in _event_score, not run to a fixed count. ---
    optimizer = torch.optim.Adam(trainable_params, lr=float(learning_rate))
    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_score = float("-inf")
    epochs_since_improvement = 0
    cumulative_steps = 0
    progress = ProgressPrinter(int(max_epochs), label="event_head_bootstrap", min_interval_seconds=progress_every_seconds)
    epochs_run = 0
    for epoch in range(1, int(max_epochs) + 1):
        epochs_run = epoch
        if use_balanced_resampling:
            training = _train_balanced_factorized_heads(
                policy, optimizer, observations, actions,
                steering_indices=train_idx, event_indices=train_idx,
                steering_values=(), event_values=event_values,
                epochs=1, batch_size=batch_size, event_loss_scale=1.0, rng=rng,
                event_target_fractions=event_fractions,
            )
            cumulative_steps += int(training["event_updates"])
        else:
            training = _train_natural_prior_epoch(
                policy, optimizer, observations, actions, train_idx,
                batch_size=batch_size, steering_weights=dummy_steering_weights,
                event_weights=natural_event_weights, event_loss_scale=1.0, rng=rng,
                steering_valid_mask=steering_forced_invalid,
            )
            cumulative_steps += int(training["updates"])
        validation = _validate()
        score = _event_score(validation)
        history.append({
            "epoch": epoch, "training": training, "score": score,
            "cumulative_steps": cumulative_steps, "cumulative_examples": cumulative_steps * int(batch_size),
            "event_gate": validation["gate"]["heads"]["event"],
        })
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(policy.state_dict())
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        progress.update(epoch, extra=f"score={score:.3f} best={best_score:.3f} no_improve={epochs_since_improvement}/{patience}")
        if epochs_since_improvement >= int(patience):
            break
    progress.finish()
    if best_state is not None:
        policy.load_state_dict(best_state)
    main_phase_validation = _validate()
    main_phase_score = best_score

    # --- optional calibration phase: natural-prior-weighted, no
    # resampling, small learning rate -- kept only if it does not regress
    # below what the main phase already achieved (see docstring). ---
    calibration_history: list[dict[str, Any]] = []
    bias_correction: dict[str, Any] | None = None
    if int(calibration_epochs) > 0:
        bias_correction = _apply_prior_bias_correction(
            policy, steering_target=np.full(3, 1.0 / 3.0), event_target=natural_event_target,
            event_sampling_fractions=event_fractions,
        )
        calibration_optimizer = torch.optim.Adam(trainable_params, lr=float(calibration_learning_rate))
        calibration_best_state = copy.deepcopy(policy.state_dict())
        calibration_best_score = _event_score(_validate())
        for epoch in range(1, int(calibration_epochs) + 1):
            training = _train_natural_prior_epoch(
                policy, calibration_optimizer, observations, actions, train_idx,
                batch_size=batch_size, steering_weights=dummy_steering_weights,
                event_weights=natural_event_weights, event_loss_scale=1.0, rng=rng,
                steering_valid_mask=steering_forced_invalid,
            )
            cumulative_steps += int(training["updates"])
            validation = _validate()
            score = _event_score(validation)
            calibration_history.append({
                "epoch": epoch, "training": training, "score": score,
                "cumulative_steps": cumulative_steps, "cumulative_examples": cumulative_steps * int(batch_size),
                "event_gate": validation["gate"]["heads"]["event"],
            })
            if score >= calibration_best_score:
                calibration_best_score = score
                calibration_best_state = copy.deepcopy(policy.state_dict())
        if calibration_best_score >= main_phase_score:
            policy.load_state_dict(calibration_best_state)
        else:
            policy.load_state_dict(best_state)

    policy.eval()
    after = _validate()
    probability_by_true_class = _event_probability_by_true_class(
        policy, observations[val_idx], actions[val_idx, 1], batch_size=batch_size,
    )
    gate_report = _event_gate_report(after, minimum_support=minimum_class_support_for_gate)

    dataset_composition = {
        "total_samples": int(len(observations)),
        "total_sessions": int(len(np.unique(session_index))),
        "direct_keyboard_sessions": int(len(continuous_session_ids)),
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
        "validation_event_counts": np.bincount(actions[val_idx, 1], minlength=3).tolist(),
        "train_event_counts": np.bincount(actions[train_idx, 1], minlength=3).tolist(),
        "natural_event_target": natural_event_target.tolist(),
    }

    return {
        "before": before,
        "history": history,
        "epochs_run": epochs_run,
        "stopped_early": epochs_run < int(max_epochs),
        "total_optimizer_steps": cumulative_steps,
        "total_examples_seen": cumulative_steps * int(batch_size),
        "main_phase_validation": main_phase_validation,
        "calibration_history": calibration_history,
        "calibration_applied": bool(calibration_history) and bias_correction is not None,
        "bias_correction": bias_correction,
        "after": after,
        "macro_f1": _macro_f1(after),
        "probability_by_true_class": probability_by_true_class,
        "dataset_composition": dataset_composition,
        "gate_passed": gate_report["passed"],
        "reasons": gate_report["reasons"],
        "underdetermined_classes": gate_report["underdetermined_classes"],
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
    }


def bootstrap_policy_from_human_recordings(
    model: Any,
    dataset_path: str | Path,
    *,
    train_heads: tuple[str, ...] = ("event",),
    epochs: int = 6,
    learning_rate: float = 3.0e-4,
    batch_size: int = 128,
    validation_fraction: float = 0.15,
    seed: int = 0,
    progress_every_seconds: float = 10.0,
) -> dict[str, Any]:
    """BC on human recordings, restricted to `train_heads` (default
    event-only: see module docstring for why human data does not bootstrap
    steering). Pass train_heads=("steering","event") or ("steering",) to
    override -- e.g. for a later, evidence-justified low-weight diagnostic
    fine-tune, never as the initial steering bootstrap. value_net is always
    left untouched: BC on human recordings has no meaningful value target,
    the value function is learned properly once Beginner's PPO phase
    provides real reward signal.

    Steering loss (when trained) is masked per-sample by steering_label_valid
    (click-to-move/EVA-only recordings can only supervise the event head);
    event loss is not masked (event_label_valid is always True in this
    dataset's contract -- see demonstrations.export_demonstrations).

    Event class weights use the NATURAL prior from continuous direct-keyboard
    sessions only (factorized_v193_training._natural_human_event_target),
    not the raw pooled distribution across all sessions -- eva_only
    recordings are curated near-100%-CAST_EVA clips (a real positive
    recognition example) and would otherwise make EVA look far more common
    in ordinary play than it is, distorting the learned prior. This mirrors
    the project's established hybrid-teacher/human pipeline
    (train_hybrid_factorized_teacher_v193), reused here rather than
    reinvented, adapted for the split steering/event architecture.
    """

    import torch
    import torch.nn.functional as F

    valid_heads = {"steering", "event"}
    if not set(train_heads) <= valid_heads or not train_heads:
        raise ValueError(f"train_heads must be a non-empty subset of {valid_heads}, got {train_heads}")

    with np.load(Path(dataset_path), allow_pickle=False) as data:
        observations = np.asarray(data["observations"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.int64)
        steering_label_valid = np.asarray(data["steering_label_valid"], dtype=np.bool_)
        session_index = np.asarray(data["session_index"], dtype=np.int64)
        session_roles = (
            np.asarray(data["source_recording_role"], dtype=str) if "source_recording_role" in data else None
        )

    if observations.shape[1] != POLICY_INPUT_SIZE:
        raise ValueError(f"dataset observations must be {POLICY_INPUT_SIZE}-valued, got {observations.shape[1]}")

    policy = model.policy
    steering_first_layer = getattr(getattr(policy, "mlp_extractor", None), "steering_net", [None])[0]
    actual_steering_input = getattr(steering_first_layer, "in_features", None)
    if actual_steering_input != STEERING_NAVIGATION_FEATURE_SIZE:
        raise ValueError(
            f"policy.mlp_extractor.steering_net expects {actual_steering_input} steering inputs; "
            f"is model.policy a SplitSteeringNavigationPolicy?"
        )

    train_idx, val_idx = _session_stratified_split(
        session_index, validation_fraction=validation_fraction, seed=seed, event_labels=actions[:, 1],
    )
    if len(val_idx) == 0 or len(train_idx) == 0:
        raise ValueError("session-stratified split produced an empty train or validation slice")

    # Class-balanced loss weighting -- without this, plain unweighted
    # cross-entropy on this dataset's real class imbalance (event: ~75%
    # NONE / 25% CAST_EVA / 0.5% JUMP; steering: ~73% STRAIGHT) collapses
    # the argmax to the majority class for every single sample (confirmed
    # empirically on the real canonical human-recording dataset: 3272/3272
    # predictions were STRAIGHT/NONE after 20 unweighted epochs, despite
    # the mean predicted probabilities roughly tracking the true class
    # priors -- the model had learned the marginal rate, not per-observation
    # discrimination). Reuses this codebase's own established fix for
    # exactly this failure mode (factorized_v193_training._sqrt_inverse_
    # class_weights, already used for the non-split architecture's BC
    # training) rather than inventing a new one.
    if session_roles is not None:
        continuous_session_ids = np.flatnonzero(session_roles == "direct_keyboard")
        continuous_mask = np.isin(session_index, continuous_session_ids)
    else:
        continuous_mask = np.ones(len(session_index), dtype=np.bool_)
    natural_event_indices = train_idx[continuous_mask[train_idx]]
    # Fall back to the full (pooled) train set only if there are truly no
    # continuous-session samples to estimate a natural prior from.
    event_weight_indices = natural_event_indices if len(natural_event_indices) > 0 else train_idx
    event_class_weights = torch.as_tensor(
        _sqrt_inverse_class_weights(actions[:, 1], event_weight_indices, 3), dtype=torch.float32,
    )
    steering_train_valid_idx = train_idx[steering_label_valid[train_idx]]
    steering_class_weights = (
        torch.as_tensor(_sqrt_inverse_class_weights(actions[:, 0], steering_train_valid_idx, 3), dtype=torch.float32)
        if len(steering_train_valid_idx) > 0
        else torch.ones(3, dtype=torch.float32)
    )

    head_param_prefixes = {
        "steering": ("mlp_extractor.steering_net", "action_net.steering_out"),
        "event": ("mlp_extractor.event_net", "action_net.event_out"),
    }
    trainable_names = tuple(prefix for head in train_heads for prefix in head_param_prefixes[head])
    trainable_params = []
    for name, param in policy.named_parameters():
        is_trainable = any(key in name for key in trainable_names)
        param.requires_grad = is_trainable
        if is_trainable:
            trainable_params.append(param)
    assert trainable_params, "unreachable: steering_input check above already validated the architecture"

    policy.eval()
    before = _prediction_diagnostics(
        policy, observations[val_idx], actions[val_idx], batch_size=batch_size,
        steering_valid_mask=steering_label_valid[val_idx],
    )

    optimizer = torch.optim.Adam(trainable_params, lr=float(learning_rate))
    policy.train()
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    total_batches = int(epochs) * max(1, (len(train_idx) + batch_size - 1) // batch_size)
    progress = ProgressPrinter(total_batches, label="basic_bootstrap_bc", min_interval_seconds=progress_every_seconds)
    batches_done = 0
    for epoch in range(1, int(epochs) + 1):
        order = rng.permutation(train_idx)
        epoch_steering_loss = 0.0
        epoch_event_loss = 0.0
        n_batches = 0
        for start in range(0, len(order), int(batch_size)):
            batch_idx = order[start : start + int(batch_size)]
            obs = torch.as_tensor(observations[batch_idx], device=policy.device)
            steering_labels = torch.as_tensor(actions[batch_idx, 0], device=policy.device, dtype=torch.long)
            event_labels = torch.as_tensor(actions[batch_idx, 1], device=policy.device, dtype=torch.long)
            steering_valid = torch.as_tensor(steering_label_valid[batch_idx], device=policy.device, dtype=torch.bool)

            distribution = policy.get_distribution(obs).distribution
            event_loss = F.cross_entropy(
                distribution[1].logits, event_labels, weight=event_class_weights.to(policy.device),
            )
            if bool(steering_valid.any()):
                steering_loss = F.cross_entropy(
                    distribution[0].logits[steering_valid], steering_labels[steering_valid],
                    weight=steering_class_weights.to(policy.device),
                )
            else:
                steering_loss = torch.zeros((), device=policy.device)
            # Only trained heads' losses enter the backward pass -- a frozen
            # head's loss is still computed and reported below (useful
            # signal: is it drifting even though nothing should update it?)
            # but never contributes gradient.
            loss = sum(
                (steering_loss if head == "steering" else event_loss) for head in train_heads
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_steering_loss += float(steering_loss.item())
            epoch_event_loss += float(event_loss.item())
            n_batches += 1
            batches_done += 1
            progress.update(batches_done, extra=f"epoch={epoch}/{epochs}")
        history.append({
            "epoch": epoch,
            "mean_steering_loss": epoch_steering_loss / max(1, n_batches),
            "mean_event_loss": epoch_event_loss / max(1, n_batches),
        })
    progress.finish()

    policy.eval()
    after = _prediction_diagnostics(
        policy, observations[val_idx], actions[val_idx], batch_size=batch_size,
        steering_valid_mask=steering_label_valid[val_idx],
    )

    for _name, param in policy.named_parameters():
        param.requires_grad = True  # restore, standard SB3 policies expect all-trainable

    return {
        "dataset": str(Path(dataset_path).resolve()),
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
        "history": history,
        "before": before,
        "after": after,
    }
