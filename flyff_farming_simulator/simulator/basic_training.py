"""Basic-stage entrypoint: fresh-initialization current-architecture policy,
human-recording BC bootstrap, and the canonical checkpoint naming scheme.

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

from .factorized_v193_training import _prediction_diagnostics
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
    np.savez_compressed(
        output,
        observations=policy_input.astype(np.float32),
        actions=actions,
        steering_label_valid=steering_label_valid,
        event_label_valid=event_label_valid,
        session_index=session_index,
        recent_contact_is_neutral_placeholder=np.asarray([True]),
    )
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


def _session_stratified_split(
    session_index: np.ndarray, *, validation_fraction: float, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out entire recording sessions for validation -- never split
    temporally-adjacent frames from the same session across train/val."""

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
    val_mask = np.isin(session_index, list(validation_sessions))
    return np.flatnonzero(~val_mask), np.flatnonzero(val_mask)


def bootstrap_policy_from_human_recordings(
    model: Any,
    dataset_path: str | Path,
    *,
    epochs: int = 6,
    learning_rate: float = 3.0e-4,
    batch_size: int = 128,
    validation_fraction: float = 0.15,
    seed: int = 0,
    progress_every_seconds: float = 10.0,
) -> dict[str, Any]:
    """Full-policy BC on human recordings: trains steering_net+steering_out
    AND event_net+event_out (unlike fine_tune_steering_branch_v193, which is
    deliberately steering-only for an already-competent policy -- a fresh
    Basic policy has no competent event head to protect yet). value_net is
    left untouched: BC on human recordings has no meaningful value target,
    the value function is learned properly once Beginner's PPO phase
    provides real reward signal.

    Steering loss is masked per-sample by steering_label_valid (click-to-move/
    EVA-only recordings can only supervise the event head); event loss is not
    masked (event_label_valid is always True in this dataset's contract --
    see demonstrations.export_demonstrations).
    """

    import torch
    import torch.nn.functional as F

    with np.load(Path(dataset_path), allow_pickle=False) as data:
        observations = np.asarray(data["observations"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.int64)
        steering_label_valid = np.asarray(data["steering_label_valid"], dtype=np.bool_)
        session_index = np.asarray(data["session_index"], dtype=np.int64)

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

    train_idx, val_idx = _session_stratified_split(session_index, validation_fraction=validation_fraction, seed=seed)
    if len(val_idx) == 0 or len(train_idx) == 0:
        raise ValueError("session-stratified split produced an empty train or validation slice")

    trainable_names = ("mlp_extractor.steering_net", "action_net.steering_out", "mlp_extractor.event_net", "action_net.event_out")
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
            event_loss = F.cross_entropy(distribution[1].logits, event_labels)
            if bool(steering_valid.any()):
                steering_loss = F.cross_entropy(
                    distribution[0].logits[steering_valid], steering_labels[steering_valid],
                )
            else:
                steering_loss = torch.zeros((), device=policy.device)
            loss = steering_loss + event_loss

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
