from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _tiny_policy():
    import torch
    from torch import nn
    from torch.distributions import Categorical

    class TinyPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action_net = nn.Linear(2, 6)
            self.device = torch.device("cpu")

        def get_distribution(self, observations):
            logits = self.action_net(observations.float())
            return SimpleNamespace(
                distribution=(
                    Categorical(logits=logits[:, :3]),
                    Categorical(logits=logits[:, 3:]),
                )
            )

    return TinyPolicy()


def _build_scripted_dataset(path: Path, *, seed: int) -> None:
    from simulator.factorized_v193_training import _layout_stratified_episode_split

    rng = np.random.default_rng(seed)
    layouts = 4
    episodes_per_layout = 4
    samples_per_episode = 20
    episode_index = np.repeat(
        np.arange(layouts * episodes_per_layout, dtype=np.int64), samples_per_episode
    )
    layout_index = np.repeat(
        np.repeat(np.arange(layouts, dtype=np.int64), episodes_per_layout), samples_per_episode
    )
    n = len(episode_index)
    observations = rng.normal(size=(n, 2)).astype(np.float32)
    steering = np.where(
        observations[:, 1] < -0.4, 1, np.where(observations[:, 1] > 0.4, 2, 0)
    ).astype(np.int64)
    event = (observations[:, 0] > 1.25).astype(np.int64)  # ~10% EVA, no jump
    labels = np.column_stack((steering, event)).astype(np.int64)

    train_indices, validation_indices, _validation_episodes = _layout_stratified_episode_split(
        episode_index, layout_index, labels, validation_fraction=0.2, seed=seed
    )
    layout_names = np.asarray([f"layout_{i}" for i in range(layouts)], dtype=str)
    np.savez_compressed(
        path,
        observations=observations,
        actions=labels,
        episode_index=episode_index,
        layout_index=layout_index,
        layout_names=layout_names,
        train_indices=train_indices,
        validation_indices=validation_indices,
        action_contract_id=np.asarray(["latched-forward-factorized-steering-event-v1"]),
        action_nvec=np.asarray([3, 3], dtype=np.int64),
    )


def _build_human_dataset_with_rare_jump(path: Path, *, seed: int) -> None:
    rng = np.random.default_rng(seed)
    sessions = 3
    samples_per_session = 200
    session_index = np.repeat(np.arange(sessions, dtype=np.int64), samples_per_session)
    n = len(session_index)
    observations = rng.normal(size=(n, 2)).astype(np.float32)
    steering = np.where(
        observations[:, 1] < -0.4, 1, np.where(observations[:, 1] > 0.4, 2, 0)
    ).astype(np.int64)
    event = np.zeros(n, dtype=np.int64)
    event[observations[:, 0] > 1.0] = 1  # ~15% EVA
    # A small handful of genuine jumps -- far below what equal-thirds
    # balancing would promote to a full third of every recognition batch,
    # but comfortably above the minimum-support threshold after the 80/20
    # session-stratified train/validation split.
    jump_positions = rng.choice(n, size=24, replace=False)
    event[jump_positions] = 2
    actions = np.column_stack((steering, event)).astype(np.int64)
    steering_label_valid = session_index < (sessions - 1)
    event_label_valid = np.ones(n, dtype=np.bool_)
    np.savez_compressed(
        path,
        observations=observations,
        actions=actions,
        session_index=session_index,
        steering_label_valid=steering_label_valid,
        event_label_valid=event_label_valid,
    )


def _build_human_dataset_with_role(path: Path, *, seed: int) -> None:
    """Two continuous direct_keyboard sessions with a realistically low EVA
    rate, plus one eva_only-role session whose every sample is a real EVA
    cast (matching how demonstrations.py exports eva_only archives: only
    recognized EVA frames survive). An unmasked natural-prior computation
    would let this session drag the "natural" EVA rate far above what
    ordinary continuous play actually shows.
    """

    rng = np.random.default_rng(seed)
    samples_per_continuous_session = 300
    eva_only_samples = 300

    session_index_parts = []
    observations_parts = []
    actions_parts = []
    steering_valid_parts = []

    for session in range(2):
        obs = rng.normal(size=(samples_per_continuous_session, 2)).astype(np.float32)
        steering = np.where(obs[:, 1] < -0.4, 1, np.where(obs[:, 1] > 0.4, 2, 0)).astype(np.int64)
        event = (obs[:, 0] > 1.8).astype(np.int64)  # a low, realistic EVA rate
        actions = np.column_stack((steering, event)).astype(np.int64)
        session_index_parts.append(np.full(samples_per_continuous_session, session, dtype=np.int64))
        observations_parts.append(obs)
        actions_parts.append(actions)
        steering_valid_parts.append(np.ones(samples_per_continuous_session, dtype=np.bool_))

    # eva_only sessions only keep frames where the player actually cast --
    # states genuinely inside the EVA-eligible region -- not random noise
    # unrelated to that condition. Draw obs[:, 0] from the same tail used to
    # trigger EVA above so the event head sees one consistent decision
    # boundary instead of contradictory labels for the same input region.
    eva_obs = np.column_stack(
        (
            rng.normal(loc=2.4, scale=0.3, size=eva_only_samples),
            rng.normal(size=eva_only_samples),
        )
    ).astype(np.float32)
    eva_actions = np.column_stack(
        (np.zeros(eva_only_samples, dtype=np.int64), np.ones(eva_only_samples, dtype=np.int64))
    )
    session_index_parts.append(np.full(eva_only_samples, 2, dtype=np.int64))
    observations_parts.append(eva_obs)
    actions_parts.append(eva_actions)
    steering_valid_parts.append(np.zeros(eva_only_samples, dtype=np.bool_))

    session_index = np.concatenate(session_index_parts)
    observations = np.concatenate(observations_parts)
    actions = np.concatenate(actions_parts)
    steering_label_valid = np.concatenate(steering_valid_parts)
    event_label_valid = np.ones(len(session_index), dtype=np.bool_)
    source_recording_role = np.asarray(["direct_keyboard", "direct_keyboard", "eva_only"])

    np.savez_compressed(
        path,
        observations=observations,
        actions=actions,
        session_index=session_index,
        steering_label_valid=steering_label_valid,
        event_label_valid=event_label_valid,
        source_recording_role=source_recording_role,
    )


def _build_human_dataset(path: Path, *, seed: int, sessions: int = 3) -> None:
    rng = np.random.default_rng(seed)
    samples_per_session = 50
    session_index = np.repeat(np.arange(sessions, dtype=np.int64), samples_per_session)
    n = len(session_index)
    observations = rng.normal(size=(n, 2)).astype(np.float32)
    steering = np.where(
        observations[:, 1] < -0.4, 1, np.where(observations[:, 1] > 0.4, 2, 0)
    ).astype(np.int64)
    event = (observations[:, 0] > 1.25).astype(np.int64)
    actions = np.column_stack((steering, event)).astype(np.int64)
    # The last session is eva_only-role: event supervision only, no steering.
    steering_label_valid = session_index < (sessions - 1)
    event_label_valid = np.ones(n, dtype=np.bool_)
    np.savez_compressed(
        path,
        observations=observations,
        actions=actions,
        session_index=session_index,
        steering_label_valid=steering_label_valid,
        event_label_valid=event_label_valid,
    )


def test_hybrid_training_falls_back_when_no_human_dataset(tmp_path: Path) -> None:
    from simulator.factorized_v193_training import train_hybrid_factorized_teacher_v193

    scripted_path = tmp_path / "scripted.npz"
    _build_scripted_dataset(scripted_path, seed=1)
    model = SimpleNamespace(policy=_tiny_policy())

    report = train_hybrid_factorized_teacher_v193(
        model,
        scripted_path,
        None,
        recognition_epochs=4,
        calibration_epochs=2,
        batch_size=64,
        seed=1,
    )

    assert report["human_dataset_used"] is False
    assert "human_fallback_reason" in report
    assert "phase" in report and "passed" in report


def test_hybrid_training_falls_back_when_too_few_human_sessions(tmp_path: Path) -> None:
    from simulator.factorized_v193_training import train_hybrid_factorized_teacher_v193

    scripted_path = tmp_path / "scripted.npz"
    _build_scripted_dataset(scripted_path, seed=2)
    human_path = tmp_path / "human.npz"
    # sessions=2 with the fixture's "last session is eva_only" rule leaves
    # exactly one steering-capable session -- below minimum_human_sessions=2.
    _build_human_dataset(human_path, seed=2, sessions=2)
    model = SimpleNamespace(policy=_tiny_policy())

    report = train_hybrid_factorized_teacher_v193(
        model,
        scripted_path,
        human_path,
        recognition_epochs=4,
        calibration_epochs=2,
        batch_size=64,
        minimum_human_sessions=2,
        seed=2,
    )

    assert report["human_dataset_used"] is False
    assert "only 1 human session" in report["human_fallback_reason"]


def test_hybrid_training_runs_source_aware_mix_and_reports_separately(tmp_path: Path) -> None:
    from simulator.factorized_v193_training import train_hybrid_factorized_teacher_v193

    scripted_path = tmp_path / "scripted.npz"
    _build_scripted_dataset(scripted_path, seed=3)
    human_path = tmp_path / "human.npz"
    _build_human_dataset(human_path, seed=3, sessions=3)
    model = SimpleNamespace(policy=_tiny_policy())

    report = train_hybrid_factorized_teacher_v193(
        model,
        scripted_path,
        human_path,
        recognition_epochs=6,
        calibration_epochs=4,
        batch_size=64,
        human_fraction=0.35,
        minimum_human_sessions=2,
        seed=3,
    )

    assert report["human_dataset_used"] is True
    assert report["phase"] in {"recognition", "calibration", "complete"}
    # Recognition mixed real scripted and human rounds, not just one source.
    sources = {round_["source"] for round_ in report["recognition_rounds"]}
    assert sources == {"scripted", "human"}
    assert "scripted" in report["recognition_validation"]
    assert "human" in report["recognition_validation"]

    if report["phase"] == "complete":
        assert report["passed"] is True
        assert set(report["validation"]) == {"scripted", "human", "combined"}
        assert len(report["per_layout_validation"]) == 4
        # Only 2 sessions are steering-capable; the eva_only-role session
        # (index 2) may still appear if its samples fell in the human
        # validation split, since event supervision remains valid for it.
        assert len(report["per_session_validation"]) >= 1


def test_factorized_stage_gate_excludes_masked_steering_rows_from_scoring() -> None:
    from simulator.factorized_training import factorized_stage_gate

    # Rows 0-3 have trustworthy steering labels and are predicted exactly.
    # Rows 4-5 carry a fabricated STRAIGHT(0) steering default (as click-to-
    # move sessions do) that the model correctly does NOT reproduce -- those
    # two rows must not count as steering errors once masked.
    truth = np.array([[0, 0], [1, 0], [2, 1], [0, 0], [0, 0], [0, 0]], dtype=np.int64)
    predicted = np.array([[0, 0], [1, 0], [2, 1], [0, 0], [2, 0], [2, 0]], dtype=np.int64)
    mask = np.array([True, True, True, True, False, False])

    unmasked = factorized_stage_gate(truth, predicted)
    masked = factorized_stage_gate(truth, predicted, steering_valid_mask=mask)

    assert unmasked["heads"]["steering"]["accuracy"] < 1.0
    assert masked["heads"]["steering"]["accuracy"] == 1.0
    assert masked["heads"]["steering"]["samples"] == 4
    assert masked["exact_command_accuracy"] == 1.0
    # The event head is scored over every row regardless of the steering mask.
    assert masked["heads"]["event"]["samples"] == unmasked["heads"]["event"]["samples"] == 6


def test_prediction_diagnostics_reports_zero_steering_samples_without_crashing() -> None:
    from simulator.factorized_v193_training import _prediction_diagnostics

    policy = _tiny_policy()
    rng = np.random.default_rng(0)
    n = 20
    observations = rng.normal(size=(n, 2)).astype(np.float32)
    labels = np.column_stack(
        (rng.integers(0, 3, size=n), rng.integers(0, 2, size=n))
    ).astype(np.int64)
    # An eva_only-role session's own validation slice: nothing here is
    # steering-valid, which previously crashed head_report's empty quantile.
    mask = np.zeros(n, dtype=np.bool_)

    report = _prediction_diagnostics(
        policy, observations, labels, batch_size=8, steering_valid_mask=mask
    )

    assert report["steering"]["samples"] == 0
    assert report["steering"]["mean_margin"] == 0.0
    # The event head is unaffected by the steering mask.
    assert report["event"]["samples"] == n


def test_train_natural_prior_epoch_skips_steering_loss_for_masked_rows() -> None:
    import torch

    from simulator.factorized_v193_training import _train_natural_prior_epoch

    policy = _tiny_policy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    rng = np.random.default_rng(0)
    n = 32
    observations = rng.normal(size=(n, 2)).astype(np.float32)
    labels = np.column_stack(
        (rng.integers(0, 3, size=n), rng.integers(0, 2, size=n))
    ).astype(np.int64)
    train_indices = np.arange(n, dtype=np.int64)
    mask = np.zeros(n, dtype=np.bool_)  # nothing steering-valid in this batch

    before_steering_weight = policy.action_net.weight[:3].clone()
    before_steering_bias = policy.action_net.bias[:3].clone()
    before_event_weight = policy.action_net.weight[3:].clone()

    _train_natural_prior_epoch(
        policy,
        optimizer,
        observations,
        labels,
        train_indices,
        batch_size=16,
        steering_weights=np.ones(3, dtype=np.float32),
        event_weights=np.ones(3, dtype=np.float32),
        event_loss_scale=1.0,
        rng=rng,
        steering_valid_mask=mask,
    )

    # No gradient should reach the steering rows -- they were never a real
    # supervision target, so the parameters that only affect steering logits
    # must be untouched, while the event-only rows still trained normally.
    assert torch.equal(policy.action_net.weight[:3], before_steering_weight)
    assert torch.equal(policy.action_net.bias[:3], before_steering_bias)
    assert not torch.equal(policy.action_net.weight[3:], before_event_weight)


def test_hybrid_training_caps_human_jump_oversampling_instead_of_equal_thirds(
    tmp_path: Path,
) -> None:
    from farming.actions import FarmingEvent

    from simulator.factorized_v193_training import train_hybrid_factorized_teacher_v193

    scripted_path = tmp_path / "scripted.npz"
    _build_scripted_dataset(scripted_path, seed=5)
    human_path = tmp_path / "human.npz"
    _build_human_dataset_with_rare_jump(human_path, seed=5)
    model = SimpleNamespace(policy=_tiny_policy())

    report = train_hybrid_factorized_teacher_v193(
        model,
        scripted_path,
        human_path,
        recognition_epochs=6,
        calibration_epochs=2,
        batch_size=64,
        human_fraction=0.5,
        minimum_human_sessions=2,
        seed=5,
    )

    human_rounds = [r for r in report["recognition_rounds"] if r["source"] == "human"]
    assert human_rounds
    jump_key = str(int(FarmingEvent.JUMP))
    rounds_with_jump = [
        r for r in human_rounds if jump_key in r["event_sample_fractions_last_epoch"]
    ]
    assert rounds_with_jump
    for round_ in rounds_with_jump:
        # Equal-thirds balancing (the pre-fix behaviour) would put this near
        # 33%; the capped fractional scheme should keep it near the
        # scripted path's 5% cap instead.
        assert round_["event_sample_fractions_last_epoch"][jump_key] < 0.15


def test_natural_human_event_target_excludes_eva_only_sessions(tmp_path: Path) -> None:
    from farming.actions import FarmingEvent

    from simulator.factorized_v193_training import _natural_human_event_target

    human_path = tmp_path / "human.npz"
    _build_human_dataset_with_role(human_path, seed=7)
    with np.load(human_path, allow_pickle=False) as data:
        labels = data["actions"]
        sessions = data["session_index"]
        event_valid = data["event_label_valid"]
        role = np.asarray(data["source_recording_role"], dtype=str)

    continuous_ids = np.flatnonzero(role == "direct_keyboard")
    continuous_mask = np.isin(sessions, continuous_ids)
    event_train_indices = np.flatnonzero(event_valid)
    scripted_fallback = np.asarray([0.9, 0.1, 0.0])

    target = _natural_human_event_target(
        labels, event_train_indices, continuous_mask, scripted_fallback
    )
    unfiltered = _natural_human_event_target(
        labels,
        event_train_indices,
        np.ones_like(continuous_mask),  # simulates the pre-fix, unfiltered behaviour
        scripted_fallback,
    )

    # The eva_only session is ~100% CAST_EVA by construction (demonstrations.py
    # only ever exports its recognized EVA frames); folding it in should push
    # the estimate well above the continuous-only rate.
    assert target[int(FarmingEvent.CAST_EVA)] < 0.1
    assert unfiltered[int(FarmingEvent.CAST_EVA)] > 0.3
    assert unfiltered[int(FarmingEvent.CAST_EVA)] > target[int(FarmingEvent.CAST_EVA)]


def test_natural_human_event_target_falls_back_when_no_continuous_data() -> None:
    from simulator.factorized_v193_training import _natural_human_event_target

    labels = np.array([[0, 1], [1, 1], [2, 1]], dtype=np.int64)
    event_train_indices = np.array([0, 1, 2], dtype=np.int64)
    continuous_mask = np.zeros(3, dtype=np.bool_)  # nothing continuous at all
    scripted_fallback = np.asarray([0.7, 0.25, 0.05])

    target = _natural_human_event_target(
        labels, event_train_indices, continuous_mask, scripted_fallback
    )

    assert np.array_equal(target, scripted_fallback)


def test_interleave_source_schedule_matches_requested_proportions() -> None:
    from simulator.factorized_v193_training import _interleave_source_schedule

    schedule = _interleave_source_schedule(8, 4)
    assert schedule.count("scripted") == 8
    assert schedule.count("human") == 4
    assert len(schedule) == 12
    # Genuinely interleaved, not one source run to completion first.
    first_half = schedule[:6]
    assert "human" in first_half
    assert "scripted" in first_half

    assert _interleave_source_schedule(0, 0) == []
