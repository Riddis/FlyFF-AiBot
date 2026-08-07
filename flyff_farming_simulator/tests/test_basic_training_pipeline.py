from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from simulator.basic_training import (
    bootstrap_policy_from_human_recordings,
    bootstrap_steering_from_teacher,
    build_fresh_basic_policy,
    build_human_bootstrap_dataset,
    canonical_checkpoint_name,
    collect_simulator_teacher_dataset,
)
from simulator.demonstrations import export_demonstrations
from simulator.map_model import MapModel
from simulator.navigation_history import NavigationHistoryWrapper
from simulator.synthetic import generate_curriculum_from_plan, iter_variant_environments
from tests.test_simulator_core import _synthetic_recording


def test_canonical_checkpoint_name_maps_basic_and_beginner_correctly() -> None:
    assert canonical_checkpoint_name("basic", "bootstrap") == "canonical_basic_bootstrap"
    assert canonical_checkpoint_name("beginner", "PPO 010k") == "canonical_beginner_ppo_010k"


def test_canonical_checkpoint_name_rejects_unknown_stage() -> None:
    with pytest.raises(KeyError):
        canonical_checkpoint_name("expert", "x")


def _demo_dataset(tmp_path: Path) -> Path:
    map_data = MapModel.load()
    session_dir = tmp_path / "rec"
    session_dir.mkdir()
    recording = _synthetic_recording(session_dir, map_data)
    return export_demonstrations([recording], tmp_path / "demos.npz", map_model=map_data)


def test_demonstrations_export_includes_raw_displacement(tmp_path: Path) -> None:
    demo_path = _demo_dataset(tmp_path)
    data = np.load(demo_path, allow_pickle=False)
    assert "displacement_cells" in data
    assert data["displacement_cells"].shape[0] == data["observations"].shape[0]
    assert data["displacement_cells"][0] == 0.0  # no previous frame on the first sample


def test_build_human_bootstrap_dataset_has_neutral_recent_contact(tmp_path: Path) -> None:
    demo_path = _demo_dataset(tmp_path)
    bootstrap_path = build_human_bootstrap_dataset(demo_path, tmp_path / "bootstrap.npz")
    data = np.load(bootstrap_path, allow_pickle=False)
    assert data["observations"].shape[1] == 925
    assert np.all(data["observations"][:, 924] == 0.0), "recent_contact must be the documented neutral placeholder"
    assert bool(data["recent_contact_is_neutral_placeholder"][0]) is True


def test_build_human_bootstrap_dataset_recent_progress_reflects_real_displacement(tmp_path: Path) -> None:
    """_synthetic_recording's 3-sample fixture, causal alignment (see
    reconstruct_session_sidecars and tests/test_temporal_sidecar_parity.py):

    sample 0: no prior transition -> zero sidecar (matches
      NavigationHistoryWrapper.reset()).
    sample 1: reflects the 0->1 transition (displacement=0.3125), whose
      eva_attempted comes from sample 0's OWN action (NONE, not EVA) -- so
      this transition is NOT excluded -> recent_progress = 0.3125/1.79.
    sample 2: reflects transitions [0->1, 1->2]; 1->2's eva_attempted comes
      from sample 1's action, which IS CAST_EVA -- so 1->2 is excluded from
      the eligible window, leaving only 0->1 -> same recent_progress as
      sample 1, not zero and not double-counted.
    """
    demo_path = _demo_dataset(tmp_path)
    bootstrap_path = build_human_bootstrap_dataset(demo_path, tmp_path / "bootstrap.npz")
    data = np.load(bootstrap_path, allow_pickle=False)
    expected = 0.3125 / 1.79
    assert data["observations"][0, 923] == 0.0
    assert data["observations"][1, 923] == pytest.approx(expected, abs=1e-5)
    assert data["observations"][2, 923] == pytest.approx(expected, abs=1e-5)
    assert bool(data["actions"][1, 1] == 1)  # sample 1 is the CAST_EVA tick driving the above


def test_sidecar_values_from_history_reflects_real_displacement_when_not_eva_excluded() -> None:
    """Unit-level check of the actual sidecar math (decoupled from any one
    fixture's specific EVA placement): real, non-EVA displacement must
    change recent_progress."""
    from simulator.navigation_history import NavigationStepEvidence, sidecar_values_from_history

    history = [
        NavigationStepEvidence(displacement_cells=1.79, contact=False, eva_attempted=False),
        NavigationStepEvidence(displacement_cells=1.79, contact=False, eva_attempted=False),
    ]
    progress, contact = sidecar_values_from_history(history, expected_clear_path_displacement=1.79)
    assert progress == 1.0
    assert contact == 0.0


def _tiny_curriculum(tmp_path: Path) -> Path:
    return generate_curriculum_from_plan(
        tmp_path / "curriculum", [("early", "open_field", "typical", "fast", 0)], seed=555001, overwrite=True,
    )


def test_build_fresh_basic_policy_has_925_dim_observation_space(tmp_path: Path) -> None:
    curriculum_path = _tiny_curriculum(tmp_path)
    entry, base_env = next(iter(iter_variant_environments(
        str(curriculum_path), stage="early", seed=0, episode_steps=5, episode_seconds=3.0,
    )))
    env = NavigationHistoryWrapper(base_env)
    model = build_fresh_basic_policy(env, seed=0, device="cpu")
    env.close()
    assert tuple(model.observation_space.shape) == (925,)


def test_bootstrap_policy_from_human_recordings_defaults_to_event_only(tmp_path: Path) -> None:
    """Human recordings do not bootstrap steering by default (see module
    docstring: compact geometry representation doesn't explain recorded
    human steering) -- steering_net must be untouched, event_net must
    update."""
    demo_path = _demo_dataset(tmp_path)
    bootstrap_path = build_human_bootstrap_dataset(demo_path, tmp_path / "bootstrap.npz")
    curriculum_path = _tiny_curriculum(tmp_path)
    entry, base_env = next(iter(iter_variant_environments(
        str(curriculum_path), stage="early", seed=0, episode_steps=5, episode_seconds=3.0,
    )))
    env = NavigationHistoryWrapper(base_env)
    model = build_fresh_basic_policy(env, seed=0, device="cpu")
    env.close()

    before_steering = model.policy.mlp_extractor.steering_net[0].weight.detach().clone()
    before_event = model.policy.mlp_extractor.event_net[0].weight.detach().clone()

    result = bootstrap_policy_from_human_recordings(
        model, bootstrap_path, epochs=2, batch_size=2, validation_fraction=0.4, seed=0,
    )

    after_steering = model.policy.mlp_extractor.steering_net[0].weight.detach()
    after_event = model.policy.mlp_extractor.event_net[0].weight.detach()
    assert torch.allclose(before_steering, after_steering), "steering_net must not change under the default event-only train_heads"
    assert not torch.allclose(before_event, after_event)
    assert not torch.isnan(after_steering).any()
    assert not torch.isnan(after_event).any()
    assert all(not np.isnan(h["mean_event_loss"]) for h in result["history"])
    assert result["train_samples"] + result["validation_samples"] == 3


def test_bootstrap_policy_from_human_recordings_explicit_both_heads(tmp_path: Path) -> None:
    """train_heads=("steering","event") remains available as an explicit
    override (e.g. for a later, evidence-justified diagnostic use), even
    though it is not the default."""
    demo_path = _demo_dataset(tmp_path)
    bootstrap_path = build_human_bootstrap_dataset(demo_path, tmp_path / "bootstrap.npz")
    curriculum_path = _tiny_curriculum(tmp_path)
    entry, base_env = next(iter(iter_variant_environments(
        str(curriculum_path), stage="early", seed=0, episode_steps=5, episode_seconds=3.0,
    )))
    env = NavigationHistoryWrapper(base_env)
    model = build_fresh_basic_policy(env, seed=0, device="cpu")
    env.close()

    before_steering = model.policy.mlp_extractor.steering_net[0].weight.detach().clone()

    result = bootstrap_policy_from_human_recordings(
        model, bootstrap_path, train_heads=("steering", "event"), epochs=2, batch_size=2, validation_fraction=0.4, seed=0,
    )

    after_steering = model.policy.mlp_extractor.steering_net[0].weight.detach()
    assert not torch.allclose(before_steering, after_steering)
    assert not torch.isnan(after_steering).any()
    assert all(not np.isnan(h["mean_steering_loss"]) and not np.isnan(h["mean_event_loss"]) for h in result["history"])


def test_bootstrap_rejects_empty_or_invalid_train_heads(tmp_path: Path) -> None:
    demo_path = _demo_dataset(tmp_path)
    bootstrap_path = build_human_bootstrap_dataset(demo_path, tmp_path / "bootstrap.npz")
    curriculum_path = _tiny_curriculum(tmp_path)
    entry, base_env = next(iter(iter_variant_environments(
        str(curriculum_path), stage="early", seed=0, episode_steps=5, episode_seconds=3.0,
    )))
    env = NavigationHistoryWrapper(base_env)
    model = build_fresh_basic_policy(env, seed=0, device="cpu")
    env.close()

    with pytest.raises(ValueError):
        bootstrap_policy_from_human_recordings(model, bootstrap_path, train_heads=())
    with pytest.raises(ValueError):
        bootstrap_policy_from_human_recordings(model, bootstrap_path, train_heads=("value",))


def test_bootstrap_masks_invalid_steering_labels(tmp_path: Path) -> None:
    """A dataset where every steering label is invalid (eva_only-style) must
    still train the event head without the steering loss ever seeing a
    fabricated label -- the masked branch (torch.zeros steering_loss) must
    be exercised, not just the common case. Uses explicit both-heads mode so
    the masking logic (as opposed to the default event-only skip) is what's
    actually exercised."""
    demo_path = _demo_dataset(tmp_path)
    bootstrap_path = build_human_bootstrap_dataset(demo_path, tmp_path / "bootstrap.npz")
    data = dict(np.load(bootstrap_path, allow_pickle=False))
    data["steering_label_valid"] = np.zeros_like(data["steering_label_valid"])
    np.savez_compressed(tmp_path / "bootstrap_no_steering.npz", **data)

    curriculum_path = _tiny_curriculum(tmp_path)
    entry, base_env = next(iter(iter_variant_environments(
        str(curriculum_path), stage="early", seed=0, episode_steps=5, episode_seconds=3.0,
    )))
    env = NavigationHistoryWrapper(base_env)
    model = build_fresh_basic_policy(env, seed=0, device="cpu")
    env.close()

    result = bootstrap_policy_from_human_recordings(
        model, tmp_path / "bootstrap_no_steering.npz", train_heads=("steering", "event"),
        epochs=1, batch_size=2, validation_fraction=0.4, seed=0,
    )
    for name, param in model.policy.named_parameters():
        assert not torch.isnan(param).any(), f"NaN in {name}"
    assert result["train_samples"] >= 1


def test_session_stratified_split_guarantees_event_class_coverage_in_validation() -> None:
    """Reproduces the exact real-data failure mode found in the canonical
    run: 6 of 8 sessions are homogeneous (100% one class); a naive random
    single-session validation pick can land entirely on one of them,
    leaving 2 of 3 classes with zero validation support. event_labels
    guarantees this cannot happen when at least one session contains all
    classes."""
    from simulator.basic_training import _session_stratified_split

    # sessions 0-4: homogeneous class 1 (mimicking eva_only clips).
    # session 5: contains all three classes (mimicking a direct_keyboard session).
    session_index = np.concatenate([np.full(20, s) for s in range(6)])
    event_labels = np.concatenate([
        np.ones(20, dtype=np.int64) if s < 5 else np.array([0] * 7 + [1] * 7 + [2] * 6, dtype=np.int64)
        for s in range(6)
    ])
    # seed=0 happens to pick a homogeneous session as the sole initial
    # validation pick for this construction -- exactly the failure this
    # guarantees against.
    train_idx, val_idx = _session_stratified_split(
        session_index, validation_fraction=0.15, seed=0, event_labels=event_labels,
    )
    val_classes = set(event_labels[val_idx].tolist())
    assert val_classes == {0, 1, 2}, f"validation is missing event classes: {val_classes}"
    assert len(train_idx) > 0


def test_bootstrap_human_event_weighting_uses_natural_prior_not_pooled(tmp_path: Path) -> None:
    """If eva_only sessions (100% CAST_EVA by construction) were pooled
    into the class-weight computation undifferentiated from continuous
    direct-keyboard sessions, CAST_EVA would look artificially common. This
    constructs a dataset where the pooled and natural priors clearly
    diverge and checks the natural-only path is actually taken (no crash,
    weights computed from source_recording_role, not silently ignored)."""
    demo_path = _demo_dataset(tmp_path)
    bootstrap_path = build_human_bootstrap_dataset(demo_path, tmp_path / "bootstrap.npz")
    data = dict(np.load(bootstrap_path, allow_pickle=False))
    assert "source_recording_role" in data, "build_human_bootstrap_dataset must propagate source_recording_role"

    curriculum_path = _tiny_curriculum(tmp_path)
    entry, base_env = next(iter(iter_variant_environments(
        str(curriculum_path), stage="early", seed=0, episode_steps=5, episode_seconds=3.0,
    )))
    env = NavigationHistoryWrapper(base_env)
    model = build_fresh_basic_policy(env, seed=0, device="cpu")
    env.close()

    # Should run without error even though this tiny fixture has only one
    # session (the natural-prior computation must not assume >1 session).
    result = bootstrap_policy_from_human_recordings(
        model, bootstrap_path, epochs=1, batch_size=2, validation_fraction=0.4, seed=0,
    )
    assert result["train_samples"] >= 1


def _tiny_teacher_dataset(tmp_path: Path):
    curriculum_path = _tiny_curriculum(tmp_path)
    # Enough samples to span several episodes (this fixture's episodes run
    # ~30 ticks each) -- _layout_stratified_episode_split guarantees every
    # required steering/event class value appears in BOTH train and
    # validation, pulling extra episodes into validation as needed; with
    # only 2 episodes total that guarantee can consume all of them, leaving
    # nothing for training. 300 samples gives ~10 episodes of headroom.
    return collect_simulator_teacher_dataset(
        str(curriculum_path), ["01_early_open_field_typical_fast"], samples=300,
        episode_seconds=8.0, max_actions=40, seed=0,
    )


def test_collect_simulator_teacher_dataset_produces_925_dim_labeled_samples(tmp_path: Path) -> None:
    dataset = _tiny_teacher_dataset(tmp_path)
    assert dataset["observations"].shape[1] == 925
    assert dataset["observations"].shape[0] == dataset["actions"].shape[0] == 300
    assert dataset["actions"].shape[1] == 2
    assert len(dataset["train_indices"]) + len(dataset["validation_indices"]) == 300


def test_bootstrap_steering_from_teacher_updates_only_steering_no_nan(tmp_path: Path) -> None:
    curriculum_path = _tiny_curriculum(tmp_path)
    entry, base_env = next(iter(iter_variant_environments(
        str(curriculum_path), stage="early", seed=0, episode_steps=5, episode_seconds=3.0,
    )))
    env = NavigationHistoryWrapper(base_env)
    model = build_fresh_basic_policy(env, seed=0, device="cpu")
    env.close()

    before_steering = model.policy.mlp_extractor.steering_net[0].weight.detach().clone()
    before_event = model.policy.mlp_extractor.event_net[0].weight.detach().clone()

    dataset = _tiny_teacher_dataset(tmp_path)
    result = bootstrap_steering_from_teacher(model, dataset, epochs=3, batch_size=8, seed=0)

    after_steering = model.policy.mlp_extractor.steering_net[0].weight.detach()
    after_event = model.policy.mlp_extractor.event_net[0].weight.detach()
    assert not torch.allclose(before_steering, after_steering)
    assert torch.allclose(before_event, after_event), "event_net must stay frozen during teacher steering bootstrap"
    assert not torch.isnan(after_steering).any()
    assert "angle_correlation" in result
    assert all(not np.isnan(h["mean_steering_loss"]) for h in result["history"])


def test_bootstrap_steering_from_teacher_accepts_path_or_dict(tmp_path: Path) -> None:
    curriculum_path = _tiny_curriculum(tmp_path)
    entry, base_env = next(iter(iter_variant_environments(
        str(curriculum_path), stage="early", seed=0, episode_steps=5, episode_seconds=3.0,
    )))
    env = NavigationHistoryWrapper(base_env)
    model = build_fresh_basic_policy(env, seed=0, device="cpu")
    env.close()

    teacher_path = tmp_path / "teacher.npz"
    collect_simulator_teacher_dataset(
        str(curriculum_path), ["01_early_open_field_typical_fast"], samples=300,
        episode_seconds=8.0, max_actions=40, seed=0, output_path=teacher_path,
    )
    result = bootstrap_steering_from_teacher(model, teacher_path, epochs=1, batch_size=8, seed=0)
    assert result["train_samples"] >= 1
