from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from simulator.basic_training import (
    build_fresh_basic_policy,
    build_human_bootstrap_dataset,
    bootstrap_policy_from_human_recordings,
    canonical_checkpoint_name,
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


def test_bootstrap_policy_from_human_recordings_updates_weights_without_nan(tmp_path: Path) -> None:
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
    assert not torch.allclose(before_steering, after_steering)
    assert not torch.allclose(before_event, after_event)
    assert not torch.isnan(after_steering).any()
    assert not torch.isnan(after_event).any()
    assert all(not np.isnan(h["mean_steering_loss"]) and not np.isnan(h["mean_event_loss"]) for h in result["history"])
    assert result["train_samples"] + result["validation_samples"] == 3


def test_bootstrap_masks_invalid_steering_labels(tmp_path: Path) -> None:
    """A dataset where every steering label is invalid (eva_only-style) must
    still train the event head without the steering loss ever seeing a
    fabricated label -- the masked branch (torch.zeros steering_loss) must
    be exercised, not just the common case."""
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
        model, tmp_path / "bootstrap_no_steering.npz", epochs=1, batch_size=2, validation_fraction=0.4, seed=0,
    )
    for name, param in model.policy.named_parameters():
        assert not torch.isnan(param).any(), f"NaN in {name}"
    assert result["train_samples"] >= 1
