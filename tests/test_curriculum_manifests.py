from __future__ import annotations

from pathlib import Path

import pytest

from simulator.curriculum_manifests import (
    ChallengeManifest,
    FixedRegressionScenario,
    GeneratorValidationManifest,
    HeldoutManifest,
    assert_disjoint_from_training,
    load_challenge_manifest,
    load_generator_validation_manifest,
    load_heldout_manifest,
    save_manifest,
)
from simulator.synthetic import generate_curriculum_from_plan


def _tiny_plan(stage: str = "early"):
    return [(stage, "open_field", "typical", "fast", 0), (stage, "wide_neck", "low", "bursty", 0)]


def test_heldout_manifest_round_trips(tmp_path: Path) -> None:
    manifest = HeldoutManifest(
        stage="early", curriculum_path="foo/curriculum.json", layouts=("a", "b"), notes="x"
    )
    path = save_manifest(manifest, tmp_path / "heldout.json")
    loaded = load_heldout_manifest(path)
    assert loaded == manifest


def test_challenge_manifest_round_trips(tmp_path: Path) -> None:
    manifest = ChallengeManifest(
        stage="early",
        fixed_regression_scenarios=(
            FixedRegressionScenario(
                id="x", curriculum_path="foo/curriculum.json", layout="a", seed=1,
                episode_seconds=150.0, max_actions=1000, expected_failure_signature="lock-in",
                discovered="2026-08-06",
            ),
        ),
        challenge_family_curriculum_path="bar/curriculum.json",
        challenge_family_layouts=("c", "d"),
    )
    path = save_manifest(manifest, tmp_path / "challenge.json")
    loaded = load_challenge_manifest(path)
    assert loaded == manifest


def test_generator_validation_manifest_round_trips(tmp_path: Path) -> None:
    manifest = GeneratorValidationManifest(stage="early", entries=(), notes="none yet")
    path = save_manifest(manifest, tmp_path / "gv.json")
    loaded = load_generator_validation_manifest(path)
    assert loaded == manifest


def test_assert_disjoint_from_training_passes_for_a_genuinely_separate_curriculum(tmp_path: Path) -> None:
    training_path = generate_curriculum_from_plan(
        tmp_path / "training", _tiny_plan(), seed=1000, overwrite=True
    )
    heldout_path = generate_curriculum_from_plan(
        tmp_path / "heldout", _tiny_plan(), seed=2000, overwrite=True
    )
    manifest = HeldoutManifest(
        stage="early",
        curriculum_path=str((tmp_path / "heldout" / "curriculum.json")),
        layouts=("01_early_open_field_typical_fast", "02_early_wide_neck_low_bursty"),
    )
    assert_disjoint_from_training(manifest, training_path, manifest_root=".")


def test_assert_disjoint_from_training_catches_a_real_seed_collision(tmp_path: Path) -> None:
    # Same seed used for both "training" and "held-out" -- a real leakage
    # scenario the check must catch, not just a coincidence of naming.
    training_path = generate_curriculum_from_plan(
        tmp_path / "training2", _tiny_plan(), seed=5000, overwrite=True
    )
    leaked_path = generate_curriculum_from_plan(
        tmp_path / "leaked", _tiny_plan(), seed=5000, overwrite=True
    )
    manifest = HeldoutManifest(
        stage="early",
        curriculum_path=str((tmp_path / "leaked" / "curriculum.json")),
        layouts=("01_early_open_field_typical_fast",),
    )
    with pytest.raises(ValueError, match="collides"):
        assert_disjoint_from_training(manifest, training_path, manifest_root=".")


def test_assert_disjoint_from_training_rejects_pointing_at_training_itself(tmp_path: Path) -> None:
    training_path = generate_curriculum_from_plan(
        tmp_path / "training3", _tiny_plan(), seed=9000, overwrite=True
    )
    manifest = HeldoutManifest(
        stage="early", curriculum_path=str(training_path), layouts=("01_early_open_field_typical_fast",)
    )
    with pytest.raises(ValueError, match="training curriculum itself"):
        assert_disjoint_from_training(manifest, training_path, manifest_root=".")
