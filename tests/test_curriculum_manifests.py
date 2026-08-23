from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulator.curriculum_manifests import (
    SIMULATOR_ROOT,
    ChallengeManifest,
    FixedRegressionScenario,
    GeneratorValidationManifest,
    HeldoutManifest,
    assert_disjoint_from_training,
    load_challenge_manifest,
    load_generator_validation_manifest,
    load_heldout_manifest,
    resolve_manifest_curriculum_path,
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


def test_resolve_manifest_curriculum_path_is_relative_to_the_simulator_package_root() -> None:
    """Manifest curriculum_path fields (e.g. "curricula/foo/curriculum.json")
    are stored relative to `simulator/` (this module's own directory), never
    the repo root, the manifest JSON's directory, or a subprocess's cwd --
    the actual bug behind the Basic raw-diagnostic FileNotFoundError (see
    MISTAKES.md 2026-08-24)."""
    resolved = resolve_manifest_curriculum_path("curricula/synthetic_curriculum_heldout/curriculum.json")
    assert resolved == SIMULATOR_ROOT / "curricula" / "synthetic_curriculum_heldout" / "curriculum.json"
    assert resolved.exists()


def test_resolve_manifest_curriculum_path_passes_through_an_already_absolute_path(tmp_path: Path) -> None:
    absolute = tmp_path / "curriculum.json"
    assert resolve_manifest_curriculum_path(str(absolute)) == absolute
    assert resolve_manifest_curriculum_path(absolute) == absolute


def test_every_current_manifest_curriculum_path_resolves_to_an_existing_file() -> None:
    """The same relative-path convention is shared by every manifest under
    simulator/evaluations/manifests/ (Basic diagnostics, Beginner,
    Intermediate, Advanced all load through load_heldout_manifest/
    load_challenge_manifest) -- prove none of the current manifests'
    curriculum_path fields are broken the same way Basic's raw diagnostic
    was, not just the one that already crashed."""
    manifests_dir = SIMULATOR_ROOT / "evaluations" / "manifests"
    checked = 0
    for manifest_path in sorted(manifests_dir.glob("*.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        role = payload.get("role")
        if role == "heldout":
            resolved = resolve_manifest_curriculum_path(payload["curriculum_path"])
            assert resolved.exists(), f"{manifest_path.name}: curriculum_path {payload['curriculum_path']!r} -> {resolved} does not exist"
            checked += 1
        elif role == "challenge":
            resolved = resolve_manifest_curriculum_path(payload["challenge_family_curriculum_path"])
            assert resolved.exists(), f"{manifest_path.name}: challenge_family_curriculum_path -> {resolved} does not exist"
            for scenario in payload["fixed_regression_scenarios"]:
                scenario_resolved = resolve_manifest_curriculum_path(scenario["curriculum_path"])
                assert scenario_resolved.exists(), f"{manifest_path.name}: scenario {scenario['id']!r} curriculum_path -> {scenario_resolved} does not exist"
            checked += 1
    assert checked > 0, "expected at least one heldout/challenge manifest under simulator/evaluations/manifests/"
