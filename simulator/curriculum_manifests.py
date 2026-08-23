"""Immutable curriculum-evaluation manifests: three roles per stage.

- ``easy_heldout`` (etc.): representative unseen same-stage layouts used for
  normal graduation thresholds.
- ``easy_challenge`` (etc.): valid but unusually difficult layouts. Two
  layers -- fixed regression scenarios (exact layout/seed/heading tuples
  that previously exposed a real failure, kept so future checkpoints are
  automatically re-tested against known regressions) and challenge
  families (fresh immutable siblings with the same stress characteristics,
  so a checkpoint cannot pass by memorizing the exact regression seeds).
  Lower performance floor than held-out, but strict no-collapse requirements.
- ``generator_validation``: layouts suspected of violating the generator's
  own constraints (escapability, valid spawn population, intended stage
  geometry). Excluded from policy scoring until a human confirms the
  violation; this role exists so a future audit has somewhere to put a
  layout without silently discarding it or letting it contaminate scoring.

All manifests are plain, hand-inspectable JSON files under
``evaluations/manifests/`` -- never regenerated implicitly, so a promotion
decision always evaluates the exact same layouts run after run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .synthetic import SyntheticCurriculum

# Every manifest's curriculum_path field (HeldoutManifest.curriculum_path,
# ChallengeManifest.challenge_family_curriculum_path,
# FixedRegressionScenario.curriculum_path) is stored relative to this
# module's own directory (the `simulator/` package root), e.g.
# "curricula/synthetic_curriculum_heldout/curriculum.json" ->
# simulator/curricula/synthetic_curriculum_heldout/curriculum.json -- NOT
# relative to the repo root, the manifest JSON's own directory, or a
# subprocess's cwd. An already-absolute curriculum_path (as many tests and
# scratchpad tools construct directly) passes through unchanged.
SIMULATOR_ROOT = Path(__file__).resolve().parent


def resolve_manifest_curriculum_path(curriculum_path: str | Path) -> Path:
    """The one canonical resolution rule for a manifest curriculum_path
    field -- every caller that turns `curriculum_path` into a filesystem
    path (to load a `SyntheticCurriculum` or run an episode against it) must
    go through this function instead of treating the stored string as
    directly openable relative to whatever the current process's cwd
    happens to be."""

    path = Path(curriculum_path)
    return path if path.is_absolute() else (SIMULATOR_ROOT / path)


@dataclass(frozen=True)
class FixedRegressionScenario:
    """One exact, reproducible known-failure case."""

    id: str
    curriculum_path: str
    layout: str
    seed: int
    episode_seconds: float
    max_actions: int
    expected_failure_signature: str
    discovered: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "FixedRegressionScenario":
        return FixedRegressionScenario(**data)


@dataclass(frozen=True)
class HeldoutManifest:
    stage: str
    curriculum_path: str
    layouts: tuple[str, ...]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "stage": self.stage,
            "role": "heldout",
            "curriculum_path": self.curriculum_path,
            "layouts": list(self.layouts),
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "HeldoutManifest":
        return HeldoutManifest(
            stage=data["stage"],
            curriculum_path=data["curriculum_path"],
            layouts=tuple(data["layouts"]),
            notes=data.get("notes", ""),
        )


@dataclass(frozen=True)
class ChallengeManifest:
    stage: str
    fixed_regression_scenarios: tuple[FixedRegressionScenario, ...]
    challenge_family_curriculum_path: str
    challenge_family_layouts: tuple[str, ...]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "stage": self.stage,
            "role": "challenge",
            "fixed_regression_scenarios": [s.to_dict() for s in self.fixed_regression_scenarios],
            "challenge_family_curriculum_path": self.challenge_family_curriculum_path,
            "challenge_family_layouts": list(self.challenge_family_layouts),
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ChallengeManifest":
        return ChallengeManifest(
            stage=data["stage"],
            fixed_regression_scenarios=tuple(
                FixedRegressionScenario.from_dict(s) for s in data["fixed_regression_scenarios"]
            ),
            challenge_family_curriculum_path=data["challenge_family_curriculum_path"],
            challenge_family_layouts=tuple(data["challenge_family_layouts"]),
            notes=data.get("notes", ""),
        )


@dataclass(frozen=True)
class GeneratorValidationManifest:
    stage: str
    entries: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "stage": self.stage,
            "role": "generator_validation",
            "entries": list(self.entries),
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "GeneratorValidationManifest":
        return GeneratorValidationManifest(
            stage=data["stage"], entries=tuple(data.get("entries", ())), notes=data.get("notes", "")
        )


def save_manifest(manifest: HeldoutManifest | ChallengeManifest | GeneratorValidationManifest, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8")
    return target


def load_heldout_manifest(path: str | Path) -> HeldoutManifest:
    return HeldoutManifest.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_challenge_manifest(path: str | Path) -> ChallengeManifest:
    return ChallengeManifest.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_generator_validation_manifest(path: str | Path) -> GeneratorValidationManifest:
    return GeneratorValidationManifest.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _variant_seeds(curriculum_path: str | Path) -> dict[str, int]:
    curriculum = SyntheticCurriculum.load(Path(curriculum_path))
    return {entry.name: entry.seed for entry in curriculum.variants}


def assert_disjoint_from_training(
    manifest: HeldoutManifest | ChallengeManifest,
    training_curriculum_path: str | Path,
    *,
    manifest_root: str | Path,
) -> None:
    """Prove none of a manifest's layouts can be the exact same generated
    layout as anything in the training curriculum.

    Checks both the curriculum file path (different directories are
    trivially disjoint) and, more robustly, the per-variant generation
    seed recorded in each curriculum's own manifest -- so a future accident
    that points a held-out/challenge manifest at the training curriculum's
    directory, or regenerates a manifest with a colliding seed, fails loudly
    instead of silently leaking training layouts into evaluation.
    """

    training_seeds = _variant_seeds(training_curriculum_path)
    root = Path(manifest_root)

    def check_curriculum(curriculum_path: str, layouts: tuple[str, ...]) -> None:
        resolved = (root / curriculum_path).resolve()
        if resolved == Path(training_curriculum_path).resolve():
            raise ValueError(
                f"Manifest points at the training curriculum itself ({curriculum_path}); "
                "held-out/challenge layouts must come from a separate generation."
            )
        seeds = _variant_seeds(resolved)
        for layout in layouts:
            if layout not in seeds:
                raise ValueError(f"{layout!r} not found in {curriculum_path}")
            seed = seeds[layout]
            if seed in training_seeds.values():
                raise ValueError(
                    f"{layout!r} (seed {seed}) collides with a training-curriculum seed -- "
                    "possible leakage into PPO training or teacher-data generation."
                )

    if isinstance(manifest, HeldoutManifest):
        check_curriculum(manifest.curriculum_path, manifest.layouts)
    else:
        check_curriculum(manifest.challenge_family_curriculum_path, manifest.challenge_family_layouts)
        for scenario in manifest.fixed_regression_scenarios:
            check_curriculum(scenario.curriculum_path, (scenario.layout,))
