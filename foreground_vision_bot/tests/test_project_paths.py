from __future__ import annotations

from pathlib import Path

import project_paths


def test_default_model_folders_are_separated() -> None:
    assert project_paths.FARMING_MODEL_RELATIVE == Path(
        "models/farming/flyff_ppo"
    )
    assert project_paths.MAPPING_MODEL_RELATIVE == Path(
        "models/mapping/mapper_explorer_ppo"
    )
    assert project_paths.FARMING_MODELS_DIR.parent == project_paths.MODELS_DIR
    assert project_paths.MAPPING_MODELS_DIR.parent == project_paths.MODELS_DIR


def test_relative_paths_resolve_from_application_root() -> None:
    resolved = project_paths.resolve_app_path("models/mapping/example.zip")
    assert resolved == project_paths.APP_ROOT / "models" / "mapping" / "example.zip"


def test_absolute_path_override_is_preserved(tmp_path: Path) -> None:
    model_path = tmp_path / "custom.zip"
    assert project_paths.resolve_app_path(model_path) == model_path
