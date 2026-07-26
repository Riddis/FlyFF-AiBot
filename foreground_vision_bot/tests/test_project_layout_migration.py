from __future__ import annotations

from pathlib import Path

import migrate_project_layout as migration


def test_model_classifier_separates_mapping_and_farming(tmp_path: Path) -> None:
    assert migration._model_target_root(tmp_path / "mapper_explorer_ppo.zip") == (
        migration.MAPPING_MODELS_DIR
    )
    assert migration._model_target_root(tmp_path / "mapper_checkpoints") == (
        migration.MAPPING_MODELS_DIR
    )
    assert migration._model_target_root(tmp_path / "flyff_ppo.zip") == (
        migration.FARMING_MODELS_DIR
    )
    assert migration._model_target_root(tmp_path / "checkpoints") == (
        migration.FARMING_MODELS_DIR
    )
    assert migration._model_target_root(tmp_path / "mapping") == (
        migration.MAPPING_MODELS_DIR
    )
    assert migration._model_target_root(tmp_path / "best_modelv17.zip") == (
        migration.MAPPING_MODELS_DIR
    )
    assert migration._model_target_root(tmp_path / "evaluations.npz") == (
        migration.MAPPING_MODELS_DIR
    )
    assert migration._model_target_root(tmp_path / "farming") == (
        migration.FARMING_MODELS_DIR
    )
    assert migration._model_target_root(tmp_path / "unknown.bin") is None


def test_move_path_does_not_overwrite_different_file(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "target" / "source.bin"
    source.write_bytes(b"source")
    destination.parent.mkdir()
    destination.write_bytes(b"different")
    report = migration.MigrationReport()

    migration._move_path(source, destination, report, dry_run=False)

    assert source.read_bytes() == b"source"
    assert destination.read_bytes() == b"different"
    assert report.conflicts


def test_move_path_removes_duplicate_source(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "target" / "source.bin"
    source.write_bytes(b"same")
    destination.parent.mkdir()
    destination.write_bytes(b"same")
    report = migration.MigrationReport()

    migration._move_path(source, destination, report, dry_run=False)

    assert not source.exists()
    assert destination.read_bytes() == b"same"
    assert not report.conflicts


def test_legacy_mapper_logs_are_flattened_into_mapping_logs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "training_logs"
    legacy_mapper_logs = source / "mapper_rl"
    legacy_mapper_logs.mkdir(parents=True)
    (legacy_mapper_logs / "events.out").write_text("event", encoding="utf-8")
    target = tmp_path / "app" / "training_logs" / "mapping"
    monkeypatch.setattr(migration, "MAPPING_TRAINING_LOGS_DIR", target)
    monkeypatch.setattr(
        migration,
        "FARMING_TRAINING_LOGS_DIR",
        tmp_path / "app" / "training_logs" / "farming",
    )
    report = migration.MigrationReport()

    migration._migrate_training_logs(source, report, dry_run=False)

    assert (target / "events.out").read_text(encoding="utf-8") == "event"
    assert not legacy_mapper_logs.exists()


def test_existing_farming_log_folder_is_flattened(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "training_logs"
    legacy_farming_logs = source / "farming"
    legacy_farming_logs.mkdir(parents=True)
    (legacy_farming_logs / "events.out").write_text("farm", encoding="utf-8")
    target = tmp_path / "app" / "training_logs" / "farming"
    monkeypatch.setattr(migration, "FARMING_TRAINING_LOGS_DIR", target)
    monkeypatch.setattr(
        migration,
        "MAPPING_TRAINING_LOGS_DIR",
        tmp_path / "app" / "training_logs" / "mapping",
    )
    report = migration.MigrationReport()

    migration._migrate_training_logs(source, report, dry_run=False)

    assert (target / "events.out").read_text(encoding="utf-8") == "farm"
    assert not legacy_farming_logs.exists()


def test_moved_test_paths_drop_duplicate_application_segment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    path = tests_dir / "conftest.py"
    path.write_text(
        'from pathlib import Path\n'
        'PROJECT_DIR = Path(__file__).resolve().parents[1] / ' +
        '"foreground_vision_bot"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(migration, "TESTS_DIR", tests_dir)
    report = migration.MigrationReport()

    migration._repair_moved_test_paths(report, dry_run=False)

    assert ' / "foreground_vision_bot"' not in path.read_text(encoding="utf-8")
    assert report.updated == [str(path)]
