from __future__ import annotations

import argparse
import filecmp
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from project_paths import (
    APP_ROOT,
    FARMING_MODELS_DIR,
    FARMING_TRAINING_LOGS_DIR,
    MAPPING_MODELS_DIR,
    MAPPING_TRAINING_LOGS_DIR,
    TESTS_DIR,
    ensure_project_layout,
)


@dataclass
class MigrationReport:
    moved: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.conflicts


def migrate_project_layout(*, dry_run: bool = False) -> MigrationReport:
    """Move legacy root output into the application-owned folder structure.

    The operation is idempotent and never overwrites a different existing file.
    It checks both the application directory and its parent because older runs
    may have been launched from either location.
    """

    report = MigrationReport()
    if not dry_run:
        ensure_project_layout()

    roots = _legacy_roots()
    for root in roots:
        _migrate_models(root / "models", report, dry_run=dry_run)
        _migrate_training_logs(root / "training_logs", report, dry_run=dry_run)

    repository_tests = APP_ROOT.parent / "tests"
    if repository_tests != TESTS_DIR:
        _merge_tree(repository_tests, TESTS_DIR, report, dry_run=dry_run)

    _repair_moved_test_paths(report, dry_run=dry_run)
    return report



def _repair_moved_test_paths(
    report: MigrationReport,
    *,
    dry_run: bool,
) -> None:
    """Repair test paths that assumed tests lived beside the app folder.

    Before v1.8.1, tests were stored at ``<repo>/tests`` and commonly built
    the application path by appending ``foreground_vision_bot``.  Once those
    tests are moved into ``foreground_vision_bot/tests``, that expression
    points at a non-existent nested application directory.
    """

    replacements = (
        (
            'Path(__file__).resolve().parents[1] / "foreground_vision_bot"',
            'Path(__file__).resolve().parents[1]',
        ),
        (
            "Path(__file__).resolve().parents[1] / 'foreground_vision_bot'",
            "Path(__file__).resolve().parents[1]",
        ),
    )
    if not TESTS_DIR.is_dir():
        return

    for path in TESTS_DIR.rglob("*.py"):
        # Do not rewrite the migration regression fixture itself. Earlier
        # versions changed the literal sample inside this test on every real
        # migration run, making the next pytest invocation fail.
        if path.name == "test_project_layout_migration.py":
            continue
        original = path.read_text(encoding="utf-8")
        updated = original
        for old, new in replacements:
            updated = updated.replace(old, new)
        if updated == original:
            continue
        report.updated.append(str(path))
        if not dry_run:
            path.write_text(updated, encoding="utf-8")

def _legacy_roots() -> tuple[Path, ...]:
    roots = [APP_ROOT]
    if APP_ROOT.parent != APP_ROOT:
        roots.append(APP_ROOT.parent)
    return tuple(roots)


def _migrate_models(source: Path, report: MigrationReport, *, dry_run: bool) -> None:
    if not source.is_dir():
        return
    for item in tuple(source.iterdir()):
        if item in {FARMING_MODELS_DIR, MAPPING_MODELS_DIR}:
            continue
        target_root = _model_target_root(item)
        if target_root is None:
            report.skipped.append(f"unclassified model item: {item}")
            continue
        destination_name = _normalised_model_name(item)
        _move_path(item, target_root / destination_name, report, dry_run=dry_run)


def _model_target_root(item: Path) -> Path | None:
    name = item.name.lower()
    if (
        name.startswith("mapper_")
        or name.startswith("best_model")
        or name in {
            "evaluations.npz",
            "mapper_best",
            "mapper_eval",
            "mapper_checkpoints",
            "mapping",
        }
    ):
        return MAPPING_MODELS_DIR
    if name.startswith("flyff_ppo") or name in {"checkpoints", "farming"}:
        return FARMING_MODELS_DIR
    if name == "archive":
        # The archive folder was introduced by mapper release instructions.
        return MAPPING_MODELS_DIR
    return None


def _normalised_model_name(item: Path) -> str:
    return {
        "mapper_best": "best",
        "mapper_eval": "evaluations",
        "mapper_checkpoints": "checkpoints",
    }.get(item.name.lower(), item.name)


def _migrate_training_logs(
    source: Path,
    report: MigrationReport,
    *,
    dry_run: bool,
) -> None:
    if not source.is_dir():
        return
    for item in tuple(source.iterdir()):
        if item in {FARMING_TRAINING_LOGS_DIR, MAPPING_TRAINING_LOGS_DIR}:
            continue
        name = item.name.lower()
        if name in {"mapper_rl", "mapping"}:
            _merge_tree(
                item,
                MAPPING_TRAINING_LOGS_DIR,
                report,
                dry_run=dry_run,
            )
            if not dry_run and item.exists() and not any(item.iterdir()):
                item.rmdir()
            continue
        if name == "farming":
            _merge_tree(
                item,
                FARMING_TRAINING_LOGS_DIR,
                report,
                dry_run=dry_run,
            )
            if not dry_run and item.exists() and not any(item.iterdir()):
                item.rmdir()
            continue
        # Legacy root logs predate mapper training and belong to farming.
        _move_path(
            item,
            FARMING_TRAINING_LOGS_DIR / item.name,
            report,
            dry_run=dry_run,
        )


def _merge_tree(
    source: Path,
    destination: Path,
    report: MigrationReport,
    *,
    dry_run: bool,
) -> None:
    if not source.is_dir():
        return
    for item in tuple(source.iterdir()):
        _move_path(item, destination / item.name, report, dry_run=dry_run)


def _move_path(
    source: Path,
    destination: Path,
    report: MigrationReport,
    *,
    dry_run: bool,
) -> None:
    if not source.exists():
        return
    if source.resolve() == destination.resolve():
        return

    if source.is_dir():
        if not destination.exists():
            _record_move(source, destination, report, dry_run=dry_run)
            return
        if not destination.is_dir():
            report.conflicts.append(f"{source} -> {destination} (destination is a file)")
            return
        for child in tuple(source.iterdir()):
            _move_path(child, destination / child.name, report, dry_run=dry_run)
        if not dry_run and source.exists() and not any(source.iterdir()):
            source.rmdir()
        return

    if destination.exists():
        if destination.is_file() and filecmp.cmp(source, destination, shallow=False):
            report.skipped.append(f"identical file already present: {destination}")
            if not dry_run:
                source.unlink()
            return
        report.conflicts.append(f"{source} -> {destination} (different file exists)")
        return

    _record_move(source, destination, report, dry_run=dry_run)


def _record_move(
    source: Path,
    destination: Path,
    report: MigrationReport,
    *,
    dry_run: bool,
) -> None:
    report.moved.append(f"{source} -> {destination}")
    if dry_run:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))


def _print_report(report: MigrationReport, *, dry_run: bool) -> None:
    prefix = "Would move" if dry_run else "Moved"
    print(f"{prefix}: {len(report.moved)}")
    for item in report.moved:
        print(f"  {item}")
    if report.updated:
        print(f"Updated moved test paths: {len(report.updated)}")
        for item in report.updated:
            print(f"  {item}")
    if report.skipped:
        print(f"Skipped: {len(report.skipped)}")
        for item in report.skipped:
            print(f"  {item}")
    if report.conflicts:
        print(f"Conflicts: {len(report.conflicts)}")
        for item in report.conflicts:
            print(f"  {item}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move legacy Flyff RL models, logs and tests into the app folder."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without moving anything.",
    )
    args = parser.parse_args()
    report = migrate_project_layout(dry_run=args.dry_run)
    _print_report(report, dry_run=args.dry_run)
    if not report.ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
