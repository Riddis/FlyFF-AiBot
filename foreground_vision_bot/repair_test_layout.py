from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent
LOCAL_TESTS = APP_ROOT / "tests"
LEGACY_TESTS = APP_ROOT.parent / "tests"
BACKUP_ROOT = APP_ROOT / "migration_backups"

# These narrow regression files were introduced while diagnosing failures that
# are already covered by the canonical suite. Keeping both copies doubled the
# same assertions and, worse, encoded two incompatible project layouts.
REDUNDANT_TEST_FILES = {
    "test_map_logger_schema_regression.py",
    "test_minimap_heading_geometry_regression.py",
    "test_occupancy_grid_metadata_regression.py",
}


@dataclass
class RepairReport:
    source_directories: list[str] = field(default_factory=list)
    collected_files: int = 0
    removed_redundant_files: list[str] = field(default_factory=list)
    patched_files: list[str] = field(default_factory=list)
    backup_path: str | None = None
    installed_path: str | None = None


def _is_link_like(path: Path) -> bool:
    is_junction = getattr(os.path, "isjunction", lambda _path: False)
    return path.is_symlink() or bool(is_junction(path))


def _real_key(path: Path) -> str:
    try:
        return os.path.normcase(str(path.resolve()))
    except OSError:
        return os.path.normcase(str(path.absolute()))


def _source_directories() -> list[Path]:
    sources: list[Path] = []
    seen: set[str] = set()
    # Copy the legacy/full suite first, then let an existing local suite overlay
    # newer files supplied by recent releases.
    for candidate in (LEGACY_TESTS, LOCAL_TESTS):
        if not candidate.exists():
            continue
        key = _real_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        sources.append(candidate)
    return sources


def _copy_suite(sources: list[Path], stage: Path, report: RepairReport) -> None:
    stage.mkdir(parents=True, exist_ok=True)
    for source in sources:
        report.source_directories.append(str(source))
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            relative = path.relative_to(source)
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    report.collected_files = sum(1 for path in stage.rglob("*") if path.is_file())


def _write_canonical_conftest(stage: Path, report: RepairReport) -> None:
    content = '''from __future__ import annotations

import os
import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
if not (APP_ROOT / "project_paths.py").is_file() or not (APP_ROOT / "mapper").is_dir():
    raise RuntimeError(
        "Tests must live under foreground_vision_bot/tests; "
        f"resolved application root was {APP_ROOT}"
    )

# Remove stale parent-level application paths before importing project modules.
# This prevents an old Flyff RL/mapper package from shadowing the maintained
# foreground_vision_bot/mapper package.
app_key = os.path.normcase(str(APP_ROOT.resolve()))
cleaned: list[str] = []
for entry in sys.path:
    try:
        entry_key = os.path.normcase(str(Path(entry or ".").resolve()))
    except OSError:
        entry_key = os.path.normcase(str(entry))
    if entry_key == app_key:
        continue
    cleaned.append(entry)
sys.path[:] = [str(APP_ROOT), *cleaned]
'''
    path = stage / "conftest.py"
    path.write_text(content, encoding="utf-8")
    report.patched_files.append(str(path.name))


def _patch_minimap_test_paths(stage: Path, report: RepairReport) -> None:
    path = stage / "test_minimap_heading.py"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    original = text
    marker = "TEST_APP_ROOT = Path(__file__).resolve().parents[1]\n"
    if marker not in text:
        insertion = "from time import monotonic, perf_counter, sleep\n\n"
        text = text.replace(insertion, insertion + marker + "\n", 1)

    # Handle both compact and line-broken forms used by the historical suite.
    text = re.sub(
        r"Path\(__file__\)(?:\.resolve\(\))?\.parents\[1\]\s*/\s*"
        r"[\"']foreground_vision_bot[\"']",
        "TEST_APP_ROOT",
        text,
    )
    if text != original:
        path.write_text(text, encoding="utf-8")
        report.patched_files.append(str(path.name))


def _remove_redundant_tests(stage: Path, report: RepairReport) -> None:
    for name in sorted(REDUNDANT_TEST_FILES):
        path = stage / name
        if path.exists():
            path.unlink()
            report.removed_redundant_files.append(name)


def _archive_sources(sources: list[Path]) -> Path:
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = BACKUP_ROOT / f"tests_before_v19_{stamp}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for index, source in enumerate(sources):
            label = "legacy_parent" if source == LEGACY_TESTS else f"source_{index}"
            for path in source.rglob("*"):
                if not path.is_file() or "__pycache__" in path.parts:
                    continue
                handle.write(path, Path(label) / path.relative_to(source))
    return archive


def _remove_tree_or_link(path: Path) -> None:
    if not path.exists() and not _is_link_like(path):
        return
    if _is_link_like(path):
        # Path.unlink handles symlinks; Windows directory junctions require rmdir.
        try:
            path.unlink()
        except (IsADirectoryError, PermissionError, OSError):
            os.rmdir(path)
        return
    shutil.rmtree(path)


def repair(*, apply: bool) -> RepairReport:
    report = RepairReport()
    sources = _source_directories()
    if not sources:
        raise FileNotFoundError(
            f"No test suite found at {LOCAL_TESTS} or {LEGACY_TESTS}."
        )

    with tempfile.TemporaryDirectory(prefix="flyff-tests-v19-") as temp:
        stage = Path(temp) / "tests"
        _copy_suite(sources, stage, report)
        _remove_redundant_tests(stage, report)
        _write_canonical_conftest(stage, report)
        _patch_minimap_test_paths(stage, report)

        if not apply:
            return report

        archive = _archive_sources(sources)
        report.backup_path = str(archive)

        # Sources may include a junction from app/tests to ../tests. Remove the
        # local entry first, then the distinct legacy tree, and install one real
        # directory under the application root.
        _remove_tree_or_link(LOCAL_TESTS)
        if LEGACY_TESTS.exists() and _real_key(LEGACY_TESTS) != _real_key(LOCAL_TESTS):
            _remove_tree_or_link(LEGACY_TESTS)
        shutil.copytree(stage, LOCAL_TESTS)
        report.installed_path = str(LOCAL_TESTS)

    return report


def _print_report(report: RepairReport, *, apply: bool) -> None:
    mode = "Applied" if apply else "Dry run"
    print(f"{mode}: consolidated test layout")
    for source in report.source_directories:
        print(f"  source: {source}")
    print(f"  collected files: {report.collected_files}")
    if report.removed_redundant_files:
        print("  removed redundant tests: " + ", ".join(report.removed_redundant_files))
    if report.patched_files:
        print("  canonicalised: " + ", ".join(report.patched_files))
    if report.backup_path:
        print(f"  backup: {report.backup_path}")
    if report.installed_path:
        print(f"  installed: {report.installed_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate the Flyff test suite under foreground_vision_bot/tests."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the repair; without this flag only a dry run is shown",
    )
    args = parser.parse_args()
    report = repair(apply=args.apply)
    _print_report(report, apply=args.apply)


if __name__ == "__main__":
    main()
