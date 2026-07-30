from __future__ import annotations

import argparse
import py_compile
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PATCH_VERSION = "v0.7.0.7"
SOURCE_FILES = (
    Path("native_farming.py"),
    Path("native_farming.json"),
    Path("libs/V0707TeleportSafety.py"),
)
TEST_FILE = Path("tests/test_v0707_teleport_session_regressions.py")
FOCUSED_TESTS = (
    "tests/test_v0707_teleport_session_regressions.py",
    "tests/test_v0706_camera_focus_regressions.py",
    "tests/test_v0705_unified_executor_only_regressions.py",
    "tests/test_v0704_exact_map_adapter_regressions.py",
    "tests/test_v0700_unified_farming_regressions.py",
    "tests/test_v0673_eva_movement_regressions.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the v0.7.0.7 teleport-zone penalty and clean farm-session "
            "shutdown patch."
        )
    )
    parser.add_argument("--project", required=True, help="foreground_vision_bot root")
    parser.add_argument("--run-tests", action="store_true")
    return parser.parse_args()


def _verify_project(project: Path) -> None:
    required = (
        project / "native_farming.py",
        project / "native_farming.json",
        project / "libs/V0700UnifiedFarming.py",
        project / "libs/CameraDiscoverySweep.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Project is missing required files: " + ", ".join(missing))

    farming = (project / "native_farming.py").read_text(encoding="utf-8")
    unified = (project / "libs/V0700UnifiedFarming.py").read_text(encoding="utf-8")
    sweep = (project / "libs/CameraDiscoverySweep.py").read_text(encoding="utf-8")
    markers = {
        "v0.7 unified runtime": "install_v0700_unified_farming()" in farming,
        "executor-only mode": "load_policy=False" in farming,
        "mapped observation adapter": "local_map_available" in unified,
        "camera autofocus": "focus_timeout_seconds" in sweep,
    }
    absent = [name for name, present in markers.items() if not present]
    if absent:
        raise RuntimeError(
            "The project is missing required earlier patches: " + ", ".join(absent)
        )


def _copy_patch_files(project: Path, patch_root: Path) -> None:
    for relative in (*SOURCE_FILES, TEST_FILE):
        source = patch_root / "files" / relative
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _verify_result(project: Path) -> None:
    farming = (project / "native_farming.py").read_text(encoding="utf-8")
    safety = (project / "libs/V0707TeleportSafety.py").read_text(encoding="utf-8")
    config = (project / "native_farming.json").read_text(encoding="utf-8")

    required_farming = (
        "install_v0707_teleport_safety()",
        "env.configure_teleport_safety(",
        "class SessionEndCallback",
        "FARM SESSION END DETECTED",
        "_write_training_session_report(",
        "Creating a new unified farming PPO policy.",
    )
    required_safety = (
        "DEFAULT_TRIGGER_PENALTY = 50.0",
        "farm_time_expired_or_external_teleport",
        "forbidden_teleport_zone",
        "v0700._MapAdapter.local_grid = _local_policy_grid",
        "return observation, 0.0, False, False, info",
    )
    required_config = (
        '"unified_control_interval_seconds": 0.2',
        '"teleport_trigger_penalty": 50.0',
        '"teleport_jump_threshold_cells": 25.0',
        '"session_report_dir": "training_logs/farming/native_sessions"',
    )
    missing = [marker for marker in required_farming if marker not in farming]
    missing += [marker for marker in required_safety if marker not in safety]
    missing += [marker for marker in required_config if marker not in config]
    if missing:
        raise RuntimeError("Patch verification failed; missing: " + ", ".join(missing))

    for relative in (Path("native_farming.py"), Path("libs/V0707TeleportSafety.py")):
        py_compile.compile(str(project / relative), doraise=True)


def _run_tests(project: Path) -> None:
    existing = [test for test in FOCUSED_TESTS if (project / test).is_file()]
    command = [sys.executable, "-B", "-m", "pytest", "-q", *existing]
    print("Running focused tests:", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=project, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Focused tests failed with exit code {completed.returncode}")


def main() -> int:
    args = _parse_args()
    project = Path(args.project).expanduser().resolve()
    patch_root = Path(__file__).resolve().parent
    backup_root = project / ".patch_backups" / (
        "v0707_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    created: list[Path] = []
    backed_up: list[Path] = []

    try:
        _verify_project(project)
        for relative in (*SOURCE_FILES, TEST_FILE):
            target = project / relative
            if target.exists():
                backup = backup_root / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup)
                backed_up.append(relative)
            else:
                created.append(relative)

        _copy_patch_files(project, patch_root)
        _verify_result(project)
        if args.run_tests:
            _run_tests(project)
    except Exception as error:
        for relative in backed_up:
            backup = backup_root / relative
            target = project / relative
            if backup.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup, target)
        for relative in created:
            target = project / relative
            if target.exists():
                target.unlink()
        print(f"PATCH FAILED: {error}", file=sys.stderr)
        print("The previous source files were restored.", file=sys.stderr)
        return 1

    print(f"{PATCH_VERSION} installed successfully.")
    print(f"Backup: {backup_root}")
    print(
        "Mapped teleport cells now have distinct observations and severe reward "
        "penalties. Teleports/session expiry stop movement, save the model and "
        "write a clean session report instead of crashing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
