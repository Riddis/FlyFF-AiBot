from __future__ import annotations

import argparse
import py_compile
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PATCH_VERSION = "v0.7.0.6"
SOURCE_FILES = (
    Path("libs/HumanKeyboard.py"),
    Path("libs/CameraDiscoverySweep.py"),
)
TEST_FILE = Path("tests/test_v0706_camera_focus_regressions.py")
FOCUSED_TESTS = (
    "tests/test_v0706_camera_focus_regressions.py",
    "tests/test_v0705_unified_executor_only_regressions.py",
    "tests/test_v0704_exact_map_adapter_regressions.py",
    "tests/test_v0700_unified_farming_regressions.py",
    "tests/test_v0673_eva_movement_regressions.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the v0.7.0.6 FlyFF autofocus/camera-sweep startup patch."
    )
    parser.add_argument("--project", required=True, help="foreground_vision_bot root")
    parser.add_argument("--run-tests", action="store_true")
    return parser.parse_args()


def _verify_project(project: Path) -> None:
    required = (
        project / "native_farming.py",
        project / "libs/HumanKeyboard.py",
        project / "libs/CameraDiscoverySweep.py",
        project / "libs/V0700UnifiedFarming.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Project is missing required files: " + ", ".join(missing))

    farming = (project / "native_farming.py").read_text(encoding="utf-8")
    if "load_policy=False" not in farming:
        raise RuntimeError(
            "v0.7.0.5 executor-only mode was not detected. Apply v0.7.0.5 first."
        )
    unified = (project / "libs/V0700UnifiedFarming.py").read_text(encoding="utf-8")
    if "local_map_available" not in unified:
        raise RuntimeError(
            "The exact-repo unified map adapter was not detected. Apply v0.7.0.4 first."
        )


def _copy_patch_files(project: Path, patch_root: Path) -> None:
    for relative in (*SOURCE_FILES, TEST_FILE):
        source = patch_root / "files" / relative
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _verify_result(project: Path) -> None:
    keyboard = (project / "libs/HumanKeyboard.py").read_text(encoding="utf-8")
    sweep = (project / "libs/CameraDiscoverySweep.py").read_text(encoding="utf-8")
    required_keyboard = (
        "def focus_target_window(self) -> bool:",
        "SetForegroundWindow",
        "AttachThreadInput",
    )
    required_sweep = (
        "auto_focus_on_start: bool = True",
        "focus_timeout_seconds: float = 8.0",
        "def _wait_for_target_focus(self, keyboard) -> bool:",
        "Click the FlyFF window now",
    )
    missing = [marker for marker in required_keyboard if marker not in keyboard]
    missing += [marker for marker in required_sweep if marker not in sweep]
    if missing:
        raise RuntimeError("Patch verification failed; missing: " + ", ".join(missing))
    if "Focus the FlyFF window before starting the camera discovery sweep." in sweep:
        raise RuntimeError("The old instant-focus failure path is still present.")

    for relative in SOURCE_FILES:
        py_compile.compile(str(project / relative), doraise=True)


def _run_tests(project: Path) -> None:
    existing = [test for test in FOCUSED_TESTS if (project / test).is_file()]
    command = [sys.executable, "-B", "-m", "pytest", "-q", *existing]
    print("Running focused tests:", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=project, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Focused tests failed with exit code {completed.returncode}"
        )


def main() -> int:
    args = _parse_args()
    project = Path(args.project).expanduser().resolve()
    patch_root = Path(__file__).resolve().parent
    backup_root = project / ".patch_backups" / (
        "v0706_" + datetime.now().strftime("%Y%m%d_%H%M%S")
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
        "Camera discovery now attempts to focus FlyFF automatically and waits "
        "up to eight seconds for manual focus before failing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
