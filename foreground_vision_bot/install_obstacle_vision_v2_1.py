from __future__ import annotations

import argparse
import py_compile
import shutil
from pathlib import Path

BEGIN = "# BEGIN OBSTACLE_VISION_V2_1_DROPIN"
END = "# END OBSTACLE_VISION_V2_1_DROPIN"
HOOK = f'''\n\n{BEGIN}\nfrom .obstacle_vision.integration import install_obstacle_vision as _install_obstacle_vision_v21\n\n_install_obstacle_vision_v21(AdaptiveMapper)\ndel _install_obstacle_vision_v21\n{END}\n'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install obstacle-vision v2.1 as a reversible AdaptiveMapper hook."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="foreground_vision_bot directory (defaults to this script's directory)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    target = root / "mapper" / "AdaptiveMapper.py"
    package = root / "mapper" / "obstacle_vision" / "integration.py"
    if not target.is_file():
        raise SystemExit(f"Missing {target}. Extract the ZIP into foreground_vision_bot.")
    if not package.is_file():
        raise SystemExit(f"Missing {package}. Extract the complete ZIP first.")

    text = target.read_text(encoding="utf-8")
    if BEGIN in text:
        print("Obstacle Vision v2.1 is already installed; no changes made.")
        return
    if "class AdaptiveMapper" not in text:
        raise SystemExit("AdaptiveMapper.py does not contain class AdaptiveMapper; stopped safely.")

    backup = target.with_name(target.name + ".before_obstacle_vision_v2_1")
    if not backup.exists():
        shutil.copy2(target, backup)
    target.write_text(text.rstrip() + HOOK, encoding="utf-8")
    try:
        py_compile.compile(str(target), doraise=True)
        py_compile.compile(str(package), doraise=True)
    except py_compile.PyCompileError:
        shutil.copy2(backup, target)
        raise

    version_note = (
        "Detected the expected v1.9.7 baseline."
        if "1.9.7-contact-adjacent-camera-fallback" in text
        else "Baseline version string differs from v1.9.7; the generic hook compiled, but run tests before live use."
    )
    print("Installed Obstacle Vision v2.1 drop-in.")
    print(version_note)
    print(f"Backup: {backup}")
    print("Next: python -B -m pytest -q tests")


if __name__ == "__main__":
    main()
