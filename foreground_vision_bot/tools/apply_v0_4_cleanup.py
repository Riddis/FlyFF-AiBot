from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OBSOLETE_FILES = (
    ROOT / "COORDINATE_MAPPER_V0_3.md",
    ROOT / "tools" / "cleanup_legacy_mapping_v0_3.py",
)
CACHE_NAMES = {"__pycache__", ".pytest_cache", ".pytest_tmp"}


def main() -> None:
    removed: list[Path] = []
    for path in OBSOLETE_FILES:
        if path.is_file():
            path.unlink()
            removed.append(path)
    for path in sorted(ROOT.rglob("*"), reverse=True):
        if path.is_dir() and path.name in CACHE_NAMES:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path)
    if removed:
        print("Removed obsolete v0.3 files and local caches:")
        for path in removed:
            print(f"  {path.relative_to(ROOT)}")
    else:
        print("No obsolete v0.3 files or caches were present.")


if __name__ == "__main__":
    main()
