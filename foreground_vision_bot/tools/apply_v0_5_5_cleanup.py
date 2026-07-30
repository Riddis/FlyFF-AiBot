from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    removed = 0
    for cache in ROOT.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
        removed += 1
    for cache in ROOT.rglob(".pytest_cache"):
        shutil.rmtree(cache, ignore_errors=True)
        removed += 1
    print(f"v0.5.5 cleanup complete; removed {removed} cache directorie(s).")


if __name__ == "__main__":
    main()
