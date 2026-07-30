from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    files = (
        ROOT / "COORDINATE_MAPPER_V0_5_2.md",
        ROOT / "COORDINATE_MAPPER_V0_5_2_1.md",
        ROOT / "PATCH_V0_5_2_INSTRUCTIONS.md",
    )
    directories = (
        ROOT / ".pytest_cache",
        ROOT / ".pytest_tmp",
        ROOT / "__pycache__",
        ROOT / "mapper" / "__pycache__",
        ROOT / "tests" / "__pycache__",
        ROOT / "tools" / "__pycache__",
    )
    for path in files:
        if path.exists():
            path.unlink()
            print(f"Removed {path.relative_to(ROOT)}")
    for path in directories:
        if path.exists():
            shutil.rmtree(path)
            print(f"Removed {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
