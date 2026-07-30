from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = (
    ROOT / "COORDINATE_MAPPER_V0_4.md",
    ROOT / "PATCH_V0_4_INSTRUCTIONS.md",
    ROOT / ".pytest_cache",
)


def main() -> None:
    for target in TARGETS:
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
            print(f"removed directory: {target.relative_to(ROOT)}")
        elif target.exists():
            target.unlink()
            print(f"removed file: {target.relative_to(ROOT)}")
    print("v0.5 cleanup complete; mapper/maps was not touched.")


if __name__ == "__main__":
    main()
