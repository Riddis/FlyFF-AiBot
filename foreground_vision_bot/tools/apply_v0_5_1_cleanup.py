from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

for relative in (
    "COORDINATE_MAPPER_V0_5.md",
    "PATCH_V0_5_INSTRUCTIONS.md",
    ".pytest_cache",
    ".pytest_tmp",
):
    path = ROOT / relative
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)

for cache in ROOT.rglob("__pycache__"):
    shutil.rmtree(cache, ignore_errors=True)

print("v0.5.1 cleanup complete. Existing mapper/maps data was preserved.")
