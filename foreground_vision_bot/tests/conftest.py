from __future__ import annotations

import os
import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_FARMING_PARENT = APP_ROOT.parent / "flyff_farming_simulator"
if not (APP_ROOT / "project_paths.py").is_file() or not (APP_ROOT / "mapper").is_dir():
    raise RuntimeError(
        "Tests must live under foreground_vision_bot/tests; "
        f"resolved application root was {APP_ROOT}"
    )

# Remove stale parent-level application paths before importing project modules.
# This prevents an old Flyff RL/mapper package from shadowing the maintained
# foreground_vision_bot/mapper package.
if not (CANONICAL_FARMING_PARENT / "farming" / "observation_contract.py").is_file():
    raise RuntimeError(f"Canonical farming package is missing at {CANONICAL_FARMING_PARENT}")

# BRIDGE B1 — removed in Phase 7
bridge_keys = {
    os.path.normcase(str(APP_ROOT.resolve())),
    os.path.normcase(str(CANONICAL_FARMING_PARENT.resolve())),
}
cleaned: list[str] = []
for entry in sys.path:
    try:
        entry_key = os.path.normcase(str(Path(entry or ".").resolve()))
    except OSError:
        entry_key = os.path.normcase(str(entry))
    if entry_key in bridge_keys:
        continue
    cleaned.append(entry)
sys.path[:] = [str(CANONICAL_FARMING_PARENT), str(APP_ROOT), *cleaned]
