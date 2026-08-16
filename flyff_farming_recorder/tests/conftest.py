from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_FARMING_PARENT = ROOT.parent / "flyff_farming_simulator"
CANONICAL_POSITION_PARENT = ROOT.parent / "foreground_vision_bot"
if not (CANONICAL_FARMING_PARENT / "farming" / "observation_contract.py").is_file():
    raise RuntimeError(f"Canonical farming package is missing at {CANONICAL_FARMING_PARENT}")
if not (CANONICAL_POSITION_PARENT / "position" / "policy.py").is_file():
    raise RuntimeError(f"Canonical position package is missing at {CANONICAL_POSITION_PARENT}")

# BRIDGE B1 — removed in Phase 7
bridge_keys = {
    os.path.normcase(str(ROOT.resolve())),
    os.path.normcase(str(CANONICAL_FARMING_PARENT.resolve())),
    os.path.normcase(str(CANONICAL_POSITION_PARENT.resolve())),
}
remaining_paths = [
    entry
    for entry in sys.path
    if os.path.normcase(str(Path(entry or ".").resolve())) not in bridge_keys
]
# BRIDGE B2 — removed in Phase 7
sys.path[:] = [
    str(CANONICAL_FARMING_PARENT),
    str(CANONICAL_POSITION_PARENT),
    str(ROOT),
    *remaining_paths,
]
