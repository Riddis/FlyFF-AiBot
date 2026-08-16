from __future__ import annotations

import os
import sys
from pathlib import Path

# BRIDGE B1 — removed in Phase 7
_APP_ROOT = Path(__file__).resolve().parent
_CANONICAL_FARMING_PARENT = _APP_ROOT.parent / "flyff_farming_simulator"
_CANONICAL_POSITION_PARENT = _APP_ROOT.parent / "foreground_vision_bot"
if not (_CANONICAL_FARMING_PARENT / "farming" / "observation_contract.py").is_file():
    raise RuntimeError(f"Canonical farming package is missing at {_CANONICAL_FARMING_PARENT}")
if not (_CANONICAL_POSITION_PARENT / "position" / "policy.py").is_file():
    raise RuntimeError(f"Canonical position package is missing at {_CANONICAL_POSITION_PARENT}")
_bridge_keys = {
    os.path.normcase(str(_APP_ROOT.resolve())),
    os.path.normcase(str(_CANONICAL_FARMING_PARENT.resolve())),
    os.path.normcase(str(_CANONICAL_POSITION_PARENT.resolve())),
}
_remaining_paths = [
    entry
    for entry in sys.path
    if os.path.normcase(str(Path(entry or ".").resolve())) not in _bridge_keys
]
# BRIDGE B2 — removed in Phase 7
sys.path[:] = [
    str(_CANONICAL_FARMING_PARENT),
    str(_CANONICAL_POSITION_PARENT),
    str(_APP_ROOT),
    *_remaining_paths,
]

from recorder.gui import run_gui


if __name__ == "__main__":
    run_gui()
