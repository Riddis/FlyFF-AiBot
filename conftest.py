from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import uuid4


APP_ROOT = Path(__file__).resolve().parent
CANONICAL_FARMING_PARENT = APP_ROOT.parent / "flyff_farming_simulator"
if not (CANONICAL_FARMING_PARENT / "farming" / "observation_contract.py").is_file():
    raise RuntimeError(f"Canonical farming package is missing at {CANONICAL_FARMING_PARENT}")

# BRIDGE B1 — removed in Phase 7
bridge_keys = {
    os.path.normcase(str(APP_ROOT.resolve())),
    os.path.normcase(str(CANONICAL_FARMING_PARENT.resolve())),
}
remaining_paths = [
    entry
    for entry in sys.path
    if os.path.normcase(str(Path(entry or ".").resolve())) not in bridge_keys
]
sys.path[:] = [str(CANONICAL_FARMING_PARENT), str(APP_ROOT), *remaining_paths]

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Keep pytest temporary trees inside the app with a unique run folder.

    Some Windows installations deny pytest access to the shared
    ``%LOCALAPPDATA%\\Temp\\pytest-of-<user>`` directory.  A per-process local
    base directory avoids both that ACL problem and collisions with stale test
    runs that still have files open.
    """

    if config.option.basetemp:
        return
    temp_root = APP_ROOT / ".pytest_tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    session = f"run-{os.getpid()}-{uuid4().hex[:8]}"
    config.option.basetemp = str(temp_root / session)
