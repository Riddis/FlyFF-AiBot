from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import uuid4

import pytest


APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


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
