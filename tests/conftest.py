from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# scratchpad/*.py modules are imported bare (e.g. `import
# scratchpad_historical_reproduction_guard`) by several tests, matching
# how the scratchpad scripts import each other as siblings. They moved
# from repository root to scratchpad/ in the 2026-08-21 repository
# cleanup; this keeps the bare-name import working without rewriting
# every import site to a dotted scratchpad.* form.
SCRATCHPAD_DIR = ROOT / "scratchpad"
if str(SCRATCHPAD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCHPAD_DIR))

# bot/*.py (Bot.py, Gui.py, runtime_controller.py, recording_sink.py,
# preview_service.py) moved from repository root to bot/ in the
# 2026-08-21 repository cleanup; same bare-import-preservation reasoning
# as SCRATCHPAD_DIR above.
BOT_DIR = ROOT / "bot"
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))
