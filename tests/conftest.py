from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1] / "foreground_vision_bot"
sys.path.insert(0, str(PROJECT_DIR))
