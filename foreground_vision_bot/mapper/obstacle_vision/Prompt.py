from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def request_label(
    image: np.ndarray,
    *,
    motion_label: str,
    suggested_label: str,
    reason: str,
    floor_risk: float | None,
    timeout_seconds: float = 1800.0,
) -> str | None:
    """Open the standalone reviewer without touching the application's GUI thread."""
    script = Path(__file__).resolve().parents[2] / "prompt_obstacle_label.py"
    if not script.is_file():
        return None
    with tempfile.TemporaryDirectory(prefix="flyff-obstacle-review-") as temp:
        image_path = Path(temp) / "review.png"
        if not cv2.imwrite(str(image_path), image):
            return None
        command = [
            sys.executable,
            str(script),
            "--image",
            str(image_path),
            "--motion-label",
            str(motion_label),
            "--suggested-label",
            str(suggested_label),
            "--reason",
            str(reason),
        ]
        if floor_risk is not None:
            command.extend(["--floor-risk", f"{float(floor_risk):.6f}"])
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=max(1.0, float(timeout_seconds)),
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        for line in reversed(completed.stdout.splitlines()):
            result = line.strip().lower()
            if result in {"clear", "blocked", "ignore", "skip"}:
                return None if result == "skip" else result
        return None
