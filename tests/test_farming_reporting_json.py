from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from farming.reporting import atomic_write_json, to_json_safe


def test_atomic_json_converts_nested_numpy_terminal_observation(tmp_path: Path) -> None:
    destination = tmp_path / "session.json"
    payload = {
        "latest_info": {
            "terminal_observation": np.asarray([1.25, -0.5], dtype=np.float32),
            "native_kills": np.int64(7),
            "done": np.bool_(True),
            "nested": (np.float64(2.5), Path("models/policy.zip")),
        }
    }

    atomic_write_json(destination, payload)

    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert saved["latest_info"]["terminal_observation"] == [1.25, -0.5]
    assert saved["latest_info"]["native_kills"] == 7
    assert saved["latest_info"]["done"] is True
    assert saved["latest_info"]["nested"] == [2.5, "models/policy.zip"]


def test_json_safe_conversion_handles_recursive_diagnostics() -> None:
    payload: dict[str, object] = {}
    payload["self"] = payload

    assert to_json_safe(payload) == {"self": "<recursive-reference>"}
