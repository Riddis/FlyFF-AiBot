from __future__ import annotations

import json

from native_farming import NativeFarmingConfig


def test_native_farming_config_loads_user_tunable_values(tmp_path) -> None:
    path = tmp_path / "native_farming.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "dry_run_seconds": 12.5,
                "vision_radius_cells": 42.0,
                "unknown_future_key": "ignored",
            }
        ),
        encoding="utf-8",
    )

    config = NativeFarmingConfig.load(path)

    assert config.dry_run_seconds == 12.5
    assert config.vision_radius_cells == 42.0
    assert config.max_targets == 32
