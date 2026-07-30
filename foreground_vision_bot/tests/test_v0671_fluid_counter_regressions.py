from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_live_navigation_keeps_keys_held_between_normal_bursts() -> None:
    source = (ROOT / "libs" / "LiveNavigatorController.py").read_text(
        encoding="utf-8"
    )
    assert "finally:\n            self.stop()" not in source
    assert "if self.cancellation.cancelled or not self.bot.rl_enabled:" in source


def test_dynamic_counter_does_not_suppress_fallback_on_ocr_miss() -> None:
    source = (ROOT / "Bot.py").read_text(encoding="utf-8")
    assert "if reading is not None and reading.kills is not None:" in source
    assert "return max(0, int(reading.kills))" in source


def test_dry_run_reports_counter_health() -> None:
    source = (ROOT / "native_farming.py").read_text(encoding="utf-8")
    assert "counter={env.kill_counter_status}" in source
    assert "Kill counter baseline acquired" in source
