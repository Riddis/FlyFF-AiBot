from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v0672_runtime_layer_is_installed() -> None:
    source = (ROOT / "native_farming.py").read_text(encoding="utf-8")
    assert "install_v0672_fixes()" in source


def test_v0672_uses_native_kills_and_rejects_ocr_outliers() -> None:
    source = (ROOT / "libs" / "V0672NativeFarmingFixes.py").read_text(
        encoding="utf-8"
    )
    assert "kill_delta = int(native_delta)" in source
    assert "if delta > maximum:" in source
    assert "preserve the last accepted baseline" in source


def test_v0672_latches_targets_and_gates_recovery() -> None:
    source = (ROOT / "libs" / "V0672NativeFarmingFixes.py").read_text(
        encoding="utf-8"
    )
    assert "_v0672_active_target_goal" in source
    assert "ACTIVE_TARGET_TIMEOUT_SECONDS" in source
    assert "RECOVERY_COOLDOWN_SECONDS" in source
    assert "self.executor.execute(NavigatorAction.RUN_FORWARD)" in source
