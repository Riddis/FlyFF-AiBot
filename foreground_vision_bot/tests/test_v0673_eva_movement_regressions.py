from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v0673_runtime_layer_is_installed_after_v0672() -> None:
    source = (ROOT / "native_farming.py").read_text(encoding="utf-8")
    assert source.index("install_v0672_fixes()") < source.index("install_v0673_fixes()")


def test_v0673_preserves_and_reasserts_movement_during_eva() -> None:
    source = (ROOT / "libs" / "V0673EvaMovementFix.py").read_text(encoding="utf-8")
    assert '_temporarily_replace(self, "stop", suppressed_release' in source
    assert "self.executor.execute(resume_action)" in source
    assert "if _cancelled_or_disabled(self):" in source
    assert "original_stop(self)" in source


def test_v0673_dry_run_reports_eva_resume_action() -> None:
    source = (ROOT / "native_farming.py").read_text(encoding="utf-8")
    assert "eva_resume={info.get('eva_resume_action', '--')}" in source
