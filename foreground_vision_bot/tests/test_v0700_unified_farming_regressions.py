from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v0700_installs_after_previous_runtime_layers() -> None:
    source = (ROOT / "native_farming.py").read_text(encoding="utf-8")
    assert "install_v0700_unified_farming()" in source
    assert source.index("install_v0673_fixes()") < source.index("install_v0700_unified_farming()")


def test_v0700_has_one_four_action_policy() -> None:
    source = (ROOT / "libs" / "V0700UnifiedFarming.py").read_text(encoding="utf-8")
    assert '"RUN_FORWARD"' in source
    assert '"RUN_FORWARD_LEFT"' in source
    assert '"RUN_FORWARD_RIGHT"' in source
    assert '"CAST_EVA"' in source
    assert "env.action_space = spaces.Discrete(len(UNIFIED_ACTION_NAMES))" in source
    assert "self.navigator.cast_eva()" in source
    assert "_execute_movement(self, movement_action)" in source


def test_v0700_map_is_observation_not_controller() -> None:
    source = (ROOT / "libs" / "V0700UnifiedFarming.py").read_text(encoding="utf-8")
    assert "LOCAL_GRID_SIDE" in source
    assert "normalized_position" in source
    assert "direct_path_state" in source
    assert "navigate_toward_cell" not in source
    assert "target_blacklist" not in source
    assert "opposite-steering" not in source


def test_v0700_uses_native_kills_and_ocr_only_as_diagnostic() -> None:
    source = (ROOT / "libs" / "V0700UnifiedFarming.py").read_text(encoding="utf-8")
    assert "kill_delta = int(native_delta)" in source
    assert "ocr_delta = int(v0672._validated_ocr_delta" in source
    assert 'components["kill"] = kill_delta' in source


def test_v0700_dry_run_reports_direct_control() -> None:
    source = (ROOT / "native_farming.py").read_text(encoding="utf-8")
    assert "held={navigation.get('held_action', '--')}" in source
    assert "contact={navigation.get('contact', False)}" in source
    assert "direct_clear={info.get('direct_clear_fraction', '--')}" in source
    assert "eva_resume={info.get('eva_resume_action', '--')}" in source


def test_v0702_dry_run_never_forwards_legacy_target_actions() -> None:
    source = (ROOT / "native_farming.py").read_text(encoding="utf-8")
    chooser = "action = int(env.choose_unified_dry_run_action())"
    step = "env.step(action)"
    assert chooser in source
    assert source.index(chooser) < source.index(step)


def test_v0702_systems_check_chooser_only_returns_unified_actions() -> None:
    source = (ROOT / "libs" / "V0700UnifiedFarming.py").read_text(encoding="utf-8")
    assert "def _choose_unified_dry_run_action" in source
    assert "return 3" in source
    assert "return (0, 1, 0, 2, 0)[phase]" in source
    assert "choose_unified_dry_run_action = _choose_unified_dry_run_action" in source
