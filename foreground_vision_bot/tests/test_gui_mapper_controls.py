from __future__ import annotations

from pathlib import Path


def test_persistent_map_management_controls_are_retained() -> None:
    gui_source = (Path(__file__).resolve().parents[1] / "Gui.py").read_text(
        encoding="utf-8"
    )

    for required_key in (
        "-ADD_MAP-",
        "-EDIT_MAP_MOBS-",
        "-RESET_MAP-",
        "-DELETE_MAP-",
    ):
        assert required_key in gui_source

    for required_handler in (
        "def __add_map_popup",
        "def __edit_map_mobs_popup",
        "def __reset_selected_map",
        "def __delete_selected_map",
    ):
        assert required_handler in gui_source


def test_only_obsolete_mapper_calibration_controls_are_removed() -> None:
    gui_source = (Path(__file__).resolve().parents[1] / "Gui.py").read_text(
        encoding="utf-8"
    )

    assert "-CALIBRATE_MAPPER-" not in gui_source
    assert "-DEBUG_CALIBRATE_MAPPER-" not in gui_source
    assert "-START_MAPPER-" in gui_source
    assert "-START_MANUAL_MAPPER-" in gui_source
    assert "Trace Map While I Drive" in gui_source
    assert "-SET_MINIMAP_ANCHOR-" in gui_source
    assert "Calibrate Minimap (optional)" in gui_source
    assert (
        "from mapper.MinimapAnchorSetup import MinimapAnchorSetup"
        in gui_source
    )


def test_gui_exposes_managed_native_health_and_recovery_commands() -> None:
    gui_source = (Path(__file__).resolve().parents[1] / "Gui.py").read_text(
        encoding="utf-8"
    )

    assert 'key="-NATIVE_HEALTH-"' in gui_source
    assert 'key="-RECOVER_POINTERS-"' in gui_source
    assert "self.controller.start_native_diagnostic(" in gui_source
    assert "self.controller.stop_native_diagnostic()" in gui_source
    assert "Player HP is read automatically from the status panel." not in gui_source
    assert "-POINTER-CURRENT-HP-" not in gui_source
    assert "-POINTER-MAX-HP-" not in gui_source


def test_gui_exposes_three_functional_bot_vision_toggles() -> None:
    gui_source = (Path(__file__).resolve().parents[1] / "Gui.py").read_text(
        encoding="utf-8"
    )

    assert 'key="-SHOW_FRAMES-"' in gui_source
    assert 'key="-SHOW_UI_ELEMENTS-"' in gui_source
    assert 'key="-SHOW_MOB_MARKERS-"' in gui_source
    assert "Show bot vision" in gui_source
    assert "Show detected UI elements" in gui_source
    assert "Show mobs markers" in gui_source
    assert 'key="-SHOW_MATCHES_TEXT-"' not in gui_source
    assert 'key="-SHOW_BOXES-"' not in gui_source
