from __future__ import annotations

from pathlib import Path


def test_persistent_map_management_controls_are_retained() -> None:
    gui_source = (Path(__file__).resolve().parents[1] / "bot" / "Gui.py").read_text(
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


def test_removed_legacy_buttons_have_no_live_gui_references() -> None:
    gui_source = (Path(__file__).resolve().parents[1] / "bot" / "Gui.py").read_text(
        encoding="utf-8"
    )

    assert "Native Dry Run (No Learning)" not in gui_source
    assert "Calibrate Minimap (optional)" not in gui_source
    assert "Map Area (Automatic)" not in gui_source
    for removed_key in (
        "-DRY_RUN-",
        "-START_MAPPER-",
        "-SET_MINIMAP_ANCHOR-",
        "-BOT_THRESHOLD_OPTIONS-",
        "-MOBS_KILL_GOAL-",
        "-FIGHT_TIME_LIMIT_SEC-",
        "-DELAY_TO_CHECK_MOB_STILL_ALIVE_SEC-",
    ):
        assert removed_key not in gui_source
    assert "Trace Map While I Drive" in gui_source

def test_gui_exposes_managed_native_health_and_recovery_commands() -> None:
    gui_source = (Path(__file__).resolve().parents[1] / "bot" / "Gui.py").read_text(
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
    gui_source = (Path(__file__).resolve().parents[1] / "bot" / "Gui.py").read_text(
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


def test_gui_has_session_statistics_and_configurable_eva_key() -> None:
    gui_source = (Path(__file__).resolve().parents[1] / "bot" / "Gui.py").read_text(
        encoding="utf-8"
    )

    for key in (
        "-SESSION_LENGTH-",
        "-SESSION_KILLS-HR-",
        "-SESSION_PENYA-HR-",
        "-SESSION_TOTAL-KILLS-",
        "-SESSION_TOTAL-PERIN-",
        "-EVA-HOTKEY-",
    ):
        assert key in gui_source
    assert "Session Statistics:" in gui_source
    assert "Penya / hour" in gui_source
    assert "Total Perin" in gui_source


def test_mapper_uses_selected_eva_hotkey_instead_of_hardcoded_f1() -> None:
    mapper_source = (
        Path(__file__).resolve().parents[1] / "mapper" / "CoordinateMapper.py"
    ).read_text(encoding="utf-8")

    assert 'self.bot.config.get("eva_hotkey", "F1")' in mapper_source
    assert 'press_key(VKEY["F1"]' not in mapper_source


def test_statistics_span_the_top_and_live_map_reflows_without_bot_vision() -> None:
    gui_source = (Path(__file__).resolve().parents[1] / "bot" / "Gui.py").read_text(
        encoding="utf-8"
    )

    assert "[session_statistics]," in gui_source
    assert "[controls, visuals]," in gui_source
    assert gui_source.index("[session_statistics],") < gui_source.index(
        "[controls, visuals],"
    )
    assert "[live_map, session_statistics]" not in gui_source
    assert 'key="-VISUALS-COLUMN-"' in gui_source
    assert 'method_name = "unhide_row" if enabled else "hide_row"' in gui_source
    assert "def __resize_live_map" in gui_source
    assert "target_width=920, max_height=240" in gui_source


def test_map_edit_buttons_share_one_row_and_ui_redetection_is_exposed() -> None:
    gui_source = (Path(__file__).resolve().parents[1] / "bot" / "Gui.py").read_text(
        encoding="utf-8"
    )

    first = gui_source.index('sg.Button("Add Map", key="-ADD_MAP-")')
    second = gui_source.index('sg.Button("Edit Map Mobs", key="-EDIT_MAP_MOBS-")')
    third = gui_source.index('sg.Button("Edit Map Cells", key="-EDIT_MAP_CELLS-")')
    assert first < second < third
    assert third - first < 300
    assert 'key="-REDETECT-UI-"' in gui_source
    assert "bot.redetect_ui_elements()" in gui_source
    assert "Timer to convert penya to perins" not in gui_source
    assert "-CONVERT_PENYA_TO_PERINS_TIMER_MIN-" not in gui_source

def test_idle_map_preview_is_rendered_from_current_grid_configuration() -> None:
    controller_source = (
        Path(__file__).resolve().parents[1] / "bot" / "runtime_controller.py"
    ).read_text(encoding="utf-8")

    assert 'radius = load_mapper_config().local_map_radius_cells' in controller_source
    assert 'grid.render_dashboard(local_radius_cells=radius)' in controller_source
    assert 'preview_path = MapCatalog().preview_path(map_name)' not in controller_source


def test_live_map_defaults_to_fifty_cell_radius() -> None:
    project = Path(__file__).resolve().parents[1]
    mapper_source = (project / "mapper" / "CoordinateMapper.py").read_text(
        encoding="utf-8"
    )
    mapper_config = (project / "mapper" / "coordinate_mapper.json").read_text(
        encoding="utf-8"
    )
    bot_source = (project / "bot" / "Bot.py").read_text(encoding="utf-8")

    assert "local_map_radius_cells: int = 50" in mapper_source
    assert '"local_map_radius_cells": 50' in mapper_config
    assert '"native_monster_local_radius_cells": 50' in bot_source
