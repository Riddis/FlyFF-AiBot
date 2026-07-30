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
