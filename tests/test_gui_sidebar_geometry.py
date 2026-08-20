"""Rendered-geometry regression test for the live-observed sidebar
overflow (see MISTAKES.md, "[2026-08-20]" entries): Phase 10 added a
four-column, 313+-row artifact Table (col_widths=[10, 30, 18, 24],
expand_x=True) directly inside the fixed-width (335px), vertically-
scroll-only ``-MAIN_COLUMN-`` Column. The Table's real requested width
(~740px on the reference machine) widened the whole scrollable Column's
inner frame far past its 335px canvas viewport; every sibling
expand_x=True sidebar control then filled to that oversized inner
width and was clipped/pushed off-screen by the canvas, which cannot
scroll horizontally (vertical_scroll_only=True).

The Development Tools panel that contained this table (and everything
else it added to the sidebar) has since been removed entirely --
docs/decisions/0007-dev-bot-first-is-not-an-ide.md -- not replaced by a
smaller launcher or an artifact-inventory dialog. This test protects
both the geometry invariant and the panel's continued absence.

This is verified by actually constructing the real Tk window
(Gui().init()) and reading its real, rendered widget geometry --
tests/test_gui_devtools_wiring.py's fake-window tests cannot catch a
regression like this (Phase 10's own tests didn't, for exactly that
reason). No bot, no FlyFF attach, no native reader, no control -- pure
local Tk geometry, same as every other offline test in this suite.

One real Tk window is constructed for the whole module (not per test):
repeatedly creating/destroying tk.Tk()-backed windows within a single
pytest process is flaky on Windows (observed directly: a per-test
fixture intermittently raised _tkinter.TclError "Can't find a usable
init.tcl" on later parametrized cases) -- a real, but unrelated,
Tcl-interpreter-lifecycle issue, not a defect in the geometry being
measured."""

from __future__ import annotations

import pytest

from Gui import Gui

# Generous tolerance, not a fragile exact-pixel assertion (font metrics,
# theme, and minor per-machine rendering differences all shift this a
# little) -- the invariant being protected is "the sidebar's real
# content roughly fits its declared viewport," not a specific pixel
# count. The pre-fix regression this guards against overflowed by
# ~450px (779px content in a 335px canvas); this bar is nowhere near
# that so a real regression will still fail it clearly.
MAX_INNER_WIDTH_OVERFLOW_PX = 150
MAX_BUTTON_WIDTH_OVERFLOW_PX = 150

ACTION_BUTTON_KEYS = (
    "-ATTACH_WINDOW-",
    "-VALIDATE_DATA-",
    "-START_BOT-",
    "-RUN_AGENT-",
    "-NATIVE_HEALTH-",
    "-RECOVER_POINTERS-",
    "-STOP_BOT-",
    "-RECORDING-START-",
    "-RECORDING-STOP-",
)

REMOVED_DEVTOOLS_KEYS = (
    "-DEVTOOLS-COMMAND-",
    "-DEVTOOLS-ARGS-",
    "-DEVTOOLS-LAUNCH-",
    "-DEVTOOLS-CANCEL-",
    "-DEVTOOLS-STATUS-",
    "-DEVTOOLS-ARTIFACTS-SUMMARY-",
    "-DEVTOOLS-ARTIFACTS-OPEN-",
    "-DEVTOOLS-ARTIFACTS-TABLE-",
    "-DEVTOOLS-ARTIFACTS-REFRESH-",
)


@pytest.fixture(scope="module")
def rendered_gui():
    gui = Gui("DarkAmber")
    gui.init()
    gui.window.read(timeout=300)
    gui.window.refresh()
    try:
        yield gui
    finally:
        gui.window.close()


def test_sidebar_inner_frame_width_roughly_matches_its_viewport(rendered_gui: Gui) -> None:
    column = rendered_gui.window["-MAIN_COLUMN-"]
    canvas_width = column.Widget.canvas.winfo_reqwidth()
    inner_width = column.Widget.TKFrame.winfo_reqwidth()

    assert inner_width - canvas_width <= MAX_INNER_WIDTH_OVERFLOW_PX, (
        f"sidebar inner content ({inner_width}px) overflows its "
        f"{canvas_width}px vertical-scroll-only viewport by "
        f"{inner_width - canvas_width}px -- some child forces the "
        "whole scrollable Column wider than intended (see MISTAKES.md's "
        "artifact-Table entry for the exact prior mechanism)"
    )


def test_action_buttons_stay_within_the_visible_sidebar(rendered_gui: Gui) -> None:
    column = rendered_gui.window["-MAIN_COLUMN-"]
    canvas_width = column.Widget.canvas.winfo_reqwidth()

    failures = []
    for key in ACTION_BUTTON_KEYS:
        button_width = rendered_gui.window[key].Widget.winfo_width()
        overflow = button_width - canvas_width
        if overflow > MAX_BUTTON_WIDTH_OVERFLOW_PX:
            failures.append(f"{key}: width={button_width}px overflow={overflow}px")

    assert not failures, (
        "these buttons extend more than "
        f"{MAX_BUTTON_WIDTH_OVERFLOW_PX}px past the {canvas_width}px "
        "sidebar viewport -- their centered labels would be partially "
        "or fully invisible, reproducing the live-observed clipped/"
        "blank button regression: " + "; ".join(failures)
    )


def test_start_training_and_run_trained_agent_are_separate_full_width_rows(
    rendered_gui: Gui,
) -> None:
    """Explicit requirement: these two do not need to share one row --
    separate full-width rows guarantee both stay readable regardless of
    sidebar width (docs/architecture/SYSTEM_OVERVIEW.md section 3b)."""
    start_training = rendered_gui.window["-START_BOT-"].Widget
    run_agent = rendered_gui.window["-RUN_AGENT-"].Widget
    # winfo_y() is relative to each button's own immediate row frame (a
    # different widget per button here), so it reports a similar small
    # offset for both regardless of row -- winfo_rooty() (absolute
    # screen position) is what actually distinguishes separate rows.
    assert start_training.winfo_rooty() != run_agent.winfo_rooty()


def test_development_tools_panel_is_fully_removed(rendered_gui: Gui) -> None:
    """The panel that caused the regression is gone entirely -- not
    replaced by a smaller launcher, popup, or artifact-inventory dialog
    (docs/decisions/0007-dev-bot-first-is-not-an-ide.md)."""
    present = [key for key in REMOVED_DEVTOOLS_KEYS if key in rendered_gui.window.AllKeysDict]
    assert not present, f"Development Tools keys still present: {present}"


def test_recording_controls_are_present(rendered_gui: Gui) -> None:
    """The compact recording section replaces the removed Development
    Tools panel's spot in the sidebar (docs/PROJECT_GOALS.md section 6)
    -- Start/Stop plus a status line, nothing larger."""
    for key in ("-RECORDING-START-", "-RECORDING-STOP-", "-RECORDING-STATUS-"):
        assert key in rendered_gui.window.AllKeysDict


def test_no_recording_metadata_popup_contract_remains(rendered_gui: Gui) -> None:
    """Forward correction (MISTAKES.md): an earlier version required a
    protocol-ID/hypothesis/controller/data-use-role/player-HP popup
    before recording could start. The user explicitly rejected this --
    none of its keys may exist anywhere in the main window, and the
    button label/tooltip must not imply a questionnaire follows."""
    popup_keys = (
        "-POPUP-PROTOCOL-",
        "-POPUP-HYPOTHESIS-",
        "-POPUP-CONTROLLER-",
        "-POPUP-DATA-USE-",
        "-POPUP-PLAYER-HP-",
        "-RECORDING-PLAYER-FULL-HP-",
    )
    present = [key for key in popup_keys if key in rendered_gui.window.AllKeysDict]
    assert not present, f"Recording metadata popup keys still present: {present}"
    assert not hasattr(rendered_gui, "_Gui__start_controlled_recording_popup")
    assert not hasattr(rendered_gui, "_Gui__cached_player_full_hp")
