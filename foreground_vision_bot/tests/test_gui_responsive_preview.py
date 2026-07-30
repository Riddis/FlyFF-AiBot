from __future__ import annotations

from types import SimpleNamespace

from Gui import Gui


class _Element:
    def __init__(self, width: int, height: int) -> None:
        self.Widget = SimpleNamespace(
            winfo_width=lambda: width,
            winfo_height=lambda: height,
        )


def test_fit_panel_uses_current_bot_vision_widget_size() -> None:
    gui = Gui.__new__(Gui)
    gui.frame_resolutions = {"Fit panel": None, "960x540": (960, 540)}
    gui.window = {
        "-DEBUG_IMAGE-": _Element(1234, 678),
        "-VISION_FRAME-": _Element(1300, 760),
    }

    assert gui._Gui__preview_target_size("Fit panel") == (1234, 678)
    assert gui._Gui__preview_target_size("960x540") == (960, 540)
