"""FlyFF AiBot desktop entry point."""

import sys
import traceback
from pathlib import Path

# One directory deep under the repository root (apps/); direct invocation
# (python apps/dev_app.py) sets sys.path[0] to this file's own directory,
# not the repository root, so the root-level GUI/runtime modules below
# need it added explicitly (same bootstrap pattern devtools/native/*.py
# already uses).
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from Bot import Bot  # noqa: E402
from Gui import Gui  # noqa: E402
from utils.helpers import print_logo  # noqa: E402

# Instances
gui = Gui("DarkAmber")
bot = Bot()


def _declare_windows_dpi_awareness() -> None:
    """Declare Per-Monitor-v2 DPI awareness before any Tk window exists.

    Without this declaration, Windows applies its own bitmap compatibility
    scaling to the whole (DPI-unaware) process while Tk's font-driven
    widget sizes still scale to the display's real DPI internally --
    fixed-pixel layout constants (e.g. Gui.py's Column ``size=`` values)
    then no longer match the actual rendered button/frame heights. This
    mismatch, only visible on a real scaled display, produced the
    vertically stretched/clipped sidebar controls seen on the first live
    acceptance run (see MISTAKES.md). No-op on non-Windows platforms;
    best-effort and never blocks startup.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_DPI_AWARE
    except Exception:  # noqa: BLE001 - best-effort DPI declaration only.
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:  # noqa: BLE001
            pass


def main():
    _declare_windows_dpi_awareness()
    gui.init()
    try:
        gui.loop(bot)
    except Exception:  # noqa: BLE001 - persist the top-level GUI failure.
        error_text = traceback.format_exc()
        crash_log = Path(__file__).with_name("gui_crash.log")
        crash_log.write_text(error_text, encoding="utf-8")
        print(error_text)
        try:
            gui.show_error(
                "The application encountered an unexpected error.\n\n"
                f"{error_text}\n"
                f"A copy was saved to:\n{crash_log}"
            )
        except Exception as popup_error:  # noqa: BLE001 - preserve original failure.
            print(f"Could not display the visible error dialog: {popup_error}")
    finally:
        # The normal close event owns shutdown. This is the one fallback for an
        # exception that exits the GUI loop before shutdown completed.
        if gui.controller is not None and not gui.controller.shutdown_finalized:
            results = gui.controller.shutdown(timeout=8.0)
            timed_out = [
                kind.value for kind, stopped in results.items() if not stopped
            ]
            if timed_out:
                print("Shutdown timed out while waiting for: " + ", ".join(timed_out))
        gui.close()


if __name__ == "__main__":
    print_logo("FlyFF AiBot", font="doom")
    main()
