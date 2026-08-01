"""Foreground Vision Farm

Farm approach: Using OpenCV it will track the name of the mob.
Currently it's aiming to all lv 150 mobs in Neo Cascada, but it can be extended.
"""

import traceback
from pathlib import Path

from Bot import Bot
from Gui import Gui
from utils.helpers import print_logo

# Instances
gui = Gui("DarkAmber")
bot = Bot()


def main():
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
    print_logo("Flyff FVF")
    main()
