"""FlyFF AiBot desktop entry point."""

import sys
import traceback
from pathlib import Path

# One directory deep under the repository root (apps/); direct invocation
# (python apps/dev_app.py) sets sys.path[0] to this file's own directory,
# not the repository root, so the top-level packages below need it added
# explicitly (same bootstrap pattern devtools/native/*.py already uses).
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from bot.Bot import Bot  # noqa: E402
from bot.Gui import Gui  # noqa: E402
from libs.helpers import print_logo  # noqa: E402

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
    print_logo("FlyFF AiBot", font="doom")
    main()
