"""FlyFF AiBot desktop entry point."""

import traceback
import os
import sys
from pathlib import Path

# BRIDGE B1 — removed in Phase 7
_APP_ROOT = Path(__file__).resolve().parent
_CANONICAL_FARMING_PARENT = _APP_ROOT.parent / "flyff_farming_simulator"
if not (_CANONICAL_FARMING_PARENT / "farming" / "observation_contract.py").is_file():
    raise RuntimeError(f"Canonical farming package is missing at {_CANONICAL_FARMING_PARENT}")
_bridge_keys = {
    os.path.normcase(str(_APP_ROOT.resolve())),
    os.path.normcase(str(_CANONICAL_FARMING_PARENT.resolve())),
}
_remaining_paths = [
    entry
    for entry in sys.path
    if os.path.normcase(str(Path(entry or ".").resolve())) not in _bridge_keys
]
sys.path[:] = [str(_CANONICAL_FARMING_PARENT), str(_APP_ROOT), *_remaining_paths]

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
    print_logo("FlyFF AiBot", font="doom")
    main()
