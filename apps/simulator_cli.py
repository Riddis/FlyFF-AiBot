import sys
from pathlib import Path

# One directory deep under the repository root (apps/); direct invocation
# (python apps/simulator_cli.py) sets sys.path[0] to this file's own
# directory, not the repository root, so the simulator package below needs
# it added explicitly.
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from simulator.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
