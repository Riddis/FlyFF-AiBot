"""Print every world-model-eligible archive under the given directories, one
absolute path per line.

world-model eligibility (a dynamically validated or explicitly attested
presence field) is independent of movement classification, so this scans
every given directory rather than assuming eligible archives live in one
particular folder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simulator.recording_discovery import discover_world_model_eligible  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()

    for path in discover_world_model_eligible(args.directories):
        print(str(path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
