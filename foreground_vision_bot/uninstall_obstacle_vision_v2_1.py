from __future__ import annotations

import argparse
from pathlib import Path

BEGIN = "# BEGIN OBSTACLE_VISION_V2_1_DROPIN"
END = "# END OBSTACLE_VISION_V2_1_DROPIN"


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove the obstacle-vision v2.1 hook.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    target = args.project_root.resolve() / "mapper" / "AdaptiveMapper.py"
    text = target.read_text(encoding="utf-8")
    start = text.find(BEGIN)
    finish = text.find(END)
    if start < 0 or finish < 0 or finish < start:
        print("Obstacle Vision v2.1 hook was not found; no changes made.")
        return
    finish += len(END)
    cleaned = (text[:start].rstrip() + "\n" + text[finish:].lstrip()).rstrip() + "\n"
    target.write_text(cleaned, encoding="utf-8")
    print("Removed the Obstacle Vision v2.1 hook. Collected data and models were preserved.")


if __name__ == "__main__":
    main()
