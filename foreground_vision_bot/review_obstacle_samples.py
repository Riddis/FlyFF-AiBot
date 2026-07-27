from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile

import cv2
from pathlib import Path

from mapper.obstacle_vision.ReviewOverrides import append_review_override, load_review_overrides
from mapper.obstacle_vision.Transforms import build_review_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Review queued obstacle samples.")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/obstacle_vision/raw"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    overrides = load_review_overrides(args.dataset)
    prompt = Path(__file__).resolve().parent / "prompt_obstacle_label.py"
    pending: list[dict[str, object]] = []
    for manifest in sorted(args.dataset.glob("*/manifest.jsonl")):
        with manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sample_id = str(payload.get("sample_id", ""))
                if not sample_id or sample_id in overrides:
                    continue
                if bool(payload.get("review_requested")) or payload.get("label") == "ignore":
                    payload["_dataset_root"] = str(args.dataset)
                    pending.append(payload)

    reviewed = 0
    for payload in pending:
        pre_path = args.dataset / str(payload["pre_path"])
        source = cv2.imread(str(pre_path), cv2.IMREAD_COLOR)
        if source is None:
            continue
        highlighted = build_review_image(
            source,
            heading_deg=0.0,
            crop_fraction=1.0,
            image_size=448,
            title="Queued sample: classify highlighted path",
        )
        temporary = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temporary.close()
        review_path = Path(temporary.name)
        cv2.imwrite(str(review_path), highlighted)
        command = [
            sys.executable,
            str(prompt),
            "--image",
            str(review_path),
            "--motion-label",
            str(payload.get("label", "unknown")),
            "--suggested-label",
            "ignore",
            "--reason",
            str(payload.get("reason", "queued review")),
        ]
        risk = payload.get("floor_risk")
        if risk is not None:
            command.extend(["--floor-risk", str(risk)])
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
        finally:
            review_path.unlink(missing_ok=True)
        result = next(
            (line.strip() for line in reversed(completed.stdout.splitlines()) if line.strip() in {"clear", "blocked", "ignore", "skip"}),
            "skip",
        )
        if result == "skip":
            continue
        append_review_override(
            args.dataset,
            sample_id=str(payload["sample_id"]),
            label=result,
        )
        reviewed += 1
        if args.limit > 0 and reviewed >= args.limit:
            break
    print(f"reviewed={reviewed} remaining={max(0, len(pending) - reviewed)}")


if __name__ == "__main__":
    main()
