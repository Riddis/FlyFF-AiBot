from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from mapper.obstacle_vision.ReviewOverrides import load_review_overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise obstacle-vision collection.")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/obstacle_vision/raw"))
    args = parser.parse_args()
    overrides = load_review_overrides(args.dataset)
    raw: Counter[str] = Counter()
    effective: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    runs: set[str] = set()
    queued = 0
    retained_clear = 0
    for manifest in sorted(args.dataset.glob("*/manifest.jsonl")):
        runs.add(manifest.parent.name)
        with manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sample_id = str(payload.get("sample_id", ""))
                label = str(payload.get("label", "unknown"))
                raw[label] += 1
                effective_label = overrides.get(sample_id, label)
                effective[effective_label] += 1
                sources[str(payload.get("label_source", "unknown"))] += 1
                queued += int(bool(payload.get("review_requested")) and sample_id not in overrides)
                retained_clear += int(label == "clear")

    model_meta = Path("models/mapping/floor_appearance.metadata.json")
    floor_status = "missing"
    if model_meta.is_file():
        try:
            metadata = json.loads(model_meta.read_text(encoding="utf-8"))
            floor_status = (
                f"ready={bool(metadata.get('ready'))} "
                f"clear_frames={metadata.get('clear_frames', 0)} "
                f"pixel_samples={metadata.get('sample_count', 0)}"
            )
        except (OSError, json.JSONDecodeError):
            floor_status = "unreadable"

    clear_count = effective["clear"]
    blocked_count = effective["blocked"]
    print(f"runs={len(runs)} retained_samples={sum(raw.values())}")
    print(f"raw_labels={dict(raw)}")
    print(f"effective_labels={dict(effective)}")
    print(f"label_sources={dict(sources)}")
    print(f"queued_for_review={queued} reviewed={len(overrides)} novel_clear_images={retained_clear}")
    print(f"floor_model={floor_status}")
    print(f"cnn_training_ready={clear_count >= 100 and blocked_count >= 30}")


if __name__ == "__main__":
    main()
