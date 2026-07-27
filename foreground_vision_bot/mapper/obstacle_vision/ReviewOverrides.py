from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def load_review_overrides(dataset_root: Path) -> dict[str, str]:
    path = Path(dataset_root) / "review_overrides.jsonl"
    overrides: dict[str, str] = {}
    if not path.is_file():
        return overrides
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = str(payload.get("sample_id", ""))
            label = str(payload.get("label", ""))
            if sample_id and label in {"clear", "blocked", "ignore"}:
                overrides[sample_id] = label
    return overrides


def append_review_override(dataset_root: Path, *, sample_id: str, label: str) -> Path:
    if label not in {"clear", "blocked", "ignore"}:
        raise ValueError(f"Unsupported review label: {label}")
    path = Path(dataset_root) / "review_overrides.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_id": str(sample_id),
        "label": label,
        "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return path
