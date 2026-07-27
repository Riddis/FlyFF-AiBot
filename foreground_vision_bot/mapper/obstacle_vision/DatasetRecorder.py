from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from .Transforms import hamming_distance, perceptual_hash, prepare_top_down_crop


class ObstacleSampleLabel(str, Enum):
    CLEAR = "clear"
    BLOCKED = "blocked"
    IGNORE = "ignore"


@dataclass(frozen=True)
class ObstacleSample:
    sample_id: str
    run_id: str
    label: ObstacleSampleLabel
    pre_path: Path
    post_path: Path
    retained: bool
    reason: str


class ObstacleDatasetRecorder:
    """Save heading-normalised weak-supervision samples with clear-frame dedupe."""

    def __init__(
        self,
        dataset_root: Path,
        *,
        run_id: str,
        map_name: str,
        enabled: bool = True,
        crop_fraction: float = 0.72,
        image_size: int = 224,
        jpeg_quality: int = 90,
        deduplicate_clear: bool = True,
        minimum_clear_hash_distance: int = 7,
        keep_clear_every: int = 40,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.run_id = str(run_id)
        self.map_name = str(map_name)
        self.enabled = bool(enabled)
        self.crop_fraction = float(crop_fraction)
        self.image_size = int(image_size)
        self.jpeg_quality = int(jpeg_quality)
        self.deduplicate_clear = bool(deduplicate_clear)
        self.minimum_clear_hash_distance = int(minimum_clear_hash_distance)
        self.keep_clear_every = int(keep_clear_every)
        self.run_dir = self.dataset_root / self.run_id
        self.manifest_path = self.run_dir / "manifest.jsonl"
        self.hash_path = self.dataset_root / "clear_hashes.jsonl"
        self._clear_hashes = self._load_clear_hashes()
        self._clear_seen = 0
        self._manifest = None
        if self.enabled:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            for label in ObstacleSampleLabel:
                (self.run_dir / label.value).mkdir(parents=True, exist_ok=True)
            self._manifest = self.manifest_path.open("a", encoding="utf-8")

    def record(
        self,
        *,
        step: int,
        before_frame: np.ndarray,
        after_frame: np.ndarray,
        heading_deg: float,
        label: ObstacleSampleLabel,
        confidence: float,
        reason: str,
        label_source: str = "motion",
        review_requested: bool = False,
        floor_risk: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ObstacleSample | None:
        if not self.enabled or self._manifest is None:
            return None

        pre = prepare_top_down_crop(
            before_frame,
            heading_deg=heading_deg,
            crop_fraction=self.crop_fraction,
            image_size=self.image_size,
        )
        post = prepare_top_down_crop(
            after_frame,
            heading_deg=heading_deg,
            crop_fraction=self.crop_fraction,
            image_size=self.image_size,
        )

        retained = True
        image_hash: int | None = None
        if label is ObstacleSampleLabel.CLEAR:
            self._clear_seen += 1
            image_hash = perceptual_hash(pre)
            heartbeat = self._clear_seen % self.keep_clear_every == 0
            duplicate = any(
                hamming_distance(image_hash, previous) < self.minimum_clear_hash_distance
                for previous in self._clear_hashes[-512:]
            )
            if self.deduplicate_clear and duplicate and not heartbeat and not review_requested:
                retained = False

        if not retained:
            return ObstacleSample(
                sample_id="",
                run_id=self.run_id,
                label=label,
                pre_path=Path(),
                post_path=Path(),
                retained=False,
                reason="deduplicated_clear",
            )

        sample_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            + "-"
            + uuid4().hex[:8]
        )
        label_dir = self.run_dir / label.value
        pre_path = label_dir / f"{sample_id}_pre.jpg"
        post_path = label_dir / f"{sample_id}_post.jpg"
        params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        if not cv2.imwrite(str(pre_path), pre, params):
            raise OSError(f"Could not write obstacle image: {pre_path}")
        if not cv2.imwrite(str(post_path), post, params):
            raise OSError(f"Could not write obstacle image: {post_path}")

        payload = {
            "sample_id": sample_id,
            "run_id": self.run_id,
            "map_name": self.map_name,
            "step": int(step),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "label": label.value,
            "label_source": str(label_source),
            "confidence": float(confidence),
            "reason": str(reason),
            "review_requested": bool(review_requested),
            "heading_deg": float(heading_deg),
            "floor_risk": None if floor_risk is None else float(floor_risk),
            "pre_path": str(pre_path.relative_to(self.dataset_root)),
            "post_path": str(post_path.relative_to(self.dataset_root)),
            "metadata": dict(metadata or {}),
        }
        self._manifest.write(json.dumps(payload, sort_keys=True) + "\n")
        self._manifest.flush()

        if image_hash is not None:
            self._clear_hashes.append(image_hash)
            self.hash_path.parent.mkdir(parents=True, exist_ok=True)
            with self.hash_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {"sample_id": sample_id, "hash": f"{image_hash:016x}"},
                        sort_keys=True,
                    )
                    + "\n"
                )

        return ObstacleSample(
            sample_id=sample_id,
            run_id=self.run_id,
            label=label,
            pre_path=pre_path,
            post_path=post_path,
            retained=True,
            reason=str(reason),
        )

    def close(self) -> None:
        if self._manifest is not None:
            self._manifest.close()
            self._manifest = None

    def _load_clear_hashes(self) -> list[int]:
        hashes: list[int] = []
        if not self.hash_path.is_file():
            return hashes
        with self.hash_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    payload = json.loads(line)
                    hashes.append(int(str(payload["hash"]), 16))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
        return hashes
