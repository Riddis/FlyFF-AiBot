from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .Transforms import floor_corridor_mask, prepare_top_down_crop


@dataclass(frozen=True)
class FloorPrediction:
    available: bool
    obstacle_risk: float | None
    floor_fraction: float | None
    median_distance: float | None
    status: str


class OnlineFloorAppearanceModel:
    """One-class appearance model trained only from motion-confirmed clear floor."""

    VERSION = 1

    def __init__(
        self,
        model_path: Path,
        *,
        crop_fraction: float = 0.72,
        image_size: int = 224,
        minimum_clear_frames: int = 5,
        minimum_pixel_samples: int = 1200,
        samples_per_frame: int = 320,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = self.model_path.with_suffix(".metadata.json")
        self.crop_fraction = float(crop_fraction)
        self.image_size = int(image_size)
        self.minimum_clear_frames = int(minimum_clear_frames)
        self.minimum_pixel_samples = int(minimum_pixel_samples)
        self.samples_per_frame = int(samples_per_frame)
        self.clear_frames = 0
        self.sample_count = 0
        self.mean = np.zeros(4, dtype=np.float64)
        self.m2 = np.zeros(4, dtype=np.float64)
        self._dirty = False
        self._rng = np.random.default_rng(2031)
        self._load()

    @property
    def ready(self) -> bool:
        return (
            self.clear_frames >= self.minimum_clear_frames
            and self.sample_count >= self.minimum_pixel_samples
        )

    @property
    def status(self) -> str:
        state = "ready" if self.ready else "learning"
        return (
            f"{state} clear_frames={self.clear_frames}/{self.minimum_clear_frames} "
            f"pixel_samples={self.sample_count}/{self.minimum_pixel_samples}"
        )

    def observe_clear(self, frame: np.ndarray, *, heading_deg: float) -> bool:
        image = prepare_top_down_crop(
            frame,
            heading_deg=heading_deg,
            crop_fraction=self.crop_fraction,
            image_size=self.image_size,
        )
        features = self._features(image)
        mask = floor_corridor_mask(self.image_size, seed_only=True).astype(bool)
        pixels = features[mask]
        if pixels.size == 0:
            return False
        take = min(self.samples_per_frame, len(pixels))
        indices = self._rng.choice(len(pixels), size=take, replace=False)
        selected = pixels[indices]
        for sample in selected:
            self.sample_count += 1
            delta = sample - self.mean
            self.mean += delta / self.sample_count
            self.m2 += delta * (sample - self.mean)
        self.clear_frames += 1
        self._dirty = True
        if self.clear_frames % 5 == 0:
            self.save()
        return True

    def predict(self, frame: np.ndarray, *, heading_deg: float) -> FloorPrediction:
        if not self.ready:
            return FloorPrediction(False, None, None, None, self.status)
        image = prepare_top_down_crop(
            frame,
            heading_deg=heading_deg,
            crop_fraction=self.crop_fraction,
            image_size=self.image_size,
        )
        features = self._features(image)
        mask = floor_corridor_mask(self.image_size).astype(bool)
        pixels = features[mask]
        variance = self.m2 / max(1, self.sample_count - 1)
        # Floors are repetitive, so guard against a near-zero learned variance.
        scale = np.sqrt(np.maximum(variance, np.asarray([18.0, 18.0, 18.0, 28.0])))
        distance = np.sqrt(np.mean(((pixels - self.mean) / scale) ** 2, axis=1))
        floor_fraction = float(np.mean(distance <= 2.8))
        median_distance = float(np.median(distance))
        # Penalise both widespread mismatch and a strongly shifted median.
        mismatch = 1.0 - floor_fraction
        median_penalty = float(np.clip((median_distance - 1.4) / 2.4, 0.0, 1.0))
        risk = float(np.clip(0.78 * mismatch + 0.22 * median_penalty, 0.0, 1.0))
        return FloorPrediction(True, risk, floor_fraction, median_distance, "ready")

    def save(self) -> None:
        if not self._dirty and self.model_path.is_file():
            return
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.model_path,
            version=np.asarray([self.VERSION], dtype=np.int32),
            clear_frames=np.asarray([self.clear_frames], dtype=np.int64),
            sample_count=np.asarray([self.sample_count], dtype=np.int64),
            mean=self.mean,
            m2=self.m2,
            crop_fraction=np.asarray([self.crop_fraction], dtype=np.float64),
            image_size=np.asarray([self.image_size], dtype=np.int32),
        )
        self.metadata_path.write_text(
            json.dumps(
                {
                    "version": self.VERSION,
                    "clear_frames": self.clear_frames,
                    "sample_count": self.sample_count,
                    "ready": self.ready,
                    "crop_fraction": self.crop_fraction,
                    "image_size": self.image_size,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self._dirty = False

    def _load(self) -> None:
        if not self.model_path.is_file():
            return
        try:
            data = np.load(self.model_path)
            version = int(np.asarray(data["version"]).ravel()[0])
            if version != self.VERSION:
                return
            self.clear_frames = int(np.asarray(data["clear_frames"]).ravel()[0])
            self.sample_count = int(np.asarray(data["sample_count"]).ravel()[0])
            self.mean = np.asarray(data["mean"], dtype=np.float64).reshape(4)
            self.m2 = np.asarray(data["m2"], dtype=np.float64).reshape(4)
        except (OSError, KeyError, ValueError):
            self.clear_frames = 0
            self.sample_count = 0
            self.mean = np.zeros(4, dtype=np.float64)
            self.m2 = np.zeros(4, dtype=np.float64)

    @staticmethod
    def _features(image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float64)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.clip(cv2.magnitude(gx, gy), 0.0, 128.0)
        return np.dstack((lab, gradient)).astype(np.float64, copy=False)
