from __future__ import annotations

import cv2
import numpy as np


def ensure_bgr(frame: np.ndarray) -> np.ndarray:
    """Return an 8-bit three-channel BGR image."""
    image = np.asarray(frame)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Unsupported frame shape: {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def prepare_top_down_crop(
    frame: np.ndarray,
    *,
    heading_deg: float,
    crop_fraction: float = 0.72,
    image_size: int = 224,
) -> np.ndarray:
    """Centre on the character and rotate so attempted forward points upward."""
    if not 0.35 <= crop_fraction <= 1.0:
        raise ValueError("crop_fraction must be between 0.35 and 1.0")
    if image_size < 64:
        raise ValueError("image_size must be at least 64 pixels")

    bgr = ensure_bgr(frame)
    height, width = bgr.shape[:2]
    side = max(64, int(round(min(height, width) * crop_fraction)))
    side = min(side, height, width)
    center_x = width // 2
    center_y = height // 2
    left = max(0, min(width - side, center_x - side // 2))
    top = max(0, min(height - side, center_y - side // 2))
    crop = bgr[top : top + side, left : left + side]

    angle = float(heading_deg) % 360.0
    matrix = cv2.getRotationMatrix2D((side / 2.0, side / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(
        crop,
        matrix,
        (side, side),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return cv2.resize(rotated, (image_size, image_size), interpolation=cv2.INTER_AREA)


def floor_corridor_mask(image_size: int, *, seed_only: bool = False) -> np.ndarray:
    """Return the forward trapezoid used for floor learning and prediction."""
    if image_size < 32:
        raise ValueError("image_size must be at least 32")
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    cx = image_size // 2
    if seed_only:
        points = np.asarray(
            [
                (int(cx - image_size * 0.10), int(image_size * 0.50)),
                (int(cx + image_size * 0.10), int(image_size * 0.50)),
                (int(cx + image_size * 0.18), int(image_size * 0.78)),
                (int(cx - image_size * 0.18), int(image_size * 0.78)),
            ],
            dtype=np.int32,
        )
    else:
        points = np.asarray(
            [
                (int(cx - image_size * 0.10), int(image_size * 0.16)),
                (int(cx + image_size * 0.10), int(image_size * 0.16)),
                (int(cx + image_size * 0.27), int(image_size * 0.78)),
                (int(cx - image_size * 0.27), int(image_size * 0.78)),
            ],
            dtype=np.int32,
        )
    cv2.fillConvexPoly(mask, points, 1)
    return mask


def build_review_image(
    frame: np.ndarray,
    *,
    heading_deg: float,
    crop_fraction: float = 0.72,
    image_size: int = 448,
    title: str | None = None,
) -> np.ndarray:
    result = prepare_top_down_crop(
        frame,
        heading_deg=heading_deg,
        crop_fraction=crop_fraction,
        image_size=image_size,
    )
    mask = floor_corridor_mask(image_size).astype(bool)
    tint = np.zeros_like(result)
    tint[..., 1] = 210
    blended = cv2.addWeighted(result, 0.76, tint, 0.24, 0.0)
    result[mask] = blended[mask]
    contours, _hierarchy = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(result, contours, -1, (0, 255, 255), 2)
    cv2.arrowedLine(
        result,
        (image_size // 2, int(image_size * 0.55)),
        (image_size // 2, int(image_size * 0.12)),
        (255, 255, 255),
        3,
        tipLength=0.08,
    )
    if title:
        cv2.rectangle(result, (0, 0), (image_size, 36), (0, 0, 0), thickness=-1)
        cv2.putText(
            result,
            str(title)[:72],
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return result


def perceptual_hash(image: np.ndarray) -> int:
    """64-bit difference hash used to suppress repetitive clear screenshots."""
    gray = cv2.cvtColor(ensure_bgr(image), cv2.COLOR_BGR2GRAY)
    tiny = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = tiny[:, 1:] > tiny[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bool(bit))
    return value


def hamming_distance(left: int, right: int) -> int:
    return int((int(left) ^ int(right)).bit_count())
