from __future__ import annotations

import json
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from mapper.MinimapHeading import MinimapHeadingDetector, signed_angle_delta


def _frame_with_arrow(image: np.ndarray) -> np.ndarray:
    anchor = json.loads(
        (Path(__file__).resolve().parents[1] / "mapper" / "minimap_anchor.json").read_text(
            encoding="utf-8"
        )
    )
    frame = np.zeros(
        (anchor["frame_height"], anchor["frame_width"], 3), dtype=np.uint8
    )
    x = anchor["arrow_center_x"]
    y = anchor["arrow_center_y"]
    height, width = image.shape
    x0 = x - width // 2
    y0 = y - height // 2
    frame[y0 : y0 + height, x0 : x0 + width] = image[:, :, None]
    return frame


def test_direction_assets_are_valid_geometry_without_template_fallback() -> None:
    detector = MinimapHeadingDetector()
    for path, expected in detector._template_sources:
        image = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
        assert image is not None
        reading = detector.read(_frame_with_arrow(image))
        assert reading is not None
        assert abs(signed_angle_delta(reading.angle_deg, expected)) <= 1.0


@pytest.mark.parametrize(("name", "anchor", "rotation"), [("s", 180.0, 20.0), ("sw", 225.0, 10.0)])
def test_rotated_south_assets_keep_valid_padded_geometry(
    name: str,
    anchor: float,
    rotation: float,
) -> None:
    detector = MinimapHeadingDetector()
    path = next(path for path, _ in detector._template_sources if path.stem == f"map_arrow_{name}")
    image = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    assert image is not None
    canvas = np.zeros((41, 41), dtype=np.uint8)
    y0 = (41 - image.shape[0]) // 2
    x0 = (41 - image.shape[1]) // 2
    canvas[y0 : y0 + image.shape[0], x0 : x0 + image.shape[1]] = image
    matrix = cv.getRotationMatrix2D((20.0, 20.0), -rotation, 1.0)
    rotated = cv.warpAffine(canvas, matrix, (41, 41), flags=cv.INTER_LINEAR)

    reading = detector.read(_frame_with_arrow(rotated))

    assert reading is not None
    assert reading.motion_angle_deg is not None
    assert abs(signed_angle_delta(reading.angle_deg, anchor + rotation)) <= 8.0


def test_normal_geometry_read_does_not_call_template_or_contour(monkeypatch) -> None:
    detector = MinimapHeadingDetector()
    path = next(path for path, heading in detector._template_sources if heading == 180.0)
    image = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    assert image is not None

    def fail(*_args, **_kwargs):
        raise AssertionError("legacy template/contour path was called")

    monkeypatch.setattr(detector, "_score_heading", fail)
    monkeypatch.setattr(detector, "_contour_heading", fail)

    reading = detector.read(_frame_with_arrow(image))

    assert reading is not None
    assert reading.source == "calibrated_grayscale_geometry"


def test_direction_assets_survive_a_raised_dark_background() -> None:
    detector = MinimapHeadingDetector()
    for path, expected in detector._template_sources:
        image = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
        assert image is not None
        frame = _frame_with_arrow(image)
        frame[:] = np.maximum(frame, 8)
        # Reinsert the arrow as contrast above the raised background.
        anchor = json.loads(
            (Path(__file__).resolve().parents[1] / "mapper" / "minimap_anchor.json").read_text(
                encoding="utf-8"
            )
        )
        x = anchor["arrow_center_x"]
        y = anchor["arrow_center_y"]
        height, width = image.shape
        x0 = x - width // 2
        y0 = y - height // 2
        raised = np.clip(image.astype(np.int16) + 8, 0, 255).astype(np.uint8)
        frame[y0 : y0 + height, x0 : x0 + width] = raised[:, :, None]

        reading = detector.read(frame)

        assert reading is not None
        assert abs(signed_angle_delta(reading.angle_deg, expected)) <= 1.0
