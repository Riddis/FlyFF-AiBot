from __future__ import annotations

import json
from pathlib import Path

import cv2 as cv


class MinimapAnchorSetup:
    """One-time fixed minimap-arrow center selection."""

    WINDOW_NAME = "Select minimap arrow center"

    def __init__(self, bot) -> None:
        self.bot = bot
        self.config_path = Path(__file__).resolve().parent / "minimap_anchor.json"
        self._selected: tuple[int, int] | None = None
        self._display_scale = 1.0

    def run(self) -> Path:
        frame = self.bot.get_debug_frame()
        if frame is None:
            raise RuntimeError(
                "No game frame is available. Attach the Flyff window first."
            )

        height, width = frame.shape[:2]
        maximum_display_width = 1500
        maximum_display_height = 900
        self._display_scale = min(
            1.0,
            maximum_display_width / width,
            maximum_display_height / height,
        )

        display_width = round(width * self._display_scale)
        display_height = round(height * self._display_scale)

        cv.namedWindow(self.WINDOW_NAME, cv.WINDOW_NORMAL)
        cv.resizeWindow(
            self.WINDOW_NAME,
            display_width,
            display_height,
        )
        cv.setMouseCallback(self.WINDOW_NAME, self._on_mouse)

        while True:
            display = frame.copy()

            cv.rectangle(
                display,
                (8, 8),
                (min(width - 8, 760), 82),
                (0, 0, 0),
                thickness=-1,
            )
            cv.putText(
                display,
                "Click the exact CENTER of the player arrow on the Navigator.",
                (18, 36),
                cv.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
                cv.LINE_AA,
            )
            cv.putText(
                display,
                "ENTER = save   R = retry   ESC = cancel",
                (18, 67),
                cv.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv.LINE_AA,
            )

            if self._selected is not None:
                x, y = self._selected
                cv.drawMarker(
                    display,
                    (x, y),
                    (0, 255, 0),
                    markerType=cv.MARKER_CROSS,
                    markerSize=34,
                    thickness=2,
                )
                half = 20
                cv.rectangle(
                    display,
                    (x - half, y - half),
                    (x + half, y + half),
                    (0, 255, 0),
                    2,
                )
                cv.putText(
                    display,
                    f"Selected ({x}, {y})",
                    (max(0, x - 90), max(25, y - 28)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv.LINE_AA,
                )

            shown = (
                cv.resize(
                    display,
                    (display_width, display_height),
                    interpolation=cv.INTER_AREA,
                )
                if self._display_scale != 1.0
                else display
            )
            cv.imshow(self.WINDOW_NAME, shown)
            key = cv.waitKey(30) & 0xFF

            if key in (13, 10):  # Enter
                if self._selected is None:
                    continue
                break
            if key in (ord("r"), ord("R")):
                self._selected = None
            if key == 27:  # Escape
                cv.destroyWindow(self.WINDOW_NAME)
                raise RuntimeError("Minimap anchor selection cancelled.")

            if (
                cv.getWindowProperty(
                    self.WINDOW_NAME,
                    cv.WND_PROP_VISIBLE,
                )
                < 1
            ):
                raise RuntimeError("Minimap anchor selection cancelled.")

        cv.destroyWindow(self.WINDOW_NAME)

        x, y = self._selected
        crop_size = 41
        config = {
            "version": 1,
            "frame_width": width,
            "frame_height": height,
            "arrow_center_x": x,
            "arrow_center_y": y,
            "crop_size": crop_size,
        }
        self.config_path.write_text(
            json.dumps(config, indent=2),
            encoding="utf-8",
        )
        return self.config_path

    def _on_mouse(
        self,
        event: int,
        x: int,
        y: int,
        flags: int,
        parameter,
    ) -> None:
        del flags, parameter
        if event != cv.EVENT_LBUTTONDOWN:
            return

        frame_x = round(x / self._display_scale)
        frame_y = round(y / self._display_scale)
        self._selected = (frame_x, frame_y)
