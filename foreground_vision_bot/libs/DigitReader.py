from pathlib import Path

import cv2 as cv
import numpy as np


class DigitReader:
    def __init__(
        self,
        digits_dir: Path,
        threshold: float = 0.85,
    ) -> None:
        self.threshold = threshold
        self.templates = self._load_templates(digits_dir)
        self.bracket_template = self._load_bracket_template(digits_dir)
        self._segmented_templates = {
            digit: self._normalise_mask(template >= 150)
            for digit, template in self.templates.items()
        }

    @staticmethod
    def _load_templates(digits_dir: Path) -> dict[str, np.ndarray]:
        templates: dict[str, np.ndarray] = {}

        for digit in range(10):
            path = digits_dir / f"{digit}.png"

            template = cv.imread(
                str(path),
                cv.IMREAD_GRAYSCALE,
            )

            if template is None:
                raise FileNotFoundError(f"Could not load digit template: {path}")

            templates[str(digit)] = template

        return templates

    @staticmethod
    def _load_bracket_template(
        digits_dir: Path,
    ) -> np.ndarray:
        path = digits_dir / "bracket.png"

        template = cv.imread(
            str(path),
            cv.IMREAD_GRAYSCALE,
        )

        if template is None:
            raise FileNotFoundError(f"Could not load bracket template: {path}")

        return template

    def read_number(
        self,
        image: np.ndarray,
    ) -> int | None:
        """
        Read a sequence of digits from an image.

        The input should be cropped tightly around the number.
        """
        if image is None or image.size == 0:
            return None

        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        matches: list[tuple[int, int, float, str]] = []

        for digit, template in self.templates.items():
            template_h, template_w = template.shape[:2]

            if gray.shape[0] < template_h or gray.shape[1] < template_w:
                continue

            result = cv.matchTemplate(
                gray,
                template,
                cv.TM_CCOEFF_NORMED,
            )

            locations = np.where(result >= self.threshold)

            for y, x in zip(*locations):
                score = float(result[y, x])

                matches.append(
                    (
                        int(x),
                        int(x + template_w),
                        score,
                        digit,
                    )
                )

        filtered = self._remove_overlapping_matches(matches)

        if not filtered:
            return None

        bracket_x = self._find_bracket_x(gray)

        if bracket_x is not None:
            filtered = [match for match in filtered if match[0] < bracket_x]

        if not filtered:
            return None

        filtered.sort(key=lambda match: match[0])

        # Commas are ignored because there is no comma template.
        # Example: "1,234 (+5)" becomes "1234".
        number_text = "".join(match[3] for match in filtered)

        return int(number_text)

    @staticmethod
    def _normalise_mask(mask: np.ndarray) -> np.ndarray:
        rows, columns = np.where(mask)
        if rows.size == 0 or columns.size == 0:
            return np.zeros((24, 18), dtype=bool)
        glyph = mask[
            int(rows.min()) : int(rows.max()) + 1,
            int(columns.min()) : int(columns.max()) + 1,
        ].astype(np.uint8)
        return cv.resize(
            glyph,
            (18, 24),
            interpolation=cv.INTER_NEAREST,
        ).astype(bool)

    def read_segmented_number(
        self,
        image: np.ndarray,
        *,
        bright_threshold: int = 170,
        minimum_score: float = 0.78,
        minimum_margin: float = 0.02,
        maximum_digits: int = 8,
    ) -> int | None:
        """Read isolated bright FlyFF glyphs from a tightly bounded field.

        This handles the small outlined status-panel font, whose white glyph
        cores share the existing digit-template shapes but not their colours.
        Large connected UI backgrounds and incomplete glyphs are rejected.
        """

        if image is None or image.size == 0:
            return None
        if image.ndim == 2:
            bright = image >= int(bright_threshold)
        elif image.ndim == 3 and image.shape[2] in (3, 4):
            bright = np.all(image[:, :, :3] >= int(bright_threshold), axis=2)
        else:
            return None

        count, labels, statistics, _centroids = cv.connectedComponentsWithStats(
            bright.astype(np.uint8)
        )
        components: list[tuple[int, str]] = []
        for label in range(1, count):
            left, _top, width, height, area = (
                int(value) for value in statistics[label]
            )
            if not (2 <= width <= 8 and 7 <= height <= 11 and 8 <= area <= 40):
                continue
            candidate = self._normalise_mask(labels == label)
            scores = sorted(
                (
                    (float(np.mean(candidate == template)), digit)
                    for digit, template in self._segmented_templates.items()
                ),
                reverse=True,
            )
            best_score, best_digit = scores[0]
            second_score = scores[1][0]
            if (
                best_score < float(minimum_score)
                or best_score - second_score < float(minimum_margin)
            ):
                return None
            components.append((left, best_digit))

        components.sort()
        if not components or len(components) > int(maximum_digits):
            return None
        return int("".join(digit for _left, digit in components))

    def _find_bracket_x(
        self,
        gray: np.ndarray,
    ) -> int | None:
        template = self.bracket_template
        template_h, template_w = template.shape[:2]

        if gray.shape[0] < template_h or gray.shape[1] < template_w:
            return None

        result = cv.matchTemplate(
            gray,
            template,
            cv.TM_CCOEFF_NORMED,
        )

        locations = np.where(result >= self.threshold)

        if locations[1].size == 0:
            return None

        # The first "(" starts the bonus section, such as "(+1)".
        # Ignore all digit matches to its right.
        return int(np.min(locations[1]))

    @staticmethod
    def _remove_overlapping_matches(
        matches: list[tuple[int, int, float, str]],
    ) -> list[tuple[int, int, float, str]]:
        """
        Keep the highest-scoring match when digit boxes overlap.
        """
        matches = sorted(
            matches,
            key=lambda match: match[2],
            reverse=True,
        )

        accepted: list[tuple[int, int, float, str]] = []

        for candidate in matches:
            candidate_left = candidate[0]
            candidate_right = candidate[1]

            overlaps = False

            for existing in accepted:
                existing_left = existing[0]
                existing_right = existing[1]

                intersection = max(
                    0,
                    min(candidate_right, existing_right)
                    - max(candidate_left, existing_left),
                )

                candidate_width = candidate_right - candidate_left
                existing_width = existing_right - existing_left

                smaller_width = min(
                    candidate_width,
                    existing_width,
                )

                overlap_ratio = intersection / smaller_width if smaller_width > 0 else 0

                if overlap_ratio >= 0.5:
                    overlaps = True
                    break

            if not overlaps:
                accepted.append(candidate)

        return accepted
