import cv2 as cv
import numpy as np

MATCH_METHODS = {
    "TM_CCOEFF": cv.TM_CCOEFF,
    "TM_CCOEFF_NORMED": cv.TM_CCOEFF_NORMED,
    "TM_CCORR": cv.TM_CCORR,
    "TM_CCORR_NORMED": cv.TM_CCORR_NORMED,
    "TM_SQDIFF": cv.TM_SQDIFF,
    "TM_SQDIFF_NORMED": cv.TM_SQDIFF_NORMED,
}


class ComputerVision:
    def __init__(self) -> None:
        pass

    @staticmethod
    def match_template(
        frame,
        crop_area=(0, 0, 0, 0),
        template=None,
        method=MATCH_METHODS["TM_CCOEFF_NORMED"],
        threshold=0.7,
        frame_to_draw=None,
        text_to_draw=None,
    ):
        """
        Match template in a frame.

        :param frame: Frame to search in.
        :param crop_area: Area to cut off from the frame, it's a tuple in the following format: (top, bottom, left, right).
            Eg.: (50, -50, 50, -50) will cut 50px from top, bottom, left and right. Default: (0, 0, 0, 0) with will not cut anything.
        :param template: Template to search for. Default is None, but it's required.
        :param method: Method to use for matching. Default is TM_CCOEFF_NORMED.
        :param threshold: Threshold to use for matching. Default is 0.7.
        :param frame_to_draw: Frame to draw on. Default is None, which will not draw.
        :param text_to_draw: Draw text with the match value. Default: None, which will not draw.

        :return: max_val, max_loc, center_loc, passed_threshold, drawn_frame
        """
        # Cut frame if needed (row_start:row_end, col_start:col_end)
        if crop_area != (0, 0, 0, 0):
            # row_end or col_end is 0, we translate it to the frame size
            if crop_area[1] == 0:
                crop_area = (crop_area[0], frame.shape[0], crop_area[2], crop_area[3])
            if crop_area[3] == 0:
                crop_area = (crop_area[0], crop_area[1], crop_area[2], frame.shape[1])
            frame = frame[crop_area[0] : crop_area[1], crop_area[2] : crop_area[3]]

        template_h = template.shape[0]
        template_w = template.shape[1]
        result = cv.matchTemplate(frame, template, method)
        _, max_val, _, max_loc = cv.minMaxLoc(result)
        max_loc_corrected = (max_loc[0] + crop_area[2], max_loc[1] + crop_area[0])
        center_loc = (
            max_loc_corrected[0] + template_w // 2,
            max_loc_corrected[1] + template_h // 2,
        )
        passed_threshold = max_val >= threshold

        if frame_to_draw is not None and passed_threshold:
            line_color = (0, 255, 0)
            line_type = cv.LINE_4
            top_left = max_loc_corrected
            bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
            cv.rectangle(
                frame_to_draw,
                top_left,
                bottom_right,
                color=line_color,
                lineType=line_type,
                thickness=2,
            )

            if text_to_draw:
                font_face = cv.FONT_HERSHEY_DUPLEX
                font_scale = 0.35
                font_color = (0, 0, 0)
                font_thickness = 1
                (text_w, text_h), _ = cv.getTextSize(
                    text_to_draw, font_face, font_scale, font_thickness
                )
                text_offset_x = (template_w - text_w) // 2
                text_offset_y = (
                    text_h + 5
                )  # 5px for some space between the text and the box
                text_pos = (
                    max_loc[0] + crop_area[2] + text_offset_x,
                    max_loc[1] + template_h + crop_area[0] + text_offset_y,
                )
                text_bg_color = (255, 255, 255)
                cv.rectangle(
                    frame_to_draw,
                    text_pos,
                    (text_pos[0] + text_w, text_pos[1] - text_h),
                    text_bg_color,
                    -1,
                )
                cv.putText(
                    frame_to_draw,
                    text_to_draw,
                    text_pos,
                    font_face,
                    font_scale,
                    font_color,
                    font_thickness,
                )

        drawn_frame = frame_to_draw
        return max_val, max_loc_corrected, center_loc, passed_threshold, drawn_frame

    @staticmethod
    def group_overlapping_rectangles(rectangles, overlap_threshold=0.5):
        """
        Merge substantially overlapping rectangles.

        Each rectangle is [x, y, width, height]. The return value is a
        NumPy array compatible with the iteration used by match_template_multi.
        """
        if not rectangles:
            return np.empty((0, 4), dtype=np.int32)

        pending = [np.asarray(rect, dtype=np.float64) for rect in rectangles]
        grouped = []

        while pending:
            cluster = [pending.pop(0)]
            changed = True

            while changed:
                changed = False
                cluster_box = np.mean(cluster, axis=0)
                cluster_x1, cluster_y1, cluster_w, cluster_h = cluster_box
                cluster_x2 = cluster_x1 + cluster_w
                cluster_y2 = cluster_y1 + cluster_h
                cluster_area = cluster_w * cluster_h
                remaining = []

                for candidate in pending:
                    candidate_x1, candidate_y1, candidate_w, candidate_h = candidate
                    candidate_x2 = candidate_x1 + candidate_w
                    candidate_y2 = candidate_y1 + candidate_h

                    intersection_w = max(
                        0.0,
                        min(cluster_x2, candidate_x2) - max(cluster_x1, candidate_x1),
                    )
                    intersection_h = max(
                        0.0,
                        min(cluster_y2, candidate_y2) - max(cluster_y1, candidate_y1),
                    )
                    intersection = intersection_w * intersection_h
                    candidate_area = candidate_w * candidate_h
                    smaller_area = min(cluster_area, candidate_area)
                    overlap = intersection / smaller_area if smaller_area > 0 else 0.0

                    if overlap >= overlap_threshold:
                        cluster.append(candidate)
                        changed = True
                    else:
                        remaining.append(candidate)

                pending = remaining

            grouped.append(np.rint(np.mean(cluster, axis=0)).astype(np.int32))

        return np.asarray(grouped, dtype=np.int32)

    @staticmethod
    def match_template_multi(
        frame,
        crop_area=(0, 0, 0, 0),
        template=None,
        method=MATCH_METHODS["TM_CCOEFF_NORMED"],
        threshold=0.7,
        box_offset=(0, 0),
        frame_to_draw=None,
        draw_rect=True,
        draw_marker=False,
        draw_text=False,
    ):
        """
        Match template, multiple times, in image. Return only the matches that pass the threshold.

        :param frame: Frame to search in.
        :param crop_area: Area to cut off from the frame, it's a tuple in the following format: (top, bottom, left, right).
            Eg.: (50, -50, 50, -50) will cut 50px from top, bottom, left and right. Default: (0, 0, 0, 0) which will not cut anything.
        :param template: Template to search for. Default is None, but it's required.
        :param method: Method to use for matching. Default is TM_CCOEFF_NORMED.
        :param threshold: Threshold to use for matching. Default is 0.7.
        :param box_offset: Increase the box size by this offset. Default is (0, 0) for (width, height).
        :param frame_to_draw: Frame to draw on. Default is None, which will not draw.
        :param draw_rect: Draw rectangle around the match. Default: True.
        :param draw_marker: Draw marker at the center of the match. Default: False.
        :param draw_text: Draw text with the match value. Default: False.

        :return: matches, drawn_frame
        """
        # Cut frame if needed (row_start:row_end, col_start:col_end)
        if crop_area != (0, 0, 0, 0):
            # row_end or col_end is 0, we translate it to the frame size
            if crop_area[1] == 0:
                crop_area = (crop_area[0], frame.shape[0], crop_area[2], crop_area[3])
            if crop_area[3] == 0:
                crop_area = (crop_area[0], crop_area[1], crop_area[2], frame.shape[1])
            frame = frame[crop_area[0] : crop_area[1], crop_area[2] : crop_area[3]]

        template_h = template.shape[0]
        template_w = template.shape[1]
        result = cv.matchTemplate(frame, template, method)

        # Get the all the positions from the match result that exceed our threshold
        locations = np.where(result >= threshold)
        # From [[y1, y2, y3], [x1, x2, x3]] to [(x1, y1), (x2, y2), (x3, y3)]
        locations = list(zip(*locations[::-1]))
        # print(locations)

        # Create a list of [x, y, w, h] rectangles.
        rectangles = []
        for loc in locations:
            rect = [
                int(loc[0]),
                int(loc[1]),
                int(template_w + box_offset[0]),
                int(template_h + box_offset[1]),
            ]
            rectangles.append(rect)

        # OpenCV 5 no longer exposes cv.groupRectangles in some Python builds.
        # Merge overlapping detections locally while preserving isolated matches.
        rectangles = ComputerVision.group_overlapping_rectangles(
            rectangles, overlap_threshold=0.5
        )
        # print(rectangles)

        matches = []
        if len(rectangles):
            # print('Found template')
            for x, y, w, h in rectangles:
                # Determine the center position, initial point + half size + correction
                center_x = x + (w // 2) + crop_area[2]
                center_y = y + (h // 2) + crop_area[0]

                # Save the matches
                matches.append((center_x, center_y))

                # Already filtered by threshold, no need to check again
                if frame_to_draw is None:
                    continue

                if draw_rect:
                    line_color = (0, 255, 0)
                    line_type = cv.LINE_4
                    top_left = (x + crop_area[2], y + crop_area[0])
                    bottom_right = (x + w + crop_area[2], y + h + crop_area[0])
                    cv.rectangle(
                        frame_to_draw,
                        top_left,
                        bottom_right,
                        color=line_color,
                        lineType=line_type,
                        thickness=2,
                    )
                if draw_marker:
                    marker_color = (0, 0, 200)
                    marker_type = cv.MARKER_CROSS
                    cv.drawMarker(
                        frame_to_draw,
                        (center_x, center_y),
                        color=marker_color,
                        markerType=marker_type,
                        markerSize=40,
                        thickness=2,
                    )
                if draw_text:
                    text = f"({center_x}, {center_y})"
                    font_face = cv.FONT_HERSHEY_DUPLEX
                    font_scale = 0.35
                    font_color = (0, 0, 0)
                    font_thickness = 1
                    (text_w, text_h), _ = cv.getTextSize(
                        text, font_face, font_scale, font_thickness
                    )
                    text_offset_x = (w - text_w) // 2
                    text_offset_y = text_h + 5 if draw_rect else -h + text_h + 10
                    text_pos = (
                        x + crop_area[2] + text_offset_x,
                        y + h + crop_area[0] + text_offset_y,
                    )
                    cv.putText(
                        frame_to_draw,
                        text,
                        text_pos,
                        font_face,
                        font_scale,
                        font_color,
                        font_thickness,
                    )

        drawn_frame = frame_to_draw
        return matches, drawn_frame
