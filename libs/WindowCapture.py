import cv2 as cv
import numpy as np
import win32con
import win32gui
import win32ui


class WindowCapture:
    def __init__(self, hwnd, crop_area=(8, 30, 8, 8)):
        """
        :param hwnd: int. Handle of the window to capture.
        :param crop_area: tuple (left, top, right, bottom). Area to crop from the window.
                Default: (8, 30, 8, 8) which accounts for the game window border and titlebar.
        """

        self.hwnd = hwnd
        if not self.hwnd:
            raise ValueError("Window not found")

        self.crop_l = crop_area[0]
        self.crop_t = crop_area[1]
        self.crop_r = crop_area[2]
        self.crop_b = crop_area[3]

        self.w = 0
        self.h = 0
        self.offset_x = 0
        self.offset_y = 0
        self.__update_size_and_offset()
        self._closed = False

    def close(self):
        """Prevent new capture operations; per-frame GDI handles are transient."""
        self._closed = True

    def get_frame(self):
        """
        Take a screenshot of the target window. Works with windows in background
        and foreground. Fullscreen or windowed. But doesn't work with minimized
        or windows outside the screen.

        :return: (numpy array, numpy array). The first array is the image in BGR format, 3 channels.
                The second array is the image in grayscale format, 1 channel.
        """

        if self._closed:
            raise RuntimeError("Window capture is closed.")

        window_dc = 0
        source_dc = None
        compatible_dc = None
        bitmap = None
        try:
            window_dc = win32gui.GetWindowDC(self.hwnd)
            if not window_dc:
                raise RuntimeError("Could not acquire the window device context.")
            source_dc = win32ui.CreateDCFromHandle(window_dc)
            compatible_dc = source_dc.CreateCompatibleDC()
            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(source_dc, self.w, self.h)
            compatible_dc.SelectObject(bitmap)
            compatible_dc.BitBlt(
                (0, 0),
                (self.w, self.h),
                source_dc,
                (self.crop_l, self.crop_t),
                win32con.SRCCOPY,
            )
            signedIntsArray = bitmap.GetBitmapBits(True)
        finally:
            if source_dc is not None:
                source_dc.DeleteDC()
            if compatible_dc is not None:
                compatible_dc.DeleteDC()
            if window_dc:
                win32gui.ReleaseDC(self.hwnd, window_dc)
            if bitmap is not None:
                win32gui.DeleteObject(bitmap.GetHandle())

        img = np.frombuffer(signedIntsArray, dtype=np.uint8)
        img.shape = (self.h, self.w, 4)

        # drop the alpha channel, or cv.matchTemplate() will throw an error like:
        #   error: (-215:Assertion failed) (depth == CV_8U || depth == CV_32F) && type == _templ.type()
        #   && _img.dims() <= 2 in function 'cv::matchTemplate'
        img = img[..., :3]

        # make image C_CONTIGUOUS to avoid errors that look like:
        #   File ... in draw_rectangles
        #   TypeError: an integer is required (got type tuple)
        # see the discussion here:
        # https://github.com/opencv/opencv/issues/14866#issuecomment-580207109
        img = np.ascontiguousarray(img)

        # DEBUGGING: Show the image
        # cv.imshow("screenshot", img)
        # cv.waitKey(1)

        # DEBUGGING: Save the screenshot to disk
        # cv.imwrite("screenshot.png", img)

        # Convert image to gray
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        return img, img_gray

    def get_screen_pos(self, pos):
        """
        Translate a pixel position on a screenshot image to a pixel position on the screen.

        :param pos: tuple (x, y). Position on the screenshot image.
        :return: tuple (x, y). Position on the screen.
        """
        self.__update_size_and_offset()
        return (pos[0] + self.offset_x, pos[1] + self.offset_y)

    def __update_size_and_offset(self):
        """
        Size doesn't change often, but it's a step to update the offset. Offset
        do change often, it updates when we move the target window.
        """
        # get the window size
        left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
        self.w = right - left - self.crop_l - self.crop_r
        self.h = bottom - top - self.crop_t - self.crop_b

        # set the cropped coordinates offset so we can translate screenshot
        # images into actual screen positions
        self.offset_x = left + self.crop_l
        self.offset_y = top + self.crop_t
