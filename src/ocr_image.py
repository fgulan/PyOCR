import sys
import cv2
import numpy as np

from utils import hist

class OCRImage:

    MAX_ROTATE = 15

    def __init__(self, image, width, height, x_offset=0, y_offset=0):
        self._image = image
        self._x_offset = x_offset
        self._y_offset = y_offset
        self._width = x_offset
        self._height = height

    def save(self, name):
        cv2.imwrite(name, self._image)

    def get_image(self):
        return self._image

    def set_image(self, image):
        self._image = image

    def get_x(self):
        return self._x_offset

    def get_y(self):
        return self._y_offset

    def get_height(self):
        return self._height

    def get_width(self):
        return self._width

    def get_bounding_box(self):
        return {'x': self.get_x(), 'y': self.get_y(), 'width': self.get_width(), 'height': self.get_height()}

    def get_segments(self):
        image = self.get_image()

        h_proj = hist.horizontal_projection(image)
        v_proj = hist.vertical_projection(image)

        min_x, max_x = hist.blob_range(v_proj)
        min_y, max_y = hist.blob_range(h_proj)

        width = max_x - min_x
        height = max_y - min_y
        roi_image = image[min_y:max_y, min_x:max_x]

        return roi_image, width, height, min_x, min_y

    def fix_skew(self):
        angle = self._calculate_skewed_angle(self._image)

        if abs(angle) < self.MAX_ROTATE:
            self._image = self._rotate_image(self._image, angle)
            self._height, self._width = self._image.shape

        return angle

    def _calculate_skewed_angle(self, image):
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            return -(90 + angle)
        else:
            return -angle

    def _rotate_image(self, image, angle):
        # Add border so when image is rotated - black pixels will be filled
        image = cv2.copyMakeBorder(
            image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix,
                              (width, height),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
