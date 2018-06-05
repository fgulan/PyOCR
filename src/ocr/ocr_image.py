import sys
import cv2
import numpy as np

from utils import hist
import imutils

from scipy.ndimage import interpolation as inter

class OCRImage:

    MAX_ROTATE_ANGLE = 3
    ANGLE_DELTA = 0.05
    MAX_SCALED_DIMENSION = 800

    def __init__(self, image, width, height, x_offset=0, y_offset=0):
        self._image = image
        self._x_offset = x_offset
        self._y_offset = y_offset
        self._width = width
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
        return {'x': self.get_x(),
                'y': self.get_y(),
                'width': self.get_width(),
                'height': self.get_height()}

    def get_segments(self):
        image = self.get_image()

        h_proj = hist.horizontal_projection(image)
        v_proj = hist.vertical_projection(image)

        min_x, max_x = hist.blob_range(v_proj)
        min_y, max_y = hist.blob_range(h_proj)

        width = max_x - min_x + 1
        height = max_y - min_y + 1
        roi_image = image[min_y:max_y + 1, min_x:max_x + 1]

        return roi_image, width, height, min_x, min_y

    def fix_skew(self):
        angle = self._calculate_skewed_angle_projection(self._image)

        if abs(angle) < self.MAX_ROTATE_ANGLE:
            self._image = self._rotate_image(self._image, angle)
            self._height, self._width = self._image.shape

        return angle

    def _calculate_skewed_angle_projection(self, input_image):
        height, width = input_image.shape

        new_image = input_image.copy()

        biggest_dimension = max(height, width)
        scale = self.MAX_SCALED_DIMENSION / biggest_dimension

        new_height, new_width = round(height * scale), round(width * scale)
        scaled_image = cv2.resize(new_image, (new_width, new_height))

        angles = np.arange(-self.MAX_ROTATE_ANGLE, self.MAX_ROTATE_ANGLE + self.ANGLE_DELTA, self.ANGLE_DELTA)
        scores = []
        for angle in angles:
            score = self._find_rotation_score(scaled_image, angle)
            scores.append(score)

        best_angle = angles[np.argmax(scores)]
        return best_angle

    def _find_rotation_score(self, image, angle):
        # Rotate image for given angle
        rotated_image = inter.rotate(image, angle, reshape=False, order=0)
        # Calculate horizontal projection
        h_proj = hist.horizontal_projection(rotated_image)
        # Calculate projection gradient
        score = np.sum((h_proj[1:] - h_proj[:-1]) ** 2)
        
        return score

    def _calculate_skewed_angle_bbox(self, image):
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
        rotated_image = cv2.warpAffine(image, rotation_matrix,
                                       (width, height),
                                       flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_REPLICATE)
        output_image = cv2.threshold(
            rotated_image, 127, 255, cv2.THRESH_BINARY)[1]
        return output_image
