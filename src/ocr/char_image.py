import cv2
import numpy as np
from ocr_image import OCRImage

from utils.helpers import debug_display_image


class CharImage(OCRImage):

    SCALED_ROI_IMAGE_SIZE = 36
    SCALED_IMAGE_SIZE = 40
    AVERAGE_LINE_HEIGHT = 60

    def __init__(self, image, width, height, x_offset=0, y_offset=0):
        super().__init__(image, width, height, x_offset, y_offset)

    def get_scaled_image(self, line_height=60):
        image = self.get_image()
        old_image_size = image.shape[:2]

        ratio = float(self.SCALED_ROI_IMAGE_SIZE) / max(old_image_size)

        if ratio < 1:
            new_image = self._scale_image(image, ratio)
        else:
            # Scale small elements
            height_ratio = self.AVERAGE_LINE_HEIGHT / line_height
            new_image = self._scale_image(image, height_ratio)

        new_height, new_width = new_image.shape[:2]

        delta_w = self.SCALED_IMAGE_SIZE - new_width
        delta_h = self.SCALED_IMAGE_SIZE - new_height
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        new_image = cv2.copyMakeBorder(new_image, top, bottom, left, right,
                                       cv2.BORDER_CONSTANT, value=0)
        return new_image

    def _scale_image(self, image, ratio):
        old_image_size = image.shape[:2]
        new_height, new_width = tuple([int(x * ratio) for x in old_image_size])
        image = cv2.resize(image, (new_width, new_height),
                           interpolation=cv2.INTER_NEAREST)
        return image
