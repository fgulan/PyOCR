import cv2
import numpy as np
from ocr_image import OCRImage


class LineImage(OCRImage):

    def __init__(self, image, width, height, x_offset=0, y_offset=0):
        super().__init__(image, width, height, x_offset, y_offset)

        self.words = []
