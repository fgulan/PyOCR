import cv2
import numpy as np
from ocr_image import OCRImage
from line_image import LineImage

from utils import hist

class TextImage(OCRImage):

    LINE_SEPARATION_NOISE_PERCENTAGE = 0.01
    LINE_HEIGHT_NOISE_PERCENTAGE = 0.1
    LINE_PADDING_PRECENTAGE = 0.01    
    ARTIFACT_PERCENTAGE_THRESHOLD = 0.08
    MINIMUM_LINE_OVERLAP = 0.25

    def __init__(self, image, width, height, x_offset=0, y_offset=0):
        super().__init__(image, width, height, x_offset, y_offset)

        self.lines = []

    def get_segments(self):
        image = self.get_image()

        lines = self._extract_lines(image)
        lines = self._filter_lines(lines)
        lines = self._process_padding(lines, image)

        blobs = self._strip_lines(lines, image)
        line_images = []
        for (y1, y2, x1, x2) in blobs:
            roi_image = image[y1:y2, x1:x2]

            x_offset = self.get_x() + x1
            y_offset = self.get_y() + y1
            width = x2 - x1
            height = y2 - y1

            line_image = LineImage(roi_image, width, height, x_offset, y_offset)
            line_images.append(line_image)

        self.lines = line_images
        return line_images

    def _strip_lines(self, lines, image):
        blobs = []

        for (y1, y2) in lines:
            roi = image[y1:y2,:]
            v_proj = hist.vertical_projection(roi)
            x1, x2 = hist.blob_range(v_proj)
            if x1 >= 0 and x2 >= 0:
                blobs.append((y1, y2, x1, x2))

        return blobs

    def _extract_lines(self, image):
        h_proj = hist.horizontal_projection(image)
        h_proj_th = np.mean(h_proj) * self.LINE_SEPARATION_NOISE_PERCENTAGE
        height, _ = image.shape[:2]

        y1, y2 = None, None
        lines = []
        for index, h_line in enumerate(h_proj):
            if index == height - 1:
                if y1 != None and y2 != None:
                    lines.append((y1, y2))

            if h_line > h_proj_th:
                if y1 == None:
                    y1 = index
                    y2 = index
                else:
                    y2 = index
            else:
                if y1 != None and y2 != None:
                    lines.append((y1, y2))
                y2 = None
                y1 = None

        return lines

    def _filter_lines(self, lines):
        line_heights = [y2 - y1 for (y1, y2) in lines]
        avg_line_height = sum(line_heights) / len(lines)

        line_height_treshold = avg_line_height * self.LINE_HEIGHT_NOISE_PERCENTAGE

        new_lines = []
        for (y1, y2) in lines:
            line_height = y2 - y1
            if line_height >= line_height_treshold:
                new_lines.append((y1, y2))
        return new_lines

    def _process_padding(self, lines, image):
        padding_precentage = 1 + self.LINE_PADDING_PRECENTAGE
        image_height, _ = image.shape[:2]
        max_index = image_height - 1

        padded_lines = []
        for (y1, y2) in lines:
            line_height = y2 - y1
            padded_line_height = line_height * padding_precentage
            half_distance = round((padded_line_height - line_height) / 2)
            y1 = max(0, y1 - half_distance)
            y2 = min(max_index, y2 + half_distance)
            padded_lines.append((y1, y2))

        # Pad lines below average height to improve word extraction
        line_heights = [y2 - y1 for (y1, y2) in lines]
        avg_line_height = sum(line_heights) / len(lines)

        new_lines = []
        for (y1, y2) in padded_lines:
            line_height = y2 - y1
            if line_height < avg_line_height:
                half_distance = round((avg_line_height - line_height) / 2)
                y1 = max(0, y1 - half_distance)
                y2 = min(max_index, y2 + half_distance)
            new_lines.append((y1, y2))

        return np.array(new_lines)
