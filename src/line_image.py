import cv2
import numpy as np
from ocr_image import OCRImage
from word_image import WordImage
from utils import hist, constants
from utils.helpers import debug_plot_array


class LineImage(OCRImage):

    SPACE_NOISE_THRESHOLD = 2
    BASELINE_DISTANCE_RATIO = 0.15
    # Idea took from https://content.sciendo.com/view/journals/amcs/27/1/article-p195.xml
    SPACE_DISTANCE_RATIO = 5

    def __init__(self, image, width, height, x_offset=0, y_offset=0):
        super().__init__(image, width, height, x_offset, y_offset)

        self.words = []

    def get_segments(self):
        image = self.get_image()
        
        spaces = self._get_word_spaces(image)
        word_coords = self._extract_word_coords(image, spaces)
        word_coords = self._strip_words(word_coords, image)
        self.words = self._map_word_coords_to_object(image, word_coords)
        
        return self.words

    def _strip_words(self, word_coords, image):
        new_coords = []

        for coord in word_coords:
            start_x, end_x = coord

            roi_image = image[:,start_x:end_x + 1]
            v_proj = hist.vertical_projection(roi_image)
            x1, x2 = hist.blob_range(v_proj)
            
            new_start_x = start_x + x1
            new_end_x = start_x + x2
            
            new_coords.append((new_start_x, new_end_x))
        
        return new_coords

    def _map_word_coords_to_object(self, image, word_coords):
        line_height = self.get_height()
        start_y = self.get_y()

        words = []
        for (start_x, end_x) in word_coords:
            word_width = end_x - start_x + 1
            roi_image = image[:,start_x:end_x + 1]
            word = WordImage(roi_image, word_width, line_height, start_x, start_y)
            words.append(word)

        return words

    def _extract_word_coords(self, image, spaces):
        _, width = image.shape[:2]

        if len(spaces) == 0:
            return [(0, width - 1)]

        current_x = 0
        word_coords = []
        for space in spaces:
            space_start_x, space_end_x = space

            # space_start_x is first whitespace pixel, so first before is background
            word = (current_x, space_start_x - 1)
            word_coords.append(word)

            # space_end_x is last whitespace pixel, so first next is foreground
            current_x = space_end_x + 1

        if current_x < width - 1:
            word = (current_x, width - 1)
            word_coords.append(word)
        
        return word_coords

    def _get_word_spaces(self, image):
        height, _ = image.shape[:2]

        # Lets ignore everything below baseline in histogram calcualtion
        # so it won't create non space issues
        offset = height * self.BASELINE_DISTANCE_RATIO
        start_y = 0
        end_y = int(height - offset)
        v_proj = hist.vertical_projection(image[start_y:end_y,:])

        start_x, end_x = None, None
        min_space_len = height * constants.WORD_SPACE_DISTANCE_HEIGHT_RATIO

        all_spaces = []
        for index, value in enumerate(v_proj):
            if value < self.SPACE_NOISE_THRESHOLD:
                if start_x == None:
                    start_x = index
                end_x = index
            else:
                if start_x != None and end_x != None:
                    all_spaces.append((start_x, end_x))
                start_x = None
                end_x = None

        spaces = list(
            filter(lambda points: points[1] - points[0] + 1 >= min_space_len, all_spaces))

        return spaces
