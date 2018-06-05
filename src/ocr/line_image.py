import cv2
import numpy as np
from ocr_image import OCRImage
from word_image import WordImage
from utils import hist, constants
from utils.helpers import debug_plot_array


class LineImage(OCRImage):

    BASELINE_DISTANCE_RATIO = 0.15
    CAP_DISTANCE_RATIO = 0.3
    # Idea took from https://content.sciendo.com/view/journals/amcs/27/1/article-p195.xml

    def __init__(self, image, width, height, x_offset=0, y_offset=0):
        super().__init__(image, width, height, x_offset, y_offset)

        self.words = []

    def get_segments(self):
        image = self.get_image()
        use_cap = True
        
        # Get all possible spaces (used image is without baseline and cap)
        spaces = self._get_word_spaces_candidates(image, use_cap=use_cap)

        # Align previous spaces to include overlap if cap is not used
        if not use_cap:
            spaces = self._align_space_candidates(spaces, image)

        word_coords = self._extract_word_coords(image, spaces)
        word_coords = self._strip_words(word_coords, image)
        self.words = self._map_word_coords_to_object(image, word_coords)
        
        return self.words
    
    def _align_space_candidates(self, space_candidates, image):
        aligned_spaces = []

        for candidate in space_candidates:
            region = self._extract_space_region(*candidate, image)
            aligned_spaces.append(region)

        return aligned_spaces

    def _extract_space_region(self, start_x, end_x, image):
        new_start = None
        new_end = None
    
        v_proj = hist.vertical_projection(image[:,start_x:end_x + 1])
        new_candidates = []
        for index, value in enumerate(v_proj):
            if value <= 2:
                if new_start == None:
                    new_start = index
                new_end = index
            else:
                if new_start != None and new_end != None:
                    new_candidates.append((new_start, new_end))
                new_start = None
                new_end = None
        
        if len(new_candidates) == 0:
            middle = (start_x + end_x + 1) / 2
            new_start = int(middle - 1)
            new_end = int(middle + 1)
        else:
            # sort them so that biggest space is used
            new_candidates = sorted(new_candidates, key=lambda region: region[1] - region[0], reverse=True)
            new_start, new_end = new_candidates[0]
            new_start += start_x
            new_end += start_x

        return new_start, new_end

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
        y_offset = self.get_y()
        line_start_x = self.get_x()

        words = []
        for (start_x, end_x) in word_coords:
            x_offset = line_start_x + start_x
            word_width = end_x - start_x + 1
            
            roi_image = image[:,start_x:end_x + 1]
            word = WordImage(roi_image, word_width, line_height, x_offset, y_offset)
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

    def _get_word_spaces_candidates(self, image, use_cap=True):
        height, _ = image.shape[:2]

        # Lets ignore everything below baseline and above cap in 
        # histogram calculation so it won't create non space issues
        offset = height * self.BASELINE_DISTANCE_RATIO
        if use_cap:
            start_y = 0
        else:
            start_y = int(height * self.CAP_DISTANCE_RATIO)
            
        end_y = int(height - offset)
        v_proj = hist.vertical_projection(image[start_y:end_y,:])

        start_x, end_x = None, None
        min_space_len = height * constants.WORD_SPACE_DISTANCE_HEIGHT_RATIO

        all_spaces = []
        for index, value in enumerate(v_proj):
            if value <= 0:
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
