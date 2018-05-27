import cv2
import numpy as np
from ocr_image import OCRImage
from utils.helpers import debug_plot_array, debug_display_image
from utils import hist, constants
from char_image import CharImage

class WordImage(OCRImage):

    BASELINE_CUTOFF_RATIO = 0.85

    def __init__(self, image, width, height, x_offset=0, y_offset=0):
        super().__init__(image, width, height, x_offset, y_offset)

        self.characters = []

    def get_segments(self):
        image = self.get_image()
        height, width = image.shape[:2]

        # Remove baseline for easier character segmentation
        median_height = height // (1 + constants.CAP_HEIGHT +
                                   constants.BASELINE_HEIGHT)
        baseline_height = int(
            median_height * constants.BASELINE_HEIGHT * self.BASELINE_CUTOFF_RATIO)

        v_proj = hist.vertical_projection(
            image[0:height - baseline_height, :])
        v_proj_smooth = v_proj #hist.running_mean(v_proj, 3)
        
        hist_spaces = hist.get_histogram_spaces(v_proj_smooth, 0)
        hist_peaks = hist.get_histogram_peaks(v_proj_smooth, hist_spaces)
        hist_peaks = hist.filter_histogram_peaks(hist_peaks, 2)
        hist_peaks = self._process_joined_characters(v_proj_smooth, hist_peaks, 1.3)

        self.characters = self._map_char_coords_to_object(image, hist_peaks)
        return self.characters

    def _process_joined_characters(self, histogram, char_coords, threshold_ratio):
        char_widths = [x2 - x1 for (x1, x2) in char_coords]
        avg_char_width = sum(char_widths) / len(char_widths)
        threshold = threshold_ratio * avg_char_width
        new_coords = []

        for coord in char_coords:
            start_x, end_x = coord
            if end_x - start_x > threshold:
                candidates = self._get_char_candidates(histogram, coord)
                new_coords.extend(candidates)
            else:
                new_coords.append(coord)
        return new_coords

    def _get_char_candidates(self, histogram, char_coord):
        start_x, end_x = char_coord
        roi_hist = histogram[start_x:end_x]
        # roi_hist = hist.running_mean(roi_hist, 7)

        hist_spaces = hist.get_histogram_spaces(roi_hist, 1)
        hist_peaks = hist.get_histogram_peaks(roi_hist, hist_spaces)
        hist_peaks = hist.filter_histogram_peaks(hist_peaks, 2)

        translated_peaks = []
        for start_peak, end_peak in hist_peaks:
            x1 = start_x + start_peak
            x2 = start_x + end_peak
            translated_peaks.append((x1, x2))

        return translated_peaks

    def _map_char_coords_to_object(self, image, char_coords):
        char_height = self.get_height()
        start_y = self.get_y()
        word_start_x = self.get_x()

        chars = []
        for (start_x, end_x) in char_coords:
            char_width = end_x - start_x
            roi_image = image[:,start_x:end_x]
            x_offset = word_start_x + start_x
            char = WordImage(roi_image, char_width, char_height, x_offset, start_y)
            chars.append(char)
        return chars
