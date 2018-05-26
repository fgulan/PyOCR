import cv2
import numpy as np
from ocr_image import OCRImage
from line_image import LineImage

from utils import hist
from utils.helpers import debug_plot_array


class TextImageBaseline(OCRImage):

    NOISE_PIXELS_THRESHOLD = 2
    MEAN_HISTOGRAM_CUTOFF = 0.9
    CAP_HEIGHT = 0.5
    BASELINE_HEIGHT = 0.7

    def __init__(self, image, width, height, x_offset=0, y_offset=0):
        super().__init__(image, width, height, x_offset, y_offset)

        self.lines = []

    def get_segments(self):
        image = self.get_image()
        height, width = image.shape[:2]
        h_proj = hist.horizontal_projection(image)
        h_proj_smooth = hist.running_mean(h_proj, 5)

        hist_spaces = self._get_hist_spaces(h_proj_smooth)
        hist_peaks = self._get_hist_peaks(h_proj_smooth, hist_spaces)
        hist_peaks = self._filter_hist_peaks(h_proj_smooth, hist_peaks)
        
        lines = self._include_typography(hist_peaks, image)

        line_images = []
        for coords in lines:
            y1, y2 = coords
            roi_image = image[y1:y2, ]

            x_offset = self.get_x()
            y_offset = self.get_y()
            width = width
            height = y2 - y1

            line_image = LineImage(
                roi_image, width, height, x_offset, y_offset)
            line_images.append(line_image)

        self.lines = line_images
        return line_images

    def _include_typography(self, lines, image):
        image_height, _ = image.shape[:2]
        max_index = image_height - 1

        padded_lines = []
        for (y1, y2) in lines:
            line_height = y2 - y1
            cap_height = int(line_height * self.CAP_HEIGHT)
            baseline_height = int(line_height * self.BASELINE_HEIGHT)

            y1 = max(0, y1 - cap_height)
            y2 = min(max_index, y2 + baseline_height)
            padded_lines.append((y1, y2))

        return np.array(padded_lines)      

    def _filter_hist_peaks(self, hist, peaks):
        peak_means = self._get_hist_peaks_mean(hist, peaks)

        new_coords = []
        for (coords, mean) in zip(peaks, peak_means):
            coords = self._get_mean_peak_cords(hist, coords, mean)
            new_coords.append(coords)
        return new_coords

    def _get_mean_peak_cords(self, hist, coords, mean):
        start_x, end_x = coords
        threshold = mean * self.MEAN_HISTOGRAM_CUTOFF
        candidates = []

        x1, x2 = None, None
        for index in range(start_x, end_x):
            value = hist[index]
            if index == end_x - 1:
                if x1 != None and x2 != None:
                    candidates.append((x1, x2))

            if value > threshold:
                if x1 == None:
                    x1 = index
                x2 = index
            else:
                if x1 != None and x2 != None:
                    candidates.append((x1, x2))
                x1, x2 = None, None

        return self._get_longest_peak_candidate(candidates)

    def _get_longest_peak_candidate(self, candidates):
        sorted_candidates = sorted(candidates, key=lambda x: (x[1] - x[0]), reverse=True)
        return sorted_candidates[0]

    def _get_hist_peaks_mean(self, hist, peaks):
        peaks_mean = []
        for (x1, x2) in peaks:
            roi_hist = hist[x1:x2]
            peak_mean = np.mean(roi_hist)
            peaks_mean.append(peak_mean)

        return peaks_mean

    def _get_hist_spaces(self, hist):
        spaces = []
        start_x, end_x = None, None

        for index, value in enumerate(hist):
            if value <= self.NOISE_PIXELS_THRESHOLD:
                if start_x == None:
                    start_x = index
                end_x = index
            else:
                if start_x != None and end_x != None:
                    spaces.append((start_x, end_x))
                start_x = None
                end_x = None

        return spaces

    def _get_hist_peaks(self, hist, spaces):
        size = len(hist)

        if len(spaces) == 0:
            return [(0, size)]

        current_x = 0
        hist_peaks = []
        for space in spaces:
            space_start_x, space_end_x = space
            word = (current_x, space_start_x)
            hist_peaks.append(word)
            current_x = space_end_x

        if current_x < size - 1:
            word = (current_x, size - 1)
            hist_peaks.append(word)

        return hist_peaks
