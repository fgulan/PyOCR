import cv2
import numpy as np
from ocr_image import OCRImage
from line_image import LineImage

from utils import hist, constants
from utils.helpers import debug_plot_array


class TextImageBaseline(OCRImage):

    def __init__(self, image, width, height, x_offset=0, y_offset=0):
        super().__init__(image, width, height, x_offset, y_offset)

        self.lines = []

    def get_segments(self):
        image = self.get_image()
        height, width = image.shape[:2]

        # Idea took from https://content.sciendo.com/view/journals/amcs/27/1/article-p195.xml
        h_proj = hist.horizontal_projection(image)
        h_proj_smooth = hist.running_mean(h_proj, 5)
        hist_spaces = hist.get_histogram_spaces(h_proj_smooth, constants.NOISE_PIXELS_THRESHOLD)
        hist_peaks = hist.get_histogram_peaks(h_proj_smooth, hist_spaces)

        hist_peaks = self._filter_hist_peaks(h_proj_smooth, hist_peaks)

        lines = self._include_typography(hist_peaks, image)
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
        index = 0
        for (y1, y2) in lines:
            index +=1
            roi = image[y1:y2,:]
            v_proj = hist.vertical_projection(roi)
            x1, x2 = hist.blob_range(v_proj)
            if x1 >= 0 and x2 >= 0:
                blobs.append((y1, y2, x1, x2))

        return blobs

    def _include_typography(self, lines, image):
        image_height, _ = image.shape[:2]
        max_index = image_height - 1

        padded_lines = []
        for (y1, y2) in lines:
            line_height = y2 - y1
            cap_height = int(line_height * constants.CAP_HEIGHT)
            baseline_height = int(line_height * constants.BASELINE_HEIGHT)

            y1 = max(0, y1 - cap_height)
            y2 = min(max_index, y2 + baseline_height)
            padded_lines.append((y1, y2))

        return np.array(padded_lines)      

    def _filter_hist_peaks(self, histogram, peaks):
        peak_means = hist.get_histogram_peak_means(histogram, peaks)

        coords_candidate = []
        for (coords, mean) in zip(peaks, peak_means):
            new_coords = self._get_mean_peak_cords(histogram, coords, mean)
            coords_candidate.extend(new_coords)
        
        lines = self._filter_lines(coords_candidate)

        return lines

    def _filter_lines(self, lines):
        line_heights = [y2 - y1 for (y1, y2) in lines]
        avg_line_height = sum(line_heights) / len(lines)

        line_height_treshold = avg_line_height * 0.9

        new_lines = []
        for (y1, y2) in lines:
            line_height = y2 - y1
            if line_height >= line_height_treshold:
                new_lines.append((y1, y2))
        return new_lines

    def _get_mean_peak_cords(self, hist, coords, mean):
        start_x, end_x = coords
        threshold = mean * constants.MEAN_HISTOGRAM_CUTOFF
        candidates = []

        x1, x2 = None, None
        for index in range(start_x, end_x):
            value = hist[index]
            if index == end_x - 1:
                if x1 != None and x2 != None:
                    candidates.append((x1, x2))
                    break

            if value > threshold:
                if x1 == None:
                    x1 = index
                x2 = index
            else:
                if x1 != None and x2 != None:
                    candidates.append((x1, x2))
                x1, x2 = None, None

        if len(candidates) == 0:
            return [coords]
        else:
            return candidates

    def _get_longest_peak_candidate(self, candidates):
        sorted_candidates = sorted(candidates, key=lambda x: (x[1] - x[0]), reverse=True)
        return sorted_candidates[0]
