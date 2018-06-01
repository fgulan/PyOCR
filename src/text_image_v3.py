import cv2
import numpy as np
from ocr_image import OCRImage
from line_image import LineImage

from utils import hist, constants
from utils.helpers import debug_plot_array


class TextImageBaseline(OCRImage):

    AVERAGE_MIN_LINE_HEIGHT_CUTOFF_RATIO = 0.25
    AVERAGE_MAX_LINE_HEIGHT_CUTOFF_RATIO = 1.5

    def __init__(self, image, width, height, x_offset=0, y_offset=0):
        super().__init__(image, width, height, x_offset, y_offset)

        self.lines = []

    def _min_distance_between_peaks(self, peak_1, peak_2):
        peak_1_start, peak_1_end = peak_1
        peak_2_start, peak_2_end = peak_2

        top_top_dist = abs(peak_1_start - peak_2_start)
        top_bottom_dist = abs(peak_1_start - peak_2_end)
        bottom_top_dist = abs(peak_1_end - peak_2_start)
        bottom_bottom_dist = abs(peak_1_end - peak_2_end)

        return min(top_top_dist, top_bottom_dist, bottom_top_dist, bottom_bottom_dist)

    def _join_peaks(self, peak_1, peak_2):
        peak_1_start, peak_1_end = peak_1
        peak_2_start, peak_2_end = peak_2

        new_start = min(peak_1_start, peak_2_start)
        new_end = max(peak_1_end, peak_2_end)
        return new_start, new_end

    def _get_average_lines_height(self, line_peaks):
        line_heights = [y2 - y1 + 1 for (y1, y2) in line_peaks]
        avg_line_height = sum(line_heights) / len(line_peaks)
        return avg_line_height

    def _join_small_candidates(self, lines, small_line_candidates, avg_line_height):
        if len(small_line_candidates) == 0 or len(lines) == 0:
            return lines
        new_lines = list(lines)

        for small_line in small_line_candidates:
            infos = []

            for index, line in enumerate(lines):
                distance = self._min_distance_between_peaks(small_line, line)
                infos.append((index, distance, small_line))

            line_index, _, small_line_peak = sorted(
                infos, key=lambda info: info[1])[0]
            line = new_lines[line_index]
            new_peak = self._join_peaks(line, small_line_peak)
            new_lines[line_index] = new_peak

        return new_lines

    def _separate_big_candidates(self, lines, big_line_candidates, avg_line_height, h_proj):
        if len(big_line_candidates) == 0 or len(lines) == 0:
            return lines
        new_lines = list(lines)
        max_index = len(h_proj) - 1
        
        # Idea took from https://content.sciendo.com/view/journals/amcs/27/1/article-p195.xml
        h_proj_smooth = hist.running_mean(h_proj, 5)
        new_candidates = []
        for big_line in big_line_candidates:
            line_start, line_end = big_line
            roi_hist = h_proj_smooth[line_start:line_end + 1]
            roi_mean = np.mean(roi_hist)
            new_coords = self._get_mean_peak_cords(h_proj_smooth, big_line, roi_mean)
            new_candidates.extend(new_coords)

        # Filter new candidates
        new_candidates = self._filter_small_lines(new_candidates, 3)
        new_candidates = self._include_typography(new_candidates, max_index, avg_line_height)

        # Add new candidates
        new_lines.extend(new_candidates)

        # Sort new lines
        new_lines = sorted(new_lines, key=lambda coord: coord[0])

        return new_lines

    def _process_line_candidates(self, line_candidates, h_proj):
        avg_line_candidate_height = self._get_average_lines_height(
            line_candidates)

        new_lines = []
        small_line_candidates = []
        big_line_candidates = []

        max_threshold = avg_line_candidate_height * \
            self.AVERAGE_MAX_LINE_HEIGHT_CUTOFF_RATIO
        min_threshold = avg_line_candidate_height * \
            self.AVERAGE_MIN_LINE_HEIGHT_CUTOFF_RATIO

        for (start_y, end_y) in line_candidates:
            line_height = end_y - start_y + 1
            if line_height <= min_threshold:
                small_line_candidates.append((start_y, end_y))
            elif line_height >= max_threshold:
                big_line_candidates.append((start_y, end_y))
            else:
                new_lines.append((start_y, end_y))

        print("Small", small_line_candidates)
        print("Big", big_line_candidates)

        new_lines_avg_height = self._get_average_lines_height(new_lines)

        new_lines = self._join_small_candidates(
            new_lines, small_line_candidates, new_lines_avg_height)
        new_lines = self._separate_big_candidates(
            new_lines, big_line_candidates, new_lines_avg_height, h_proj)

        return new_lines

    def get_segments(self):
        image = self.get_image()
        height, width = image.shape[:2]

        h_proj = hist.horizontal_projection(image)

        # First of all, get perfectly separable segments (line candidates)
        space_candidates = hist.get_histogram_spaces(h_proj, 0)
        line_candidates = hist.get_histogram_peaks(h_proj, space_candidates)

        # Then filter out all candidates
        line_candidates = self._process_line_candidates(
            line_candidates, h_proj)
        print(len(line_candidates))

        # Idea took from https://content.sciendo.com/view/journals/amcs/27/1/article-p195.xml
        h_proj_smooth = hist.running_mean(h_proj, 5)
        hist_spaces = hist.get_histogram_spaces(
            h_proj_smooth, constants.NOISE_PIXELS_THRESHOLD)
        hist_peaks = hist.get_histogram_peaks(h_proj_smooth, hist_spaces)

        lines = self._filter_hist_peaks(h_proj_smooth, hist_peaks)

        # lines = self._include_typography(lines, image)
        blobs = self._strip_lines(line_candidates, image)
        line_images = []
        for (start_y, end_y, start_x, end_x) in blobs:

            roi_image = image[start_y:end_y + 1, start_x:end_x + 1]

            x_offset = self.get_x() + start_x
            y_offset = self.get_y() + start_y
            width = end_x - start_x + 1
            height = end_y - start_y + 1

            line_image = LineImage(
                roi_image, width, height, x_offset, y_offset)
            line_images.append(line_image)

        self.lines = line_images
        return line_images

    def _strip_lines(self, lines, image):
        blobs = []
        index = 0
        for (y1, y2) in lines:
            index += 1
            roi = image[y1:y2 + 1, :]
            v_proj = hist.vertical_projection(roi)
            x1, x2 = hist.blob_range(v_proj)
            if x1 >= 0 and x2 >= 0:
                blobs.append((y1, y2, x1, x2))

        return blobs

    def _include_typography(self, lines, max_index, avg_line_height):
        padded_lines = []
        for (y1, y2) in lines:
            line_height = y2 - y1 + 1
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
        line_heights = [y2 - y1 + 1 for (y1, y2) in lines]
        avg_line_height = sum(line_heights) / len(lines)

        line_height_treshold = avg_line_height * 0.9

        filtered_lines = []
        for (y1, y2) in lines:
            line_height = y2 - y1 + 1
            if line_height >= line_height_treshold:
                filtered_lines.append((y1, y2))

        return filtered_lines

    def _filter_small_lines(self, lines, height_threshold):
        filtered_lines = []
        for (y1, y2) in lines:
            line_height = y2 - y1 + 1
            if line_height > height_threshold:
                filtered_lines.append((y1, y2))

        return filtered_lines

    def _get_mean_peak_cords(self, hist, coords, mean):
        start_x, end_x = coords
        threshold = mean * constants.MEAN_HISTOGRAM_CUTOFF
        candidates = []

        x1, x2 = None, None
        for index in range(start_x, end_x + 1):
            value = hist[index]
            if index == end_x:
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
