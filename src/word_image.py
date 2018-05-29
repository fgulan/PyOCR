import cv2
import numpy as np
from ocr_image import OCRImage
from utils.helpers import debug_plot_array, debug_display_image
from utils import hist, constants
from char_image import CharImage


class WordImage(OCRImage):

    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4457222
    # use 1.1 of width for threshold
    MINIMUM_HEIGHT_WIDTH_RATIO = 1.1
    BASELINE_CUTOFF_RATIO = 0.85

    def __init__(self, image, width, height, x_offset=0, y_offset=0):
        super().__init__(image, width, height, x_offset, y_offset)

        self.characters = []

    def get_segments(self):
        image = self.get_image()
        height, _ = image.shape[:2]

        # Remove baseline for easier character segmentation
        median_height = height // (1 + constants.CAP_HEIGHT +
                                   constants.BASELINE_HEIGHT)
        baseline_height = int(
            median_height * constants.BASELINE_HEIGHT * self.BASELINE_CUTOFF_RATIO)

        v_proj = hist.vertical_projection(
            image[0:height - baseline_height, :])
        v_proj_smooth = v_proj  # hist.running_mean(v_proj, 3)

        hist_spaces = hist.get_histogram_spaces(v_proj_smooth, 0)
        hist_peaks = hist.get_histogram_peaks(v_proj_smooth, hist_spaces)
        hist_peaks = hist.filter_histogram_peaks(hist_peaks, 2)
        hist_peaks = self._process_joined_characters(image, hist_peaks)
        self.characters = self._map_char_coords_to_object(image, hist_peaks)

        return self.characters

    def _process_joined_characters(self, image, char_coords):
        new_coords = []

        for coord_x in char_coords:
            start_x, end_x = coord_x
            horizontal_char_image = image[:, start_x:end_x + 1]
            coord_y = start_y, end_y = self._get_char_vertical_range(
                horizontal_char_image)

            width = end_x - start_x + 1
            height = end_y - start_y + 1
            if width > self.MINIMUM_HEIGHT_WIDTH_RATIO * height:
                candidates = self._segment_connected_components(
                    image, coord_x, coord_y)
                new_coords.extend(candidates)
            else:
                new_coords.append(coord_x)

        return new_coords

    def _segment_connected_components(self, image, char_coord_x, char_coord_y):
        start_x, end_x = char_coord_x
        start_y, end_y = char_coord_y
        roi_image = image[start_y:end_y + 1, start_x:end_x + 1]

        # Perform the operation
        connectivity = 8
        output = cv2.connectedComponentsWithStats(
            roi_image, connectivity, cv2.CV_32S)
        # The first cell is the number of labels
        num_labels = output[0]
        # The third cell is the stat matrix
        stats = output[2]

        # Return if there is only a background and one object
        if num_labels <= 2:
            return [char_coord_x]

        # Skip first element, it is background label
        candidates = stats[1:]
        # The fourth cell is the centroid matrix
        centroids = output[3][1:]

        width = end_x - start_x + 1
        # If connected components return almost same image
        # then it is probably something unseparable, return back
        threshold_width = 0.95 * width
        too_big_comps = list(
            filter(lambda stat: stat[cv2.CC_STAT_WIDTH] >= threshold_width, candidates))
        if (len(too_big_comps) > 0):
            return [char_coord_x]

        # Filter out almost the same segments
        candidates = self._filter_near_segments(candidates, centroids)

        # Sort them by start_x value
        candidates = sorted(
            candidates, key=lambda stat: stat[cv2.CC_STAT_LEFT])
        # Filter noise stats
        candidates = list(
            filter(lambda stat: stat[cv2.CC_STAT_WIDTH] > 4, candidates))

        new_peaks = []
        for stat in candidates:
            blob_start_x = start_x + stat[cv2.CC_STAT_LEFT]
            blob_end_x = blob_start_x + stat[cv2.CC_STAT_WIDTH] - 1
            new_peaks.append((blob_start_x, blob_end_x))

        return new_peaks

    def _get_char_vertical_range(self, image):
        h_proj = hist.horizontal_projection(image)
        start_y, end_y = hist.blob_range(h_proj)
        return start_y, end_y

    def _filter_near_segments(self, candidates, centroids):
        indexes_to_remove = set()
        count = len(candidates)
        merged_infos = zip(candidates, centroids)
        threshold = 0.9
        # Sort candidates/centroids by area (biggest first) so that smallest ones
        # are filtered if centroids are on the same x position
        merged_infos = sorted(
            merged_infos, key=lambda info: info[0][cv2.CC_STAT_AREA], reverse=True)

        for index in range(count - 1):

            # Skip current index if it is stored to be removed
            if index in indexes_to_remove:
                continue

            _, (current_centroid_x, _) = merged_infos[index]
            for next_index in range(index + 1, count):
                _, (next_centroid_x, _) = merged_infos[next_index]
                dividend = min(current_centroid_x, next_centroid_x)
                divisor = max(current_centroid_x, next_centroid_x)
                if dividend/divisor > threshold:
                    indexes_to_remove.add(next_index)

        new_candidates = []
        for index in range(count):
            if index in indexes_to_remove:
                continue
            candidate, _ = merged_infos[index]
            new_candidates.append(candidate)

        return new_candidates

    def _map_char_coords_to_object(self, image, char_coords):
        char_height = self.get_height()
        start_y = self.get_y()
        word_start_x = self.get_x()

        chars = []
        for (start_x, end_x) in char_coords:
            char_width = end_x - start_x + 1
            roi_image = image[:, start_x:end_x + 1]
            x_offset = word_start_x + start_x
            char = WordImage(roi_image, char_width,
                             char_height, x_offset, start_y)
            chars.append(char)
            
        return chars