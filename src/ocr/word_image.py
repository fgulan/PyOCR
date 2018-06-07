import cv2
import numpy as np
from ocr_image import OCRImage
from utils.helpers import debug_plot_array, debug_display_image
from utils import hist, constants
from char_image import CharImage
from skimage.morphology import skeletonize, thin
from skimage import img_as_ubyte


class WordImage(OCRImage):

    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4457222
    # use 1.5 of width for threshold
    MINIMUM_HEIGHT_WIDTH_RATIO = 1.2

    def __init__(self, image, width, height, x_offset=0, y_offset=0):
        super().__init__(image, width, height, x_offset, y_offset)

        self.characters = []

    def get_segments(self):
        image = self.get_image()

        v_proj = hist.vertical_projection(image)

        # First, get prefectly separable blobs
        hist_spaces = hist.get_histogram_spaces(v_proj, 0)
        hist_peaks = hist.get_histogram_peaks(v_proj, hist_spaces)

        # Then, filter out all small noise segments
        hist_peaks = hist.filter_histogram_peaks(hist_peaks, 2)

        # Check each segmented component for joined characters
        hist_peaks = self._process_joined_characters(image, hist_peaks)

        # Map peaks to char image
        self.characters = self._map_char_coords_to_object(image, hist_peaks)

        return self.characters

    def _process_joined_characters(self, image, char_coords):
        new_coords = []

        for coord_x in char_coords:
            candidates = self._segment_connected_components(image, coord_x)
            new_coords.extend(candidates)
        return new_coords

    def _segment_connected_components(self, image, char_coord_x):
        start_x, end_x = char_coord_x
        start_y, end_y = self._get_y_bounding_range(image, char_coord_x)

        roi_image = image[start_y:end_y + 1, start_x:end_x + 1]

        # Perform the connected components operation
        connectivity = 8
        output = cv2.connectedComponentsWithStats(
            roi_image, connectivity, cv2.CV_32S)
        # The first cell is the number of labels
        num_labels = output[0]
        # The third cell is the stat matrix
        stats = output[2]

        # Return if there is only a background and one object
        if num_labels <= 2:
            roi_width = end_x - start_x + 1
            roi_height = end_y - start_y + 1
            new_coords = [char_coord_x]

            if roi_width > self.MINIMUM_HEIGHT_WIDTH_RATIO * roi_height:
                new_coords = self._manually_separate_char(char_coord_x, roi_image)

            return new_coords

        # Skip first element, it is background label
        candidates = stats[1:]

        width = end_x - start_x + 1
        # If connected components return almost same image
        # then it is probably something unseparable like
        # croatian letter with diacritics or i or j ...

        # TODO: To sensitive to thin characters
        threshold_width = 0.9 * width
        too_big_comps = list(
            filter(lambda stat: stat[cv2.CC_STAT_WIDTH] >= threshold_width, candidates))
        if (len(too_big_comps) > 0):
            return [char_coord_x]

        # Filter out almost the same segments
        candidates = self._filter_near_segments(candidates)

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

    def _filter_near_segments(self, candidates):

        def overlap_length(min1, max1, min2, max2):
            return max(0, min(max1, max2) - max(min1, min2))
        
        def shortest_width(stat1, stat2):
            return min(stat1[cv2.CC_STAT_WIDTH], stat2[cv2.CC_STAT_WIDTH])

        def segment_position(stat):
            start_x = stat[cv2.CC_STAT_LEFT]
            end_x = start_x + stat[cv2.CC_STAT_WIDTH]
            return (start_x, end_x)

        indexes_to_remove = set()
        count = len(candidates)        
        threshold = 0.75

        # Sort candidates by area (biggest first) so that smallest ones
        # are filtered if centroids are on the same x position
        candidates = sorted(
            candidates, key=lambda info: info[cv2.CC_STAT_AREA], reverse=True)

        for index in range(count - 1):
            # Skip current index if it is stored to be removed
            if index in indexes_to_remove:
                continue

            current_stat = candidates[index]
            current_position = segment_position(current_stat)

            for next_index in range(index + 1, count):
                next_stat = candidates[next_index]
                next_position = segment_position(next_stat)

                # Append 1 for numerical stabiltiy (when overlaps are in zero)
                overlap = overlap_length(*current_position, *next_position) + 1
                min_width = shortest_width(current_stat, next_stat) + 1
                if overlap / min_width > threshold:
                    indexes_to_remove.add(next_index)

        new_candidates = []
        for index in range(count):
            if index in indexes_to_remove:
                continue
            candidate = candidates[index]
            new_candidates.append(candidate)

        return new_candidates

    def _map_char_coords_to_object(self, image, char_coords):
        word_start_y = self.get_y()
        word_start_x = self.get_x()

        chars = []
        for (start_x, end_x) in char_coords:
            vertical_image = image[:, start_x:end_x + 1]

            box_start_x, box_end_x, box_start_y, box_end_y = self._get_image_bounding_box(
                vertical_image)

            x_offset = word_start_x + start_x + box_start_x
            y_offset = word_start_y + box_start_y
            char_width = box_end_x - box_start_x + 1
            char_height = box_end_y - box_start_y + 1

            roi_image = vertical_image[box_start_y:box_end_y +
                                       1, box_start_x:box_end_x + 1]

            char = CharImage(roi_image, char_width,
                             char_height, x_offset, y_offset)
            chars.append(char)

        return chars


    def _manually_separate_char(self, char_coord_x, roi_image):
        if self._check_if_char_is_m(roi_image):
            return [char_coord_x]
        elif self._check_if_char_is_minus_sign(roi_image):
            return [char_coord_x]

        start_x, _ = char_coord_x
        height, _ = roi_image.shape[:2]
        # Remove cap part to remove possiblity of upper overlaping
        cap_start_index = int(0.15 * height)
        v_proj = hist.vertical_projection(roi_image[cap_start_index:,:])

        hist_spaces = hist.get_histogram_spaces(v_proj, 0)
        hist_peaks = hist.get_histogram_peaks(v_proj, hist_spaces)
        hist_peaks = hist.filter_histogram_peaks(hist_peaks, 2)

        if len(hist_peaks) == 1:
            return [char_coord_x]
        
        hist_peaks = hist.translate_points(hist_peaks, start_x)
        return hist_peaks

    def _check_if_char_is_m(self, roi_image):
        height, width = roi_image.shape[:2]
        thinned = img_as_ubyte(thin(roi_image))

        horizontal_line = thinned[height // 3, :]
        count = self._foreground_crossings_count(horizontal_line)
        if count != 3:
            return False

        horizontal_line = thinned[2 * height // 3, :]
        count = self._foreground_crossings_count(horizontal_line)
        if count != 3:
            return False

        vertical_line = thinned[:, width // 3]
        count = self._foreground_crossings_count(vertical_line)
        if count != 2 and count != 1:
            return False

        vertical_line = thinned[:, 2 * width // 3]
        count = self._foreground_crossings_count(vertical_line)
        if count != 2 and count != 1:
            return False

        return True
    
    def _check_if_char_is_minus_sign(self, roi_image):
        height, width = roi_image.shape[:2]
        horizontal_line = roi_image[height // 2, :]

        white_count = np.count_nonzero(horizontal_line)
        if white_count / width < 0.8:
            return False

        return True

    def _get_image_bounding_box(self, image):
        h_proj = hist.horizontal_projection(image)
        v_proj = hist.vertical_projection(image)

        min_x, max_x = hist.blob_range(v_proj)
        min_y, max_y = hist.blob_range(h_proj)

        return min_x, max_x, min_y, max_y

    def _foreground_crossings_count(self, values):
        return np.count_nonzero(values)

    def _get_y_bounding_range(self, image, char_coord_x):
        start_x, end_x = char_coord_x
        horizontal_char_image = image[:, start_x:end_x + 1]
        char_coord_y = self._get_char_vertical_range(horizontal_char_image)
        return char_coord_y

