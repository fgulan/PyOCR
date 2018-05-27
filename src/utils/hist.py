import cv2
import numpy as np


def vertical_projection(image):
    return cv2.reduce(image, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S).ravel() / 255


def horizontal_projection(image):
    return cv2.reduce(image, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S).ravel() / 255


def blob_range(projection):
    indices = np.nonzero(projection)[0]
    blob = (indices[0], indices[-1]) if len(indices) > 0 else (-1, -1)
    return blob


def get_histogram_spaces(hist, threshold):
    spaces = []
    start_x, end_x = None, None

    for index, value in enumerate(hist):
        if value <= threshold:
            if start_x == None:
                start_x = index
            end_x = index
        else:
            if start_x != None and end_x != None:
                spaces.append((start_x, end_x))
            start_x = None
            end_x = None

    return spaces


def get_histogram_peaks(hist, spaces):
    size = len(hist)

    if len(spaces) == 0:
        return [(0, size)]

    current_x = 0
    hist_peaks = []
    for space in spaces:
        space_start_x, space_end_x = space
        section = (current_x, space_start_x)
        hist_peaks.append(section)
        current_x = space_end_x

    if current_x < size - 1:
        section = (current_x, size - 1)
        hist_peaks.append(section)

    return hist_peaks


def filter_histogram_peaks(hist_peaks, threshold):
    new_peaks = []
    
    for start_x, end_x in hist_peaks:
        if end_x - start_x >= threshold:
            new_peaks.append((start_x, end_x))
        
    return new_peaks


def get_histogram_peak_means(hist, peaks):
    peaks_mean = []

    for (x1, x2) in peaks:
        roi_hist = hist[x1:x2]
        peak_mean = np.mean(roi_hist)
        peaks_mean.append(peak_mean)

    return peaks_mean


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)
