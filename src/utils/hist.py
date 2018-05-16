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

