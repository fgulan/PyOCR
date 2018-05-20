import os
import cv2
import numpy as np
import binarizer
import imutils

from utils import hist
from utils.helpers import _debug_display_image, _debug_plot_array

LINE_SEPARATION_NOISE_PERCENTAGE = 0.01
LINE_PADDING_PRECENTAGE = 0.15

def _process_padding(lines, image):
    padding_precentage = 1 + LINE_PADDING_PRECENTAGE
    image_height, _ = image.shape[:2]
    max_index = image_height - 1

    new_lines = []
    for (y1, y2) in lines:
        line_height = y2 - y1
        padded_line_height = line_height * padding_precentage
        half_distance = round((padded_line_height - line_height) / 2)
        y1 = max(0, y1 - half_distance)
        y2 = min(max_index, y2 + half_distance)
        new_lines.append((y1, y2))

    return np.array(new_lines)

def process(image):
    h_proj = hist.horizontal_projection(image)
    h_proj_th = np.mean(h_proj) * LINE_SEPARATION_NOISE_PERCENTAGE
    height, _ = image.shape[:2]

    y1, y2 = None, None
    lines = []
    for index, h_line in enumerate(h_proj):
        if index == height - 1:
            if y1 != None and y2 != None:
                lines.append((y1, y2))

        if h_line > h_proj_th:
            if y1 == None:
                y1 = index
                y2 = index
            else:
                y2 = index
        else:
            if y1 != None and y2 != None:
                lines.append((y1, y2))
            y2 = None
            y1 = None

    lines = _process_padding(lines, image)
                
    return lines

def process_debug(image, original_image):
    lines = process(image)
    _, width = image.shape[:2]

    out_image = original_image
    for x, y in lines:
        cv2.line(out_image, (0,x), (width, x), (255,0,0), 1)
        cv2.line(out_image, (0,y), (width, y), (0,255,0), 1)

    cv2.imwrite("debug_result.png", out_image)

def main():
    input_image = cv2.imread('../data/intro.jpg', cv2.IMREAD_COLOR)
    grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    preprocessed_image = binarizer.preprocess(grayscale_image)
    binarized_image = binarizer.process(preprocessed_image)
    eroded_image = binarizer.erode(binarized_image, kernel_size=(7,7))
    inverted_image = cv2.bitwise_not(eroded_image)
    process_debug(inverted_image, input_image)

if __name__ == '__main__':
    main()
