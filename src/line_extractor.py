import os
import cv2
import numpy as np
import binarizer
import imutils

from utils import hist
from utils.helpers import _debug_display_image, _debug_plot_array

def process(image):
    h_proj = hist.horizontal_projection(image)
    h_proj_th = np.mean(h_proj) * 0.01 # 5% of possible noise
    height, _ = image.shape[:2]

    upper, lower = None, None
    lines = []
    for index, h_line in enumerate(h_proj):
        if index == height - 1:
            if upper != None and lower != None:
                lines.append((upper, lower))

        if h_line > h_proj_th:
            if upper == None:
                upper = index
                lower = index
            else:
                lower = index
        else:
            if upper != None and lower != None:
                lines.append((upper, lower))
            lower = None
            upper = None
                
    return lines

def process_debug(image):
    lines = process(image)
    _, width = image.shape[:2]

    out_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for y, x in lines:
        cv2.line(out_image, (0,y), (width, y), (255,255,255), 1)

    for x, y in lines:
        cv2.line(out_image, (0,y), (width, y), (0,255,0), 1)

    cv2.imwrite("debug_result.png", out_image)

def main():
    input_image = cv2.imread('../data/intro.jpg', cv2.IMREAD_COLOR)
    grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    preprocessed_image = binarizer.preprocess(grayscale_image)
    binarized_image = binarizer.process(preprocessed_image)
    eroded_image = binarizer.erode(binarized_image, kernel_size=(7,7))
    inverted_image = cv2.bitwise_not(eroded_image)
    process_debug(inverted_image)
    cv2.imwrite("result.png", inverted_image)

if __name__ == '__main__':
    main()
