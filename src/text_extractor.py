import os
import cv2
import numpy as np
import binarizer
import imutils
import binarizer, text_extractor

from utils import hist
from utils.helpers import _debug_display_image, _debug_plot_array

def process(image):
    h_proj = hist.horizontal_projection(image)
    v_proj = hist.vertical_projection(image)

    min_y, max_y = hist.blob_range(h_proj)
    min_x, max_x = hist.blob_range(v_proj)

    return min_y, max_y, min_x, max_x

def process_debug(image):
    min_y, max_y, min_x, max_x = process(image)
    return image[min_y:max_y, min_x:max_x]

def main():
    input_image = cv2.imread('../data/uvod.jpg', cv2.IMREAD_GRAYSCALE)
    input_image = cv2.medianBlur(input_image, 3)
    input_image = binarizer.process(input_image)
    input_image = cv2.bitwise_not(input_image)

    kernel = np.ones((3, 3) , np.uint8)
    input_image = cv2.dilate(input_image, kernel, iterations = 1)
    
    output_image = process_debug(input_image)

    _debug_display_image(output_image)
    cv2.imwrite('./datae.jpg', cv2.bitwise_not(output_image))

if __name__ == '__main__':
    main()
