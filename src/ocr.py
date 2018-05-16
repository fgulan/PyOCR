import os
import cv2
import argparse
import numpy as np
import binarizer, text_extractor

from utils.helpers import _debug_display_image, _debug_plot_array

def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return image

def mask_image(image, blob_area):
    min_y, max_y, min_x, max_x = blob_area
    return image[min_y:max_y, min_x:max_x]

def process(args):
    input_image = load_image(args.image)
    preprocessed_image = binarizer.preprocess(input_image)
    binarized_image = binarizer.process(preprocessed_image)
    eroded_image = binarizer.erode(binarized_image)
    inverted_image = cv2.bitwise_not(eroded_image)

    blob_area = text_extractor.process(inverted_image)

    text_image = mask_image(input_image, blob_area)

    if (args.output):
        cv2.imwrite(args.output, text_image)
    else:
        _debug_display_image(text_image)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image', help="Input text image", required=True, type=str)
    parser.add_argument(
        '--output', help="Output text file", required=False, type=str)
    parser.add_argument(
        '--grayscale', help="Input image is grayscale", action='store_true', required=False)
    args = parser.parse_args()
    process(args)


if __name__ == '__main__':
    main()
