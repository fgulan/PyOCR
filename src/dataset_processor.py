import os
import cv2
import argparse
import numpy as np
from filters.noise.noise_removal import NoiseRemoval
from filters.binarization import otsu
from utils.helpers import load_image, debug_display_image, debug_plot_array

from ocr_image import OCRImage
from text_image_v2 import TextImageBaseline
import imutils

def process_image(image_path):
    input_image = load_image(image_path)

    binarizer = otsu.OtsuBinarization()
    binarized_img = binarizer.process(input_image)
    binarized_img = cv2.bitwise_not(binarized_img)
    
    denoiser = NoiseRemoval()
    binarized_img = denoiser.process(binarized_img)
    
    height, width = input_image.shape[:2]
    ocr_image = OCRImage(input_image, width, height)
    ocr_image.fix_skew()

    roi_image, width, height, min_x, min_y = ocr_image.get_segments()
    text_image = TextImageBaseline(roi_image, width, height, min_x, min_y)

    return text_image

def process_text_file(file_path):
    with open(file_path) as text_file:
        lines = text_file.read().splitlines()
    print(lines)

def process(args):
    process_text_file(args.text_file)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image', help="Input text image", required=True, type=str)
    parser.add_argument(
        '--text-file', help="Input text file", required=True, type=str)
    parser.add_argument(
        '--output-folder', help="Input image is grayscale", required=True, type=str)
    args = parser.parse_args()
    process(args)


if __name__ == '__main__':
    main()
