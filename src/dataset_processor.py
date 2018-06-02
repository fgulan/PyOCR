import os
import cv2
import argparse
import numpy as np
from filters.noise.noise_removal import NoiseRemoval
from filters.binarization import otsu
from utils.helpers import load_image, debug_display_image, debug_plot_array

from ocr_image import OCRImage
from text_image_v3 import TextImageBaseline
import imutils

def process_image(image_path):
    input_image = load_image(image_path)

    binarizer = otsu.OtsuBinarization()
    binarized_img = binarizer.process(input_image)
    binarized_img = cv2.bitwise_not(binarized_img)

    denoiser = NoiseRemoval()
    input_image = denoiser.process(binarized_img)

    height, width = input_image.shape[:2]
    ocr_image = OCRImage(input_image, width, height)
    ocr_image.fix_skew()

    roi_image, width, height, min_x, min_y = ocr_image.get_segments()
    text_image = TextImageBaseline(roi_image, width, height, min_x, min_y)
    return text_image

def process_text_file(file_path):
    lines = []
    with open(file_path) as text_file:
        lines = text_file.read().splitlines()
    return lines

def get_words(file_line):
    words = file_line.split(" ")
    words = list(map(lambda text: text.strip(), words))
    words = list(filter(None, words))
    return words

def process_words(ocr_words, file_words):
    for index, (ocr_word, file_word) in enumerate(zip(ocr_words, file_words)):
        ocr_chars = ocr_word.get_segments()
        file_chars = list(file_word)

        if len(ocr_chars) != len(file_chars):

            print("Neispravan broj znakova u rijeci: " + file_word)
            print("ocr_chars:", len(ocr_chars), "file_chars", len(file_chars))
            continue



def process(args):
    file_lines = process_text_file(args.text_file)
    text_image = process_image(args.image)

    text_lines = text_image.get_segments()
    if len(text_lines) != len(file_lines):
        print("Neispravan broj linija!")
        return

    for index, (ocr_line, file_line) in enumerate(zip(text_lines, file_lines)):        
        ocr_words = ocr_line.get_segments()
        file_words = get_words(file_line)

        if len(ocr_words) == len(file_words):
            process_words(ocr_words, file_words)
        else:
            import pdb; pdb.set_trace()
            print("Neispravan broj rijeci na liniji " + str(index + 1))
            continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image', help="Input text image", required=True, type=str)
    parser.add_argument(
        '--text-file', help="Input text file", required=True, type=str)
    parser.add_argument(
        '--output', help="Input image is grayscale", required=True, type=str)
    args = parser.parse_args()
    process(args)


if __name__ == '__main__':
    main()
