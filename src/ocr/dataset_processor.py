import os
import cv2
import argparse
import uuid
import numpy as np
import pathlib

from filters.noise.noise_removal import NoiseRemoval
from filters.binarization import otsu, sauvola
from utils.helpers import load_image, debug_display_image, debug_plot_array
from utils.char_mapper import vocab_letter_to_class

from ocr_image import OCRImage
from text_image_v3 import TextImageBaseline
from scipy.ndimage import interpolation as inter

INVALID_WORD_FOLDER = "/Users/filipgulan/ds/invalid_words"


def draw_box(image, ocr_image):
    b_box = ocr_image.get_bounding_box()
    cv2.rectangle(image,
                  (b_box['x'] - 1, b_box['y'] - 1),
                  (b_box['x'] + b_box['width'] - 1,
                   b_box['y'] + b_box['height'] - 1),
                  (255, 0, 0), 1)


def process_image(image_path):
    input_image = load_image(image_path)

    binarizer = otsu.OtsuBinarization()
    binarized_img = binarizer.process(input_image)
    binarized_img = cv2.bitwise_not(binarized_img)

    denoiser = NoiseRemoval()
    input_image = denoiser.process(binarized_img)

    height, width = input_image.shape[:2]
    ocr_image = OCRImage(input_image, width, height)
    angle = ocr_image.fix_skew()

    rotated = inter.rotate(input_image, angle, reshape=False, order=0)
    original_image = cv2.cvtColor(rotated, cv2.COLOR_GRAY2RGB)
    
    roi_image, width, height, min_x, min_y = ocr_image.get_segments()
    text_image = TextImageBaseline(roi_image, width, height, min_x, min_y)
    return text_image, original_image


def process_text_file(file_path):
    lines = []
    with open(file_path) as text_file:
        lines = text_file.read().splitlines()
    lines = list(map(lambda line: line.strip(), lines))
    lines = list(filter(None, lines))
    return lines


def get_words(file_line):
    words = file_line.split(" ")
    words = list(map(lambda text: text.strip(), words))
    words = list(filter(None, words))
    return words


def process_chars(ocr_chars, file_chars, root_output_folder, avg_line_height):
    for ocr_char, file_char in zip(ocr_chars, file_chars):
        scaled_image = ocr_char.get_scaled_image(avg_line_height)

        file_name = str(uuid.uuid4()) + ".jpg"
        letter_class = vocab_letter_to_class[file_char]
        output_folder = os.path.join(root_output_folder, letter_class)
        output_file = os.path.join(output_folder, file_name)
        pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_file, scaled_image)


def process_words(ocr_words, file_words, root_output_folder, line_index, avg_line_height):
    for ocr_word, file_word in zip(ocr_words, file_words):
        ocr_chars = ocr_word.get_segments()
        file_chars = list(file_word)

        if len(ocr_chars) != len(file_chars):
            # print("Neispravan broj znakova u rijeci: " + file_word, "Linija:", str(line_index + 1))
            # print("ocr_chars:", len(ocr_chars), "file_chars", len(file_chars))
            file_name = str(uuid.uuid4()) + ".jpg"
            output_path = os.path.join(INVALID_WORD_FOLDER, file_name)
            ocr_word.save(output_path)
            continue

        process_chars(ocr_chars, file_chars,
                      root_output_folder, avg_line_height)


def process(args):
    file_lines = process_text_file(args.text_file)
    text_image, original_image = process_image(args.image)

    text_lines = text_image.get_segments()
    if len(text_lines) != len(file_lines):
        print("Neispravan broj linija!")
        return

    avg_line_height = 0
    for line in text_lines:
        avg_line_height += line.get_height()
    avg_line_height /= len(text_lines)
    print(avg_line_height)

    all_words = []
    for index, (ocr_line, file_line) in enumerate(zip(text_lines, file_lines)):
        ocr_words = ocr_line.get_segments()
        file_words = get_words(file_line)

        all_words.extend(ocr_words)
        if len(ocr_words) == len(file_words):
            process_words(ocr_words, file_words, args.output,
                          index, avg_line_height)
        else:
            print("Neispravan broj rijeci na liniji " + str(index + 1))
            continue

    export_words_image(original_image, all_words)


def export_words_image(out_image, ocr_words):
    for ocr_word in ocr_words:
        draw_box(out_image, ocr_word)

    cv2.imwrite("sample_words.jpg", out_image)


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
