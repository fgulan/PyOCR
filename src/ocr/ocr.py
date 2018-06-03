import os
import cv2
import argparse
import numpy as np

from filters.noise.noise_removal import NoiseRemoval
from filters.binarization import otsu
from utils.helpers import load_image, debug_display_image, debug_plot_array
from utils.char_mapper import classifier_out_to_vocab_letter
from ocr_image import OCRImage
from text_image_v3 import TextImageBaseline

from keras.models import model_from_json


def prepare_model(model_path, weights_path):

    with open(model_path, 'r') as model_file:
        loaded_model_json = model_file.read()

    model = model_from_json(loaded_model_json)
    model.load_weights(weights_path)

    return model


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


def predict(ocr_char, model, word):
    input_image = ocr_char.get_scaled_image()
    scaled_input = input_image / 255.0
    tensor = scaled_input.reshape(1, *scaled_input.shape, 1)
    output = model.predict(tensor)
    output_index = np.argmax(output)
    return classifier_out_to_vocab_letter(output_index)


def process(args):
    text_image = process_image(args.image)
    ocr_lines = text_image.get_segments()
    lines = []
    model = prepare_model(args.model, args.weights)

    for ocr_line in ocr_lines:
        print("Line")
        ocr_words = ocr_line.get_segments()
        words = []
        for ocr_word in ocr_words:
            ocr_chars = ocr_word.get_segments()
            chars = []
            for ocr_char in ocr_chars:
                predicted_symbol = predict(ocr_char, model, chars)
                chars.append(predicted_symbol)
            word = "".join(chars)
            words.append(word)
        line = " ".join(words)
        lines.append(line)

    output_text = "\n".join(lines)
    with open(args.output, 'a') as output_file:
        output_file.write(output_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image', help="Input text image", required=True, type=str)
    parser.add_argument(
        '--model', help="Keras model", required=True, type=str)
    parser.add_argument(
        '--weights', help="Keras model weights", required=True, type=str)
    parser.add_argument(
        '--output', help="Output text file", required=True, type=str)
    args = parser.parse_args()
    process(args)


if __name__ == '__main__':
    main()
