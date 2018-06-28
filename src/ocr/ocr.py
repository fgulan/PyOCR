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

from keras.models import model_from_json, Model
from models import OCRModel


def prepare_model(weights_path):
    predictions, inputs = OCRModel((*(40, 40), 1), 67)
    model = Model(inputs=inputs, outputs=predictions)
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

def capitalize(word):
    return word[0].upper() + word[1:] if len(word) > 0 else word

def ends_with_stop_char(input_word):
    return input_word.endswith('.') or input_word.endswith('!') or input_word.endswith('?')

def process(image_path, output_path, model):
    text_image = process_image(image_path)
    ocr_lines = text_image.get_segments()
    lines = []

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
            
            # Capitalize word
            if len(words) > 0:
                if ends_with_stop_char(words[-1]):
                    word = capitalize(word)
            
            words.append(word)

        line = " ".join(words)
        # Capitalize line
        if len(lines) > 0:
            if ends_with_stop_char(lines[-1]):
                line = capitalize(line)
        else:
            line = capitalize(line)

        lines.append(line)

    output_text = "\n".join(lines)
    with open(output_path, 'a') as output_file:
        output_file.write(output_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image', help="Input text image", required=True, type=str)
    parser.add_argument(
        '--weights', help="Keras model weights", required=True, type=str)
    parser.add_argument(
        '--output', help="Output text file", required=True, type=str)
    args = parser.parse_args()
    model = prepare_model(args.weights)
    process(args.image, args.output, model)


if __name__ == '__main__':
    main()
