import os
import cv2
import numpy as np
from filters.noise.noise_removal import NoiseRemoval
from filters.binarization import otsu, sauvola, adaptive_gauss
from utils.helpers import load_image, debug_display_image, debug_plot_array

from ocr_image import OCRImage
from text_image_v2 import TextImageBaseline

input_image = load_image("../data/uvod.jpg")
binarizer = otsu.OtsuBinarization()
binarized_img = binarizer.process(input_image)
binarized_img = cv2.bitwise_not(binarized_img)
denoiser = NoiseRemoval()
input_image = denoiser.process(binarized_img)
height, width = input_image.shape[:2]
ocr_image = OCRImage(input_image, width, height)
angle = ocr_image.fix_skew()


roi_image, width, height, min_x, min_y = ocr_image.get_segments()
print(width, height)
text_image = TextImageBaseline(roi_image, width, height, min_x, min_y)
text_image.save("./out.jpg")
lines = text_image.get_segments()


chars_count = 0
words_count = 0
line_count = 0
print("Broj linija", len(lines))
for line in lines:
    line.save("../lines/" + str(line_count) + ".jpg")
#     debug_display_image(line.get_image())
    line_count += 1
    words = line.get_segments()

    for word in words:
        # debug_display_image(word.get_image())
        word.save("../words/" + str(words_count) + ".jpg")
        chars = word.get_segments()
        words_count += 1
        for char in chars:
            char.save("../chars/" + str(chars_count) + ".jpg")
            chars_count += 1
            # debug_display_image(char.get_image())

print("Broj rijeci", words_count)
print("Broj slova", chars_count)

        # word.save("./out/" + str(index) + ".jpg")
        # debug_display_image(word.get_image())