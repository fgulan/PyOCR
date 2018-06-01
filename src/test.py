import os
import cv2
import numpy as np
from filters.noise.noise_removal import NoiseRemoval
from filters.binarization import otsu, sauvola, adaptive_gauss
from utils.helpers import load_image, debug_display_image, debug_plot_array

from ocr_image import OCRImage
from text_image_v3 import TextImageBaseline
import imutils

input_image = load_image("../data/uvod.jpg")
orig_image = input_image.copy()
binarizer = otsu.OtsuBinarization()
binarized_img = binarizer.process(input_image)
binarized_img = cv2.bitwise_not(binarized_img)
denoiser = NoiseRemoval()
input_image = denoiser.process(binarized_img)
height, width = input_image.shape[:2]
ocr_image = OCRImage(input_image, width, height)
angle = ocr_image.fix_skew()

rotated = imutils.rotate(orig_image, angle)
backtorgb = cv2.cvtColor(rotated, cv2.COLOR_GRAY2RGB)

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
    b_box = line.get_bounding_box()
    cv2.rectangle(backtorgb, 
                    (b_box['x'] - 1, b_box['y'] - 1),
                    (b_box['x'] + b_box['width'] - 1, b_box['y'] + b_box['height'] - 1), 
                    (255, 0, 0), 2)
    line_count += 1
    words = line.get_segments()
    for word in words:
        word.save("../words/" + str(words_count) + ".jpg")
        chars = word.get_segments()
        words_count += 1

        for char in chars:
            char.save("../chars/" + str(chars_count) + ".jpg")
            chars_count += 1

print("Broj rijeci", words_count)
print("Broj slova", chars_count)
cv2.imwrite("backtorgb.jpg", backtorgb)

