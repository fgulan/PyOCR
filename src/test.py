import os
import cv2
import numpy as np
from filters.noise.noise_removal import NoiseRemoval
from filters.binarization import otsu, sauvola, adaptive_gauss
from utils.helpers import load_image, debug_display_image, debug_plot_array

from ocr_image import OCRImage
from text_image import TextImage
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
text_image = TextImageBaseline(roi_image, width, height, min_x, min_y)
lines = text_image.get_segments()


for line in lines:
    debug_display_image(line.get_image())
    # words = line.get_segments()
    # for word in words: