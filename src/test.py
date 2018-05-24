import os
import cv2
import numpy as np
from filters.noise.noise_removal import NoiseRemoval
from filters.binarization import otsu, sauvola, adaptive_gauss

from utils.helpers import load_image, debug_display_image, debug_plot_array

input_image = load_image("../data/testaa.jpg")
binarizer = otsu.OtsuBinarization()
binarized_img = binarizer.process(input_image)
binarized_img = cv2.bitwise_not(binarized_img)
denoiser = NoiseRemoval()
out_image = denoiser.process(binarized_img)


debug_display_image(out_image)