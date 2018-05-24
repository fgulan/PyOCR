import os
import cv2
import numpy as np
from filters.binarization.niblack import NiblackBinarization

from utils.helpers import load_image, debug_display_image, debug_plot_array

input_image = load_image("../data/test.jpg")

sauvola_bin = NiblackBinarization(window_size=31, k=0.03)
out_image = sauvola_bin.process(input_image)
debug_display_image(out_image)