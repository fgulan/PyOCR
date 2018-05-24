import os
import cv2
import numpy as np
from filters.binarization.adaptive_gauss import GaussBinarization

from utils.helpers import load_image, debug_display_image, debug_plot_array

input_image = load_image("../data/test.jpg")

height, width = input_image.shape
min_dim = min(height, width)
block_size = int(min_dim * 0.1)
if block_size % 2 == 0:
    block_size += 1
sauvola_bin = GaussBinarization(block_size=block_size, C=60)
out_image = sauvola_bin.process(input_image)
debug_display_image(out_image)