import os
import cv2
import numpy as np
from filters.noise.noise_removal import NoiseRemoval
from filters.binarization import otsu, sauvola, adaptive_gauss
from utils.helpers import load_image, debug_display_image, debug_plot_array
from utils import hist
from ocr_image import OCRImage
from text_image_v3 import TextImageBaseline
from scipy.ndimage import interpolation as inter
from matplotlib import pyplot as plt

def draw_box(image, ocr_image):
    b_box = ocr_image.get_bounding_box()
    cv2.rectangle(image, 
                    (b_box['x'] - 1, b_box['y'] - 1),
                    (b_box['x'] + b_box['width'] - 1, b_box['y'] + b_box['height'] - 1), 
                    (180, 119, 31), 2)

input_image = load_image("/Users/filipgulan/Desktop/test_set/comic.jpg")
# input_image = load_image("../../data/oversegm.jpg")
# input_image = load_image("../../data/index.jpg")
# plt.hist(input_image.ravel(),256,[0,256])
# plt.ylabel('Brojnost slikovnog elementa')
# plt.xlabel('Slikovni element')
# plt.savefig("histogram.pdf", format='pdf')
# plt.show()

# input_image = load_image("../../data/dataset_docs/calibri_12.jpg") 
# input_image = load_image("../../data/dataset_docs/calibri_12_bold.jpg")
# input_image = load_image("../../data/dataset_docs/tnr_12_bold.jpg")
# input_image = load_image("../../data/dataset_docs/tnr_12.jpg")
# input_image = load_image("../../data/dataset_docs/arial_12.jpg")
# input_image = load_image("../../data/dataset_docs/arial_12_bold.jpg")
# input_image = load_image("../../data/dataset_docs/helvetica_12.jpg")
# input_image = load_image("../../data/dataset_docs/helvetica_12_bold.jpg")
# input_image = load_image("../data/dataset_docs/verdana_12.jpg")
# input_image = load_image("../data/dataset_docs/verdana_12_bold.jpg")
# input_image = load_image("../data/dataset_docs/comic_12.jpg")
# input_image = load_image("../data/dataset_docs/comic_12_bold.jpg")
# input_image = load_image("../data/dataset_docs/cen_gothic_12.jpg")
# input_image = load_image("../data/dataset_docs/cen_gothic_12_bold.jpg")
# input_image = load_image("../../data/dataset_docs/sans_serif_12_italic.jpg")
# input_image = load_image("/Users/filipgulan/ds/300dpi/temp_5.jpg")

orig_image = input_image.copy()
binarizer = sauvola.SauvolaBinarization()
binarized_img = binarizer.process(input_image)



binarized_img = cv2.bitwise_not(binarized_img)
denoiser = NoiseRemoval()
input_image = denoiser.process(binarized_img)
height, width = input_image.shape[:2]
ocr_image = OCRImage(input_image, width, height)
angle = ocr_image.fix_skew()

rotated = inter.rotate(input_image, angle, reshape=False, order=0)
# debug_display_image(rotated)
cv2.imwrite("rotated.jpg", rotated)
backtorgb = cv2.cvtColor(rotated, cv2.COLOR_GRAY2RGB)




# debug_plot_array(h_proj)

denoiser.process(rotated)
roi_image, width, height, min_x, min_y = ocr_image.get_segments()
h_proj = hist.horizontal_projection(roi_image)
x = np.array(range(len(h_proj)))
plt.plot(h_proj, x)

plt.ylim(0, len(h_proj))  # decreasing time
plt.gca().invert_yaxis()
plt.xlabel('Broj slikovnih elemenata teksta')
plt.ylabel('Linija')

# plt.show()
print(width, height)
text_image = TextImageBaseline(roi_image, width, height, min_x, min_y)
lines = text_image.get_segments()


chars_count = 0
words_count = 0
line_count = 0
line_hg = 0
# draw_box(backtorgb, text_image)
print("Broj linija", len(lines))
for line in lines:
    line_count += 1
    words = line.get_segments()
    line_hg += line.get_height()
    # draw_box(backtorgb, line)
    for word in words:
        
        chars = word.get_segments()
        words_count += 1
        # draw_box(backtorgb, word)
        for char in chars:
            draw_box(backtorgb, char)
            chars_count += 1

print("Broj rijeci", words_count)
print("Broj slova", chars_count)
print("Line hg", line_hg / line_count)
cv2.imwrite("backtorgb.jpg", backtorgb)

