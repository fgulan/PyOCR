import cv2
import numpy as np
from cv2 import cvtColor, threshold, findContours, contourArea, drawContours, GaussianBlur
from extract_text import binarize_image_cool
import random

def binarize_image(image):
    image = cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image

def extract_receipt_contour_box(bin_image):
    contours = findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    max_c = max(contours, key=cv2.contourArea)
    left = tuple(max_c[max_c[:, :, 0].argmin()][0])
    right = tuple(max_c[max_c[:, :, 0].argmax()][0])
    top = tuple(max_c[max_c[:, :, 1].argmin()][0])
    bottom = tuple(max_c[max_c[:, :, 1].argmax()][0])
    min_x, max_x = left[0], right[0]
    min_y, max_y = top[1], bottom[1]

    return max_c, (min_x, max_x), (min_y, max_y)

def mask_image(image, contour, x_axis, y_axis):
    mask = np.full_like(image, 255)
    cv2.drawContours(mask, [contour], -1, 0, -1)
    image_out = np.full_like(image, 255)
    image_out[mask == 0] = image[mask == 0]
    image_out = image_out[y_axis[0]:y_axis[1], x_axis[0]:x_axis[1]]
    return image_out

def crop_upper_part(image, percent=0.4):
    height, _, _ = image.shape
    point = int(percent * height)
    return image[0:point,:]

def crop_downside_part(image, percent=0.6):
    height, _, _ = image.shape
    distance = 1 - percent
    start_point = int(distance * height)
    return image[start_point:height,:]

def decision(probability):
    return random.random() < probability

def adjust_contrast(image, probability=1.0):
    if not decision(probability):
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def adjust_gamma(image, gamma=0.8):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def adjust_brightness(image, probability=1.0, interval=(0 ,1)):
    if not decision(probability):
        return image
    gamma = random.uniform(*interval)
    return adjust_gamma(image, gamma)

def adjust_blur(image, probability=1.0):
    if not decision(probability):
        return image
    return GaussianBlur(image, (5, 5), 0)

img = cv2.imread('/Users/filipgulan/Downloads/test/Dataset/0.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.BRISK_create()
(kps, descs) = sift.detectAndCompute(gray, None)

img=cv2.drawKeypoints(gray,kps,img)

cv2.imshow('b_image_out', img)
cv2.waitKey(0)
cv2.destroyAllWindows()