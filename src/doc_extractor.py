import os
import cv2
import numpy as np
import binarizer
import imutils

def extract_external_contour(image):
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if imutils.is_cv2() else contours[1]
    return max(contours, key=cv2.contourArea)

def contour_bounding_box(c):
    left = tuple(c[c[:, :, 0].argmin()][0])
    top = tuple(c[c[:, :, 1].argmin()][0])
    right = tuple(c[c[:, :, 0].argmax()][0])
    bottom = tuple(c[c[:, :, 1].argmax()][0])
    return left, top, right, bottom

def mask_image(image, contour):
    left, top, right, bottom = contour_bounding_box(contour)
    min_x, min_y = left[0], top[1]
    max_x, max_y = right[0], bottom[1]
    return image[min_y:max_y, min_x:max_x]

def process(image):
    text_contour = extract_external_contour(image)
    text_image = mask_image(image, text_contour)
    return text_image

def _show_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    input_image = cv2.imread('./data/test.jpg', cv2.IMREAD_GRAYSCALE)
    input_image = cv2.medianBlur(input_image, 3)
    input_image = binarizer.process(input_image)
    input_image = cv2.bitwise_not(input_image)

    kernel = np.ones((51, 51) , np.uint8)
    input_image = cv2.dilate(input_image, kernel, iterations = 1)
    _show_image(input_image)
    
    output_image = process(input_image)

    _show_image(output_image)

if __name__ == '__main__':
    main()
