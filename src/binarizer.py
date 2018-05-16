import os
import cv2
import numpy as np

def otsu_binarize(image):
    _, output_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return output_image

def erode(image, kernel_size=(3, 3), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=iterations)
    return eroded_image

def preprocess(image):
    # blur out tiny imperfections and dots
    image = cv2.medianBlur(image, 3)
    return image

def process(image):
    return otsu_binarize(image)

def main():
    input_image = cv2.imread('./new.jpg', cv2.IMREAD_GRAYSCALE)
    output_image = process(input_image)
    cv2.imshow('image', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
