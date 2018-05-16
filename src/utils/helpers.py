import cv2
import matplotlib.pyplot as plt
import numpy as np

def _debug_display_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def _debug_plot_array(array):
    plt.plot(array)
    plt.xlabel('x label')
    plt.ylabel('y label')
    plt.title("Simple Plot")
    plt.legend()
    plt.show()