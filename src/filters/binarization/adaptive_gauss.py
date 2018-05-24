import cv2
from filters.image_filter import ImageFilter

class GaussBinarization(ImageFilter):

    def __init__(self, block_size=15, C=2):
        """Performs Adaptive Gauss binarization on given image
        """
        super().__init__()
        self.block_size = block_size
        self.C = C

    def process(self, input_image):
        """Converts given image to binary image using Adaptive Gauss algorithm

        Arguments:
            input_image {opencv image} -- input image

        Returns:
            opencv image -- Returns binary image where 0 values denotes 
                            foreground and 255 background
        """

        output_image = cv2.adaptiveThreshold(input_image, 255, 
                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY,
                                             self.block_size,
                                             self.C)

        return output_image
