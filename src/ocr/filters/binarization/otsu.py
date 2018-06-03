import cv2
from filters.image_filter import ImageFilter

class OtsuBinarization(ImageFilter):

    def __init__(self):
        """Performs Otsu binarization on given image
        """

        super().__init__()

    def process(self, input_image):
        """Converts given image to binary image using Otsu algorithm

        Arguments:
            input_image {opencv image} -- input image

        Returns:
            opencv image -- Returns binary image where 0 values denotes 
                            foreground and 255 background
        """

        _, output_image = cv2.threshold(input_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return output_image
