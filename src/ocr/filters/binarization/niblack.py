from filters.image_filter import ImageFilter
from skimage.filters import threshold_niblack
from skimage import img_as_ubyte


class NiblackBinarization(ImageFilter):

    def __init__(self, window_size=15, k=0.2):
        """Performs Niblack binarization on given image

        Keyword Arguments:
            window_size {int} -- window size (default: {15})
            k {float} -- parameter k (default: {0.2})
        """

        super().__init__()
        self.window_size = window_size
        self.k = k

    def process(self, input_image):
        """Converts given image to binary image using Niblack algorithm

        Arguments:
            input_image {opencv image} -- input image

        Returns:
            opencv image -- Returns binary image where 0 values denotes 
                            foreground and 255 background
        """

        thresh_niblack = threshold_niblack(input_image,
                                           window_size=self.window_size,
                                           k=self.k)
        
        output_image = img_as_ubyte(input_image > thresh_niblack)

        return output_image
