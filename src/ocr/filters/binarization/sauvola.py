from filters.image_filter import ImageFilter
from skimage.filters import threshold_sauvola
from skimage import img_as_ubyte


class SauvolaBinarization(ImageFilter):

    def __init__(self, window_size=15, k=0.2, r=128):
        """Performs Sauvola binarization on given image

        Keyword Arguments:
            window_size {int} -- window size (default: {15})
            k {float} -- parameter k (default: {0.2})
            r {int} -- parameter R (default: {128})
        """

        super().__init__()
        self.window_size = window_size
        self.k = k
        self.r = r

    def process(self, input_image):
        """Converts given image to binary image using Sauvola algorithm

        Arguments:
            input_image {opencv image} -- input image

        Returns:
            opencv image -- Returns binary image where 0 values denotes 
                            foreground and 255 background
        """

        thresh_sauvola = threshold_sauvola(input_image,
                                           window_size=self.window_size,
                                           k=self.k, r=self.r)
        
        output_image = img_as_ubyte(input_image > thresh_sauvola)

        return output_image
