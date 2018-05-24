import cv2
from filters.image_filter import ImageFilter


class NoiseRemoval(ImageFilter):

    def __init__(self, scaled_dimension=800, kernel_size=30):
        """Performs noise removal on given image

        Keyword Arguments:
            scaled_dimension {int} -- Scaled image size (used for cropping foreground) (default: {800})
        """

        super().__init__()
        self.scaled_dimension = scaled_dimension
        self.kernel_size = kernel_size

    def process(self, input_image):
        """Removes noise (dots, uneven fragments...) from given image

        Arguments:
            input_image {opencv image} -- input image where 0 values denotes 
                            background and 255 foreground

        Returns:
            opencv image -- Returns binary image where 0 values denotes 
                            background and 255 foreground
        """
        height, width = input_image.shape

        new_image = input_image.copy()

        biggest_dimension = max(height, width)
        scale = self.scaled_dimension / biggest_dimension

        new_height, new_width = round(height * scale), round(width * scale)
        scaled_image = cv2.resize(new_image, (new_width, new_height))

        denoised_image = cv2.fastNlMeansDenoising(
            scaled_image, None, 30, 7, 21)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.kernel_size, self.kernel_size))
        dilated_mask = cv2.dilate(denoised_image, kernel)

        binary_mask = cv2.threshold(
            dilated_mask, 127, 255, cv2.THRESH_BINARY)[1]
        binary_mask = cv2.resize(binary_mask, (width, height))

        output_image = cv2.bitwise_and(
            input_image, input_image, mask=binary_mask)

        return output_image
