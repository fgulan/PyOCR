import cv2
from skimage.filters import threshold_sauvola
from skimage import img_as_ubyte

# otvaranje slike
image = cv2.imread('image.jpg', cv2.IMREAD_COLOR)
# pretvorba slike u nijanse sive
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# izračun praga za pojedini slikovni element Sauvola metodom
thresholds = threshold_sauvola(grayscale_image,
                               window_size=15,
                               k=0.2)

binarized_image = img_as_ubyte(grayscale_image > thresholds)

cv2.imshow('logo', grayscale_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
