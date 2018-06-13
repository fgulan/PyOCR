import cv2

# otvaranje slike
image = cv2.imread('image.jpg', cv2.IMREAD_COLOR)
# pretvorba slike u nijanse sive
grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#
_, output_image = cv2.threshold(
    grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


cv2.imshow('logo', grayscale_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
