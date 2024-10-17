# Hardik Bishnoi
# September 18 2024

import cv2


path = "flowers.jpg"
image = cv2.imread(path)
img_gray = cv2.imread(path, 0)

cv2.imshow('image', image)
cv2.imshow('grayscale', img_gray)
cv2.waitKey(0)
