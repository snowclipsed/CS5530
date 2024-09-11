import cv2


path = "testimage1.png"
image = cv2.imread(path)
img_gray = cv2.imread(path, 0)

cv2.imshow('image', image)
cv2.imshow('grayscale', img_gray)
cv2.waitKey(0)
