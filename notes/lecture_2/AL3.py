import cv2

path = "dog.jpeg"
image = cv2.imread(path)
img_gray = cv2.imread(path, 0)

inverted_grayscale = 255 - img_gray
cv2.imshow('image', image)
cv2.imshow('grayscale', img_gray)
cv2.imshow('inverted_grayscale', inverted_grayscale)
print(img_gray)
cv2.waitKey(0)
