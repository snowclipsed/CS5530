import cv2
import numpy as np

# Load the image
path = "flowers.jpg"
image = cv2.imread(path)


# i am going to resize all the images so i can put them all in a screenshot
image_resized = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)

#split the image into its respective Blue, Green, and Red channels
b_channel, g_channel, r_channel = cv2.split(image)

# Apply histogram equalization to each channel
b_eq = cv2.equalizeHist(b_channel)
g_eq = cv2.equalizeHist(g_channel)
r_eq = cv2.equalizeHist(r_channel)

# then we merge the equalized channels back together
equalized_image = cv2.merge([b_eq, g_eq, r_eq])

# resize the equalized image for display as well
equalized_image_resized = cv2.resize(equalized_image, (0, 0), fx=0.3, fy=0.3)

cv2.imshow('Original Image (Resized)', image_resized)
cv2.imshow('Histogram Equalized Image (Resized, Per Channel)', equalized_image_resized)

cv2.waitKey(0)
cv2.destroyAllWindows()
