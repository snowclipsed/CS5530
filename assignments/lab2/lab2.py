# Hardik Bishnoi
# Date: September 24, 2024


import cv2
import numpy as np

# Load the image in color (default)
image = cv2.imread('flowers.jpg')


if image is None:
    print("Error loading image.")
else:

    def average_method(img):
        # Take the average of the RGB values
        return np.mean(img, axis=2).astype(np.uint8)


    def ntsc_method(img):
        # NTSC conversion formula: 0.299*R + 0.587*G + 0.114*B
        return (0.299 * img[:,:,2] + 0.587 * img[:,:,1] + 0.114 * img[:,:,0]).astype(np.uint8)

    grayscale_avg = average_method(image)

    grayscale_ntsc = ntsc_method(image)

    grayscale_opencv = cv2.imread('flowers.jpg', cv2.IMREAD_GRAYSCALE)

    # Display results
    cv2.imshow('Original Image', image)
    cv2.imshow('Grayscale - Average Method', grayscale_avg)
    cv2.imshow('Grayscale - NTSC Method', grayscale_ntsc)
    cv2.imshow('Grayscale - OpenCV', grayscale_opencv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
