# Hardik Bishnoi
# 9/24/2024

import cv2
import numpy as np
import matplotlib.pyplot as plt


dark_part = np.random.randint(50, 100, size = (500,500), dtype = np.uint8)
light_part = np.random.randint(100,150, size = (500,500),dtype=np.uint8)
dark_light = np.concatenate((dark_part, light_part), axis= 1)
cv2.imshow('darklight', dark_light)


dark_light[0:200, 0:250] = cv2.equalizeHist(dark_light[0:200, 0:250])        # Top-left
dark_light[0:200, 250:500] = cv2.equalizeHist(dark_light[0:200, 250:500])    # Top-right
dark_light[200:500, 0:250] = cv2.equalizeHist(dark_light[200:500, 0:250])    # Bottom-left
dark_light[200:500, 250:500] = cv2.equalizeHist(dark_light[200:500, 250:500])# Bottom-right

# Apply histogram equalization to the second half (light_part)
dark_light[0:200, 500:750] = cv2.equalizeHist(dark_light[0:200, 500:750])    # Top-left (light part)
dark_light[0:200, 750:1000] = cv2.equalizeHist(dark_light[0:200, 750:1000])  # Top-right (light part)
dark_light[200:500, 500:750] = cv2.equalizeHist(dark_light[200:500, 500:750])# Bottom-left (light part)
dark_light[200:500, 750:1000] = cv2.equalizeHist(dark_light[200:500, 750:1000])# Bottom-right (light part)

cv2.imshow('darklight equalized', dark_light)


cv2.waitKey(0)

