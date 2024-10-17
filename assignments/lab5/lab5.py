# Hardik Bishnoi
# Date: 15th October 2021
# About: Lab 5 - Night Vision Effect
import cv2
import numpy as np

def night_vision_effect(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization to enhance contrast
    equalized = cv2.equalizeHist(gray)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    
    # Apply edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Create a green-tinted version of the equalized image
    green_tint = cv2.merge([np.zeros_like(equalized), equalized, np.zeros_like(equalized)])
    
    # Overlay edges on the green-tinted image
    result = cv2.addWeighted(green_tint, 0.7, cv2.merge([edges, edges, edges]), 0.3, 0)
    
    return result

# Usage example
input_image = "darkscene.jpg"
night_vision_image = night_vision_effect(input_image)
cv2.imshow("Night Vision Effect", night_vision_image)
cv2.waitKey(0)
cv2.destroyAllWindows()