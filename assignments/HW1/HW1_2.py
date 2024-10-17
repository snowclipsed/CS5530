# Author: Hardik Bishnoi
# Date: 02/10/2024

import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_thermal_colormap():
    # Create a 256-length array for mapping brightness to thermal colors
    colormap = np.zeros((256, 3), dtype=np.uint8)
    
    for i in range(256):
        if i < 85:  # Dark (blue) to average (green)
            blue = 255 - i * 3
            green = i * 3
            red = 0
        elif i < 170:  # Average (green) to bright (red)
            blue = 0
            green = 255 - (i - 85) * 3
            red = (i - 85) * 3
        else:  # Brightest red
            blue = 0
            green = 0
            red = 255
            
        colormap[i] = [blue, green, red]
    
    return colormap

def apply_thermal_filter(image):
    # Convert to grayscale if not already
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Create thermal colormap
    colormap = create_thermal_colormap()
    
    # Create output image
    thermal = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    
    # Apply thermal colors based on brightness
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            thermal[i, j] = colormap[gray[i, j]]
    
    return thermal

def process_and_display_image(image_path, title):
    # Read image
    image = cv2.imread(image_path)
    
    # Apply thermal filter
    thermal = apply_thermal_filter(image)
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f'Original {title}')
    
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB))
    plt.title(f'Thermal {title}')
    
    # Create histogram of original image brightness
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.subplot(133)
    plt.hist(gray.ravel(), 256, [0, 256])
    plt.title(f'Brightness Histogram {title}')
    
    plt.tight_layout()
    
    # Save the thermal image
    cv2.imwrite(f'thermal_{title.lower().replace(" ", "_")}.jpg', thermal)
    
    return image, thermal

# Process both images
image_paths = ["frieren.png", "spacegrey.JPEG"]
titles = ["IMG 1", "IMG 2"]

for path, title in zip(image_paths, titles):
    original, thermal = process_and_display_image(path, title)
    
    # Print some statistics
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    print(f"\nStatistics for {title}:")
    print(f"Average brightness: {np.mean(gray):.2f}")
    print(f"Minimum brightness: {np.min(gray)}")
    print(f"Maximum brightness: {np.max(gray)}")

plt.show()

# Display the colormap
colormap = create_thermal_colormap()
plt.figure(figsize=(15, 2))
plt.imshow([colormap])
plt.title('Thermal Colormap')
plt.axis('off')
plt.show()