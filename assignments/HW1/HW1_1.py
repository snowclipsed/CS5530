# Author: Hardik Bishnoi
# Date: 02/10/2024

import cv2
import numpy as np
import matplotlib.pyplot as plt

def manual_histogram_equalization(image):
    # Convert image to grayscale if it's not already
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Step 1: Calculate histogram (counting)
    histogram = np.zeros(256, dtype=int)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            histogram[gray[i, j]] += 1

    # Step 2: Calculate Cumulative Distribution Function (CDF)
    cdf = np.zeros(256, dtype=float)
    cdf[0] = histogram[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + histogram[i]
    
    # Normalize CDF
    cdf = cdf / cdf.max() * 255
    cdf = cdf.astype(np.uint8)

    # Step 3: Map old values to new values
    equalized = np.zeros_like(gray)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            equalized[i, j] = cdf[gray[i, j]]
    
    return equalized, histogram, cdf

def plot_results(original, manual_equalized, cv2_equalized, manual_hist, cv2_hist, title):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(231)
    plt.imshow(original, cmap='gray')
    plt.title(f'Original {title}')
    
    plt.subplot(232)
    plt.imshow(manual_equalized, cmap='gray')
    plt.title(f'Manual Equalized {title}')
    
    plt.subplot(233)
    plt.imshow(cv2_equalized, cmap='gray')
    plt.title(f'CV2 Equalized {title}')
    
    plt.subplot(234)
    plt.hist(original.ravel(), 256, [0, 256])
    plt.title(f'Original Histogram {title}')
    
    plt.subplot(235)
    plt.plot(manual_hist)
    plt.title(f'Manual Histogram {title}')
    
    plt.subplot(236)
    plt.plot(cv2_hist)
    plt.title(f'CV2 Histogram {title}')
    
    plt.tight_layout()

def process_image(image_path, title):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Manual equalization
    manual_equalized, manual_hist, cdf = manual_histogram_equalization(image)

    # OpenCV equalization
    cv2_equalized = cv2.equalizeHist(gray)
    cv2_hist = cv2.calcHist([cv2_equalized], [0], None, [256], [0, 256]).ravel()

    # Plot results
    plot_results(gray, manual_equalized, cv2_equalized, manual_hist, cv2_hist, title)

    # Calculate MSE between manual and OpenCV results
    mse = np.mean((manual_equalized - cv2_equalized) ** 2)
    print(f"Mean Squared Error for {title}: {mse}")

    # Save images
    cv2.imwrite(f'original_{title.lower()}.jpg', gray)
    cv2.imwrite(f'manual_equalized_{title.lower()}.jpg', manual_equalized)
    cv2.imwrite(f'cv2_equalized_{title.lower()}.jpg', cv2_equalized)

    return gray, manual_equalized, cv2_equalized

# Process both images
# plt.figure(figsize=(20, 20))

# Process dark image
dark_image_path = "dark_image.jpg"
dark_results = process_image(dark_image_path, "Dark Image")

# plt.figure(figsize=(20, 20))

# Process light image
light_image_path = "light_image.jpg"
light_results = process_image(light_image_path, "Light Image")

plt.show()

# Optional: You can also print some statistics to compare the two images
def print_image_stats(original, equalized, title):
    print(f"\nStatistics for {title}:")
    print(f"Original image - Mean: {np.mean(original):.2f}, Std: {np.std(original):.2f}")
    print(f"Equalized image - Mean: {np.mean(equalized):.2f}, Std: {np.std(equalized):.2f}")

print_image_stats(dark_results[0], dark_results[2], "Dark Image")
print_image_stats(light_results[0], light_results[2], "Light Image")