# author: hardik bishnoi
# date: oct 1, 2024

import cv2
import numpy as np
import matplotlib.pyplot as plt

# function to add salt and pepper noise
def add_salt_and_pepper_noise(image, noise_level):
    noisy_image = np.copy(image)
    num_pixels = int(noise_level * image.size / 100)
    # add salt (white pixels)
    coords = [np.random.randint(0, i - 1, num_pixels) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255
    # add pepper (black pixels)
    coords = [np.random.randint(0, i - 1, num_pixels) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0
    return noisy_image

# function to apply mean filter
def apply_filter(image, kernel_size):
    return cv2.blur(image, (kernel_size, kernel_size))

# load the image and convert it to grayscale
image = cv2.imread('dog.jpeg', cv2.IMREAD_GRAYSCALE)

# noise levels
noise_levels = [1, 10, 50]

# prepare figure for 3x3 plot
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.ravel()

# loop through noise levels
for idx, noise_level in enumerate(noise_levels):
    # add salt and pepper noise
    noisy_image = add_salt_and_pepper_noise(image, noise_level)

    # plot noisy image in the first row
    axes[idx].imshow(noisy_image, cmap='gray')
    axes[idx].set_title(f"Noise Level: {noise_level}%")
    axes[idx].axis('off')

    # apply 3x3 filter
    filtered_3x3 = apply_filter(noisy_image, 3)
    # plot image filtered with 3x3 filter in the second row
    axes[idx + 3].imshow(filtered_3x3, cmap='gray')
    axes[idx + 3].set_title(f"3x3 Filter, Noise: {noise_level}%")
    axes[idx + 3].axis('off')

    # apply 5x5 filter
    filtered_5x5 = apply_filter(noisy_image, 5)
    # plot image filtered with 5x5 filter in the third row
    axes[idx + 6].imshow(filtered_5x5, cmap='gray')
    axes[idx + 6].set_title(f"5x5 Filter, Noise: {noise_level}%")
    axes[idx + 6].axis('off')

# original image in the top left corner
axes[0].imshow(image, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis('off')

plt.tight_layout()
plt.show()
