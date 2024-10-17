#Hardik Bishnoi
# 8 Oct 2024
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_resize_images(image1_path, image2_path, size=(600, 600)):
    # load images in grayscale
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    
    # resize images to be equal size
    img1_resized = cv2.resize(img1, size)
    img2_resized = cv2.resize(img2, size)
    
    return img1_resized, img2_resized

def apply_low_pass_filter(image, kernel_size=11):
    # create kernel for low pass filter
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    
    # apply filter
    return cv2.filter2D(image, -1, kernel)

def create_high_pass_image(original, low_pass):
    # subtract low pass from original and add 127
    return np.clip(original.astype(np.float32) - low_pass.astype(np.float32) + 127, 0, 255).astype(np.uint8)

def combine_images(low_pass, high_pass):
    # multiply the images and normalize
    return np.clip((low_pass.astype(np.float32) * high_pass.astype(np.float32)) / 255.0, 0, 255).astype(np.uint8)

def process_and_display_images(image1_path, image2_path):
    # load and resize images
    img1, img2 = load_and_resize_images(image1_path, image2_path)
    
    # apply low pass filter to first image
    low_pass = apply_low_pass_filter(img1)
    
    # create high pass version of second image
    low_pass_img2 = apply_low_pass_filter(img2)
    high_pass = create_high_pass_image(img2, low_pass_img2)
    
    # combine images
    combined = combine_images(low_pass, high_pass)
    
    # display results
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].imshow(low_pass, cmap='gray')
    axs[0].set_title('Low Pass Filter (Image 1)')
    
    axs[1].imshow(high_pass, cmap='gray')
    axs[1].set_title('High Pass Filter (Image 2)')
    
    axs[2].imshow(combined, cmap='gray')
    axs[2].set_title('Combined Image')
    
    # add y-axis ticks from 0 to 600
    for ax in axs:
        ax.set_yticks(np.arange(0, 601, 100))
        ax.set_xticks(np.arange(0, 601, 200))
    
    plt.tight_layout()
    plt.show()

# example usage
if __name__ == "__main__":
    # replace these paths with your actual image paths
    image1_path = "frieren.png"
    image2_path = "mountain.jpeg"
    
    process_and_display_images(image1_path, image2_path)