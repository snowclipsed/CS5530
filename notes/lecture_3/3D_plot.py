# Hardik Bishnoi
# 9/24/2024


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in RGB format
path = 'frieren.png'
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray_img = cv2.imread(path, 0)
# Get image dimensions
height, width, _ = img.shape
x = np.linspace(0, width, width, dtype=int)
y = np.linspace(0, height, height, dtype=int)
X, Y = np.meshgrid(x, y)


# Extract RGB channels
R = img[:, :, 0]
G = img[:, :, 1]
B = img[:, :, 2]


fig = plt.figure(figsize=(12, 10))


# Red channel
ax1 = fig.add_subplot(231, projection='3d')
surf1 = ax1.plot_surface(X, Y, R, cmap='Reds')
ax1.set_title('Red Channel')
ax1.set_zlim(0, 255)
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

# Green channel
ax2 = fig.add_subplot(232, projection='3d')
surf2 = ax2.plot_surface(X, Y, G, cmap='Greens')
ax2.set_title('Green Channel')
ax2.set_zlim(0, 255)
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

# Blue channel
ax3 = fig.add_subplot(233, projection='3d')
surf3 = ax3.plot_surface(X, Y, B, cmap='Blues')
ax3.set_title('Blue Channel')
ax3.set_zlim(0, 255)
fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)


# Grayscale

ax4 = fig.add_subplot(212, projection='3d')
surf4 = ax4.plot_surface(X,Y,gray_img, cmap='gray')
ax3.set_zlim(0,255)
fig.colorbar(surf4, ax=ax4, aspect = 10)

plt.tight_layout()
plt.show()

 