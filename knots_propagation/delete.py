import numpy as np
from scipy.signal import convolve2d

# Define the input matrix and kernel
input_matrix = np.array([
	[0, 0, -1, 0, 0, 0, 1, 0, 0], [0, -1, -1, -1, 0, 1, 1, 1, 0],
	[-1, -1, -1, -1, 0, 1, 1, 1, 1], [0, -1, -1, -1, 0, 1, 1, 1, 0],
	[0, 0, -1, 0, 0, 0, 1, 0, 0]
])
kernel = np.array([[0, -0.5, 0], [-0.5, 1, -0.5], [0, -0.5, 0]])
# Perform 2D convolution with 'same' mode to keep the output shape same as input
convolution_result = convolve2d(input_matrix, kernel, mode='same', boundary='fill', fillvalue=0)
print(convolution_result)

import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
# Load a standard image from skimage's data module
from skimage import data
image = data.camera()
kernel = np.array([[0, -0.5, 0], [-0.5, 1, -0.5], [0, -0.5, 0]])
filtered_image = convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0)
# Display the original and filtered images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original')
axes[1].imshow(filtered_image, cmap='gray')
axes[1].set_title('Filtered')
plt.tight_layout()
plt.show()

image = -1 * np.ones((10, 10), dtype=float)
image[3:7, 3:7] = 1  # White square
filtered_image = convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0)
# Display the original and filtered images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original')
axes[1].imshow(filtered_image, cmap='gray')
axes[1].set_title('Filtered')
plt.tight_layout()
plt.show()