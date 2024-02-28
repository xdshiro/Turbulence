import matplotlib.pyplot as plt
import numpy as np
# from skimage.measure import label
from scipy.ndimage import distance_transform_edt, label
from skimage.measure import regionprops
from skimage.morphology import dilation, square


def merge_and_fill(grid, area_threshold):
	labeled_array, num_features = label(grid)
	props = regionprops(labeled_array)
	large_labels = [prop.label for prop in props if prop.area >= area_threshold]
	
	# Initialize a distance transform map with large regions set to zero distance
	distance_map = np.ones_like(grid) * np.max(grid.shape)  # Max possible distance
	for large_label in large_labels:
		distance_map[labeled_array == large_label] = 0
	
	# Compute distance transform from large regions
	distance_transform = distance_transform_edt(distance_map)
	
	# Dilate small regions towards the nearest large region
	for prop in props:
		if prop.label not in large_labels:
			# Find pixels that need to be merged to connect to a larger region
			merge_target_mask = dilation(labeled_array == prop.label, square(2))
			# Update only those pixels that are closer to the target large region
			grid[(merge_target_mask == 1) & (distance_transform <= area_threshold)] = 1
	
	return grid


# Example grid
grid = np.array([[1, 1, 0, 1, 1],
                 [1, 1, 0, 0, 1],
                 [0, 0, 0, 1, 1],
                 [1, 1, 0, 0, 0]])
plt.imshow(grid)
plt.show()
area_threshold = 3  # Define your area threshold here

# Merge small regions with larger ones
new_grid = merge_and_fill(grid, area_threshold)
plt.imshow(new_grid)
plt.show()

grid = np.array([[0, 1, 0, 1, 1],
                 [1, 1, 0, 0, 1],
                 [0, 0, 0, 0, 1],
                 [1, 1, 0, 0, 0]])
plt.imshow(grid)
plt.show()

# Merge small regions with larger ones
new_grid = merge_and_fill(grid, area_threshold)
plt.imshow(new_grid)
plt.show()
