import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from functions.all_knots_functions import *

# Directory containing .mat files
folder_path = 'data'
test_mode = True

# Loop through each .mat file in the folder
for file_name in os.listdir(folder_path):
	if file_name.endswith('.mat'):
		# Full file path
		file_path = os.path.join(folder_path, file_name)
		
		# Load the .mat file
		mat_data = loadmat(file_path)
		
		# Assuming the complex field is stored in a variable named 'field'
		# Replace 'field' with the actual variable name in your .mat file
		if 'U' in mat_data:
			field = mat_data['U']
		else:
			# Check the keys if the variable name is different
			print(f"Complex field not found in {file_name}. Available keys: {list(mat_data.keys())}")
			continue
		
		# Ensure the data is complex (just in case)
		if not np.iscomplexobj(field):
			print(f"Data in {file_name} is not complex. Skipping this file.")
			continue
		
		if test_mode:
			plot_field_both(field)
		
		# propagation parameters.
		knot_length = 212.58897655870774 / 2 * 1.4
		z_resolution = 30
		beam_par = (0, 0, 6e-3 / np.sqrt(2), 532e-9)  # ..., ..., width, wavelength
		crop_percentage_for_3d_dots = 0.7
		# turbulence
		x_resolution = np.shape(field)[0]
		pxl_scale = 50e-3 / (x_resolution - 1)
		psh_par_0 = (1e100, x_resolution, pxl_scale, 1e100, 1e100)  # ..., resolution, pixel scale, ..., ...
		crop_3d = int(x_resolution * crop_percentage_for_3d_dots)
		print(x_resolution)
		field_3d = beam_expander(
			field, beam_par, psh_par_0,
			distance_both=knot_length,
			steps_one=z_resolution // 2
		)[:, :, :-1]
		
		if test_mode:
			plot_field_both(field_3d[:, :, -1])
			
		
		
		dots_init_dict, dots_init = sing.get_singularities(np.angle(field_3d), axesAll=False, returnDict=True)
		
		dots_cut_non_unique = cut_circle_dots(dots_init, crop_3d // 2, crop_3d // 2, crop_3d // 2)
		
		# Creating a view with a compound data type
		view = np.ascontiguousarray(dots_cut_non_unique).view(
			np.dtype((np.void, dots_cut_non_unique.dtype.itemsize * dots_cut_non_unique.shape[1]))
		)
		# Using np.unique with the view
		_, idx = np.unique(view, return_index=True)
		dots_cut = dots_cut_non_unique[idx]
		print(dots_cut)
		if test_mode:
			dots_bound = [
				[0, 0, 0],
				[crop_3d, crop_3d, z_resolution + 1],
			]
			pl.plotDots(dots_cut, dots_bound, color='black', show=True, size=10)
			
		if test_mode:
			break
