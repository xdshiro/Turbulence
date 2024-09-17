import os
from scipy.io import loadmat
from functions.all_knots_functions import *
import csv
import json
from tqdm import tqdm

# testing True, when everything is working - set to False
test_mode = True

# Directory containing .mat files
folder_path = 'data'
folder_to_save = 'dots_html'
folder_to_save_dots_arrays = 'dots_arrays_csv'

saving_dots_arrays = True
# propagation parameters.
knot_length = 212.58897655870774 / 2 * 1.4
z_resolution = 35  # total amount of points along z
beam_size = 6e-3 / np.sqrt(2)
wavelength = 532e-9
window_size = 40e-3
crop_percentage_for_3d_dots = 0.7

# Loop through each .mat file in the folder
file_list = os.listdir(folder_path)
indx = -1
for file_name in tqdm(file_list, desc="Processing Files"):
	if not os.path.exists(folder_to_save):
		os.makedirs(folder_to_save)
	if not os.path.exists(folder_to_save_dots_arrays):
		os.makedirs(folder_to_save_dots_arrays)
	if file_name.endswith('.mat'):
		
		indx += 1
		
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
		
		# turbulence
		beam_par = (0, 0, beam_size, wavelength)  # ..., ..., width, wavelength
		x_resolution = np.shape(field)[0]
		pxl_scale = window_size / (x_resolution - 1)
		psh_par_0 = (1e100, x_resolution, pxl_scale, 1e100, 1e100)  # ..., resolution, pixel scale, ..., ...
		crop_3d = int(x_resolution * crop_percentage_for_3d_dots)
		x_cent, y_cent = x_resolution // 2, x_resolution // 2
		
		field_3d = beam_expander(
			field, beam_par, psh_par_0,
			distance_both=knot_length,
			steps_one=z_resolution // 2
		)[
		           x_cent - crop_3d // 2: x_cent + crop_3d // 2,
		           y_cent - crop_3d // 2: y_cent + crop_3d // 2,
		           :-1
		           ]
		
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
		
		dots_bound = [
			[0, 0, 0],
			[crop_3d, crop_3d, z_resolution + 1],
		]
		if not test_mode:
			file_name = os.path.join(folder_to_save, folder_to_save + '_' + str(indx) + '.html')
			pl.plotDots(dots_cut, dots_bound, color='black', show=False, size=10, save=file_name)
		else:
			file_name = os.path.join(folder_to_save, folder_to_save + '_' + 'TEST.html')
			pl.plotDots(dots_cut, dots_bound, color='black', show=True, size=10, save=file_name)
		if saving_dots_arrays and not test_mode:
			resolution = (crop_3d, crop_3d)
			knot_resolution = [resolution[0], resolution[0], z_resolution]
			
			dots_cut_modified = np.vstack([[indx, 0, 0], knot_resolution, dots_cut])
			
			filename = os.path.join(folder_to_save_dots_arrays, f'{folder_path}.csv')
			dots_json = json.dumps(dots_cut_modified.tolist())
			with open(filename, 'a', newline='') as file:
				writer = csv.writer(file)
				writer.writerow([dots_json])
		if test_mode:
			break
