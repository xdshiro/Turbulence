"""
Knot and 4Foil Field Processing Script
======================================
This script processes experimental data for knots and 4foil fields, extracting
key features (dots) from the fields for later use in classification tasks.
The classification models are trained on simulated shapes, and this script
bridges the gap between experimental data and those models.

Main Workflow:
--------------
1. Load and Process Fields:
   - Reads `.mat` files containing 2D complex fields.
   - Crops each field around its center of mass.
   - Rescales cropped fields to a standard resolution.

2. Expand Fields into 3D:
   - Propagates the processed fields into 3D space using beam expansion parameters.
   - Crops the resulting 3D fields to focus on the region of interest.

3. Extract Singularities:
   - Computes singularities (dots) from the phase of the 3D complex field.
   - Filters out duplicate dots and rescales their coordinates to a standard resolution.

4. Save Processed Data:
   - Extracted features (dots) are saved in a format compatible with classification pipelines.

Key Features:
-------------
- **Center-of-Mass Cropping**: Ensures that processing focuses on the most relevant part of the field.
- **Complex Interpolation**: Maintains the fidelity of the complex field during cropping and rescaling.
- **Singularity Extraction**: Identifies phase singularities, which are key features for classification tasks.
- **Integration with Classification Pipelines**: Outputs data in a format ready for use in machine learning models trained on simulations.

Parameters:
-----------
- **folder_path**: Path to the folder containing `.mat` files with experimental data.
- **crop_size**: Size of the cropped region (assumes a square crop).
- **rescale_size**: Resolution to which the cropped fields are rescaled.
- **saving_size**: Final resolution for the extracted features (dots).
- **res_z**: Number of steps for 3D propagation.
- **crop_3d**: Crop size for the 3D propagated field.
- **knot_length**: Length of the knot used for beam expansion.
- **beam_par**: Beam parameters (e.g., beam width, wavelength).
- **psh_par_0**: Parameters for phase shift computation during propagation.

Outputs:
--------
- The script saves the extracted dots (singularities) in a format compatible with further analysis or classification tasks.

Usage:
------
1. Place experimental `.mat` files in the specified `folder_path`.
2. Configure parameters (crop size, resolutions, etc.) as needed.
3. Run the script to process the fields, extract dots, and save the results.

Dependencies:
-------------
- `numpy`
- `scipy`
- `torch`
- `matplotlib` (optional, for visualization)
- `functions.all_knots_functions` (custom module for domain-specific processing)

Code Organization:
------------------
1. **Functions**:
   - `compute_center_of_mass`: Finds the center of mass of the absolute value of a complex field.
   - `crop_and_rescale`: Crops and rescales a 2D complex field.
   - `process_file`: Processes a single `.mat` file by cropping and rescaling its field.
   - `process_folder`: Processes all `.mat` files in a folder.

2. **Main Script**:
   - Processes fields from `.mat` files.
   - Propagates 2D fields into 3D space.
   - Extracts singularities (dots) from the 3D fields.
   - Optionally visualizes intermediate steps and results.
   - Saves the processed features for use in machine learning models.

Notes:
------
- This script assumes specific structures and naming conventions for the input `.mat` files.
- The extracted dots are filtered and scaled to standard resolutions to ensure compatibility with classification models.
- Future extensions may include additional preprocessing or integration with simulation-based feature analysis.

"""

import os
import torch
import scipy.io as sio
from torch.nn.functional import interpolate
from functions.all_knots_functions import *  # Import your plotting and utility functions
import csv
import json
# ---------------------------------
# Functions
# ---------------------------------
def compute_center_of_mass(field):
	"""
	Computes the center of mass of the absolute value of a complex field.

	Args:
		field (torch.Tensor): 2D complex field tensor.

	Returns:
		(int, int): Center of mass coordinates (y, x).
	"""
	# plot_field_both(field)
	abs_field = torch.abs(field)
	y_indices, x_indices = torch.meshgrid(
		torch.arange(field.shape[0]), torch.arange(field.shape[1]), indexing='ij'
	)
	total_mass = abs_field.sum()
	center_y = (y_indices * abs_field).sum() / total_mass
	center_x = (x_indices * abs_field).sum() / total_mass
	return int(round(center_y.item())), int(round(center_x.item()))


def crop_and_rescale(field, window_size, xy_lim_2D_origin, rescale_size):
	"""
	Crops a 2D complex field around its center and rescales it to a lower resolution.

	Args:
		field (torch.Tensor): Input 2D complex field tensor.
		crop_size (int): The size of the cropped region (assumes square crop).
		rescale_size (int): The size of the rescaled region (assumes square rescale).

	Returns:
		torch.Tensor: Rescaled complex field.
	"""
	h, w = field.shape
	center_y, center_x = compute_center_of_mass(field)
	# print(center_y, center_x)
	image_resolution = np.shape(field)[0]
	# print(image_resolution)
	needed_window_size = xy_lim_2D_origin
	# print(needed_window_size)
	crop_size = int(np.round(needed_window_size / window_size * image_resolution))
	# print(crop_size)
	# Calculate crop boundaries
	crop_half = crop_size // 2
	x_min = max(center_x - crop_half, 0)
	x_max = min(center_x + crop_half, w)
	y_min = max(center_y - crop_half, 0)
	y_max = min(center_y + crop_half, h)
	
	# Crop the field
	cropped_field = field[y_min:y_max, x_min:x_max].T
	
	# Separate real and imaginary parts, rescale, and recombine
	cropped_real = cropped_field.real.unsqueeze(0).unsqueeze(0)
	cropped_imag = cropped_field.imag.unsqueeze(0).unsqueeze(0)
	rescaled_real = interpolate(cropped_real, size=(rescale_size, rescale_size), mode='bilinear').squeeze()
	rescaled_imag = interpolate(cropped_imag, size=(rescale_size, rescale_size), mode='bilinear').squeeze()
	return rescaled_real + 1j * rescaled_imag


def process_file(file_path, window_size, xy_lim_2D_origin, rescale_size):
	"""
	Reads a .mat file, processes its field data, crops and rescales it.

	Args:
		file_path (str): Path to the .mat file.
		crop_size (int): The size of the cropped region (assumes square crop).
		rescale_size (int): The size of the rescaled region (assumes square rescale).

	Returns:
		torch.Tensor: Processed (cropped and rescaled) complex field.
	"""
	mat_data = sio.loadmat(file_path)
	field_name = [key for key in mat_data.keys() if not key.startswith('__')][0]
	field_data = mat_data[field_name]
	tensor_data = torch.tensor(field_data, dtype=torch.complex64)
	return crop_and_rescale(tensor_data, window_size, xy_lim_2D_origin, rescale_size)


def process_folder(folder_path, window_size, xy_lim_2D_origin, rescale_size, one=False):
	"""
	Processes all .mat files in a folder, cropping and rescaling their data.

	Args:
		folder_path (str): Path to the folder containing .mat files.
		crop_size (int): The size of the cropped region (assumes square crop).
		rescale_size (int): The size of the rescaled region (assumes square rescale).

	Returns:
		list[torch.Tensor]: List of processed tensors.
	"""
	processed_files = []
	for file_name in os.listdir(folder_path):
		if file_name.endswith('.mat'):  # Process only .mat files
			file_path = os.path.join(folder_path, file_name)
			processed_files.append(process_file(file_path, window_size, xy_lim_2D_origin, rescale_size))
		if one:
			break
	return processed_files


# ---------------------------------
# Main Script
# ---------------------------------
if __name__ == "__main__":
	# plot = True
	# save = False
	# Toggle plot and save options
	plot = False
	save = True
	one = False
	
	# Main folder path containing subfolders
	main_folder_path = 'C:/Users/Cmex-/Box/Flowers Exp/T_30/'
	main_folder_path = 'C:/Users/Cmex-/Box/Flowers Exp/T_30_'
	folder_save = 'all_flowers20_005'
	
	# Create output folder if it doesn't exist
	
	if not os.path.exists(f'./{folder_save}'):
		os.makedirs(f'./{folder_save}')
	
	
	
	
	# Beam and window
	# size parameters
	window_size = 4.6 * 1e-3 * np.sqrt(270 / 1.5)
	# window_size = 4.4 * 1e-3 * np.sqrt(270 / 1.5)
	rescale_size = 185
	saving_size = (32, 32)
	res_z = 64
	saving_knot = [32, 32, res_z]
	crop_3d = 100
	xy_lim_2D_origin = np.array((-30.0e-3, 30.0e-3))
	knot_length = 212.58897655870774 / 2 * 1.4
	lmbda = 532e-9
	width0 = 6.0e-3 / np.sqrt(2)
	res_xy_2D_origin = 300
	k0 = 2 * np.pi / lmbda
	pxl_scale = (xy_lim_2D_origin[1] - xy_lim_2D_origin[0]) / (res_xy_2D_origin - 1)
	beam_par = (0, 0, width0, lmbda)
	psh_par_0 = (1 * 1e100, res_xy_2D_origin, pxl_scale, 1 * 1e100, 1 * 1e100)
	
	# Iterate through each subfolder in the main folder
	for folder_name in os.listdir(main_folder_path):
		subfolder_path = os.path.join(main_folder_path, folder_name)
		
		# Check if the current item is a folder
		if os.path.isdir(subfolder_path):
			flower = folder_name.replace("flower", "")  # Extract '0000' from 'stuff0000'
			print(f'Processing folder: {subfolder_path}, flower ID: {flower}')
			
			# Compute needed_size
			needed_size = (xy_lim_2D_origin[1] - xy_lim_2D_origin[0]) * 185 / 300
			
			# Process the folder to get the list of tensors
			processed_fields = process_folder(subfolder_path, window_size, needed_size, rescale_size, one=one)
			
			# Analyze processed fields
			for indx, field in enumerate(processed_fields):
				if plot:
					plot_field_both(field)
				field_3d = beam_expander(field, beam_par, psh_par_0, distance_both=knot_length, steps_one=res_z // 2)
				x_cent = field_3d.shape[0] // 2
				y_cent = field_3d.shape[1] // 2
				field_3d_crop = field_3d[
				                x_cent - crop_3d // 2: x_cent + crop_3d // 2,
				                y_cent - crop_3d // 2: y_cent + crop_3d // 2,
				                :
				                ]
				field_3d_crop = field_3d_crop[:, :, :-1]
				if plot:
					plot_field_both(field_3d_crop[:, :, res_z // 2])
				dots_init_dict, dots_init = sing.get_singularities(np.angle(field_3d_crop), axesAll=False,
				                                                   returnDict=True)
				if dots_init.size == 0:
					dots_init = np.array([[0, 0, 0]])
				dots_cut_non_unique = cut_circle_dots(dots_init, crop_3d // 2, crop_3d // 2, crop_3d // 2)
				view = np.ascontiguousarray(dots_cut_non_unique).view(
					np.dtype((np.void, dots_cut_non_unique.dtype.itemsize * dots_cut_non_unique.shape[1]))
				)
				_, idx = np.unique(view, return_index=True)
				dots_cut = dots_cut_non_unique[idx]
				
				# Scale dots to new resolution
				original_resolution = (crop_3d, crop_3d)
				scale_x = saving_size[0] / original_resolution[0]
				scale_y = saving_size[1] / original_resolution[1]
				xy = dots_cut[:, :2] * [scale_x, scale_y]
				scaled_data = np.column_stack((np.rint(xy).astype(int), dots_cut[:, 2]))
				
				dots_bound = [
					[0, 0, 0],
					[crop_3d, crop_3d, res_z + 1],
				]
				shift = (np.array(dots_bound[1]) - np.array(dots_bound[0])) / 2
				dots_cut = (dots_cut - shift) * np.array([1, 1, -1]) + shift
				if plot:
					pl.plotDots(dots_cut, dots_bound, color='black', show=True, size=10)
				
				dots_cut_modified = np.vstack([[indx, 0, 0], saving_knot, scaled_data])
				if save:
					filename = f'./{folder_save}/data_experiment_{flower}.csv'
					dots_json = json.dumps(dots_cut_modified.tolist())
					with open(filename, 'a', newline='') as file:
						writer = csv.writer(file)
						writer.writerow([dots_json])