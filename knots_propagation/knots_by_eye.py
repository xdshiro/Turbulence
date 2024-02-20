import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.signal import convolve2d
from scipy.optimize import curve_fit
from scipy.optimize import brute
# import cv2
import torch
import json
import csv
from functions.all_knots_functions import *
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import collections
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import pickle
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torchsummary import summary
import os
import csv
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

knot_types = {
	'standard_14': 0,  # 1
	'standard_16': 1,  # 2
	'standard_18': 2,  # 3
	'30both': 3,  # 4
	'30oneZ': 4,  # 5
	'optimized': 5,  # 6
	'pm_03_z': 6,  # 7
	'30oneX': 7,  # 11
	
}
knots = [
	'standard_14', 'standard_16', 'standard_18', '30both', '30oneZ',
	'optimized', 'pm_03_z',
	'30oneX'
]
folder = 'data_low_10'
desired_res = (16, 16, 16)
csv_file_path = 'test2.csv'
num_classes = len(knots)
X_list = []
Y_list = []
csv.field_size_limit(10000000)
stop_big = 0
limits_to_stop = 9
knots_breaks = []
j = 0
for knot in knots:
	j += 1
	knot_breaks = []
	filename = f'..\\{folder}\\data_{knot}.csv'
	i = 0
	with open(filename, 'r') as file:
		reader = csv.reader(file)
		stop_small = 0
		for row in reader:
			i += 1
			# Deserialize the JSON string back to a list
			data_list = json.loads(row[0])
			# Convert the list back to a NumPy array if needed
			data_array = np.array(data_list)
			points_list = data_array[2:]
			Nx, Ny, Nz = data_array[1]
			# if desired_res != (Nx, Ny, Nz):
			#     scale_x = desired_res[0] / Nx
			#     scale_y = desired_res[1] / Ny
			#     scale_z = desired_res[2] / Nz
			#     points_list = np.rint(points_list * np.array([scale_x, scale_y, scale_z])).astype(int)
			dots_bound = [
				[0, 0, 0],
				[Nx, Ny, Nz],
			]
			plot = 1
			if plot:
				fig = plt.figure()
				ax = fig.add_subplot(111, projection='3d')
				
				# Scatter plot
				ax.scatter(points_list[:, 0], points_list[:, 1], points_list[:, 2], s=100, edgecolors='k', alpha=0.8)
				ax.view_init(elev=90, azim=0)
				# fig = pl.plotDots(points_list, dots_bound, color='black', show=False, size=10)
				# fig.update_layout(
				#     scene=dict(
				#         camera=dict(
				#             eye=dict(x=0, y=0, z=5),  # Adjust x, y, and z to set the default angle of view
				#             up=dict(x=0, y=1, z=0)
				#         )
				#     )
				# )
				# fig.show()
				plt.show()
			
			while True:

				user_input = input(f"({i}, {j})  Type 1: Hopf, 2: Not, 3: skip, 0: stop this type, 10: stop everything! \n")
				if user_input.lower() == '9':
					print("Hopf")
					knot_breaks.append(9)
					break
				elif user_input.lower() == '0':
					print("Nope")
					knot_breaks.append(0)
					break
				elif user_input.lower() == '99':
					print("Nope")
					knot_breaks.append(99)
					break
				elif user_input.lower() == '-':
					print("skipped")
					knot_breaks.append(8)
					break
				elif user_input.lower() == '=':
					print("Stopping small cycle!")
					stop_small = 1
					break
				elif user_input.lower() == '==':
					print("Stopping everything!")
					stop_big = 1
					break
				else:
					print("One more time pls", user_input)
			
			# knot_breaks.append(i+j)
			if stop_small or stop_big:
				break
			if i == limits_to_stop:
				break
	knots_breaks.append(knot_breaks)
	if stop_big:
		break
print(knots_breaks)
# Check if the file exists and if headers are needed
data_dict = {name: np.array(data) for name, data in zip(knots, knots_breaks)}

new_data_df = pd.DataFrame({name: pd.Series(data) for name, data in zip(knots, knots_breaks)})

if os.path.isfile(csv_file_path):
	# File exists, so read the existing data into a DataFrame
	existing_df = pd.read_csv(csv_file_path)
	
	# Ensure all columns are present, adding them as empty if they are missing
	for col in new_data_df.columns:
		if col not in existing_df.columns:
			existing_df[col] = np.nan
	
	# Find the maximum length to pad the columns
	max_length = max(len(existing_df), len(new_data_df))
	
	# Pad existing data to match the maximum length
	for col in existing_df.columns:
		existing_df[col] = existing_df[col].reindex(range(max_length)).reset_index(drop=True)
	
	# Concatenate the new data
	updated_df = pd.concat([existing_df, new_data_df], axis=0).reset_index(drop=True)
	
	# Save the updated DataFrame back to the CSV, aligning all data correctly
	updated_df.to_csv(csv_file_path, index=False)
else:
	# File does not exist, create a new CSV file from the new data
	new_data_df.to_csv(csv_file_path, index=False)

print("Data has been added to", csv_file_path)