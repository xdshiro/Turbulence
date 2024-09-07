import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import json
import csv

def review_and_rate_dots(input_folder='processed_dots', output_csv='ratings.csv',
						 file_csv=False):
	# Initialize a list to store the ratings
	ratings = []
	i = 0
	# Load existing ratings if the file already exists
	if os.path.exists(output_csv):
		existing_ratings_df = pd.read_csv(output_csv)
		existing_files = existing_ratings_df['File Name'].tolist()
		ratings = existing_ratings_df.values.tolist()
	else:
		existing_files = []
	if file_csv:
		filename = input_folder
		knots = []
		with open(filename, 'r') as file:

			reader = csv.reader(file)

			for row in reader:
				# Deserialize the JSON string back to a list
				data_list = json.loads(row[0])
				# Convert the list back to a NumPy array if needed
				data_array = np.array(data_list)
				points_list = data_array[2:]
				Nx, Ny, Nz = data_array[1]
				knots.append(points_list)
		
		for dots_cut in knots:
			i += 1
			# Plot the dots in 3D with color gradient and larger size
			fig = plt.figure(figsize=(8, 8))
			ax = fig.add_subplot(111, projection='3d')
			# Color by z-coordinate and set size larger
			scatter = ax.scatter(dots_cut[:, 0], dots_cut[:, 1], dots_cut[:, 2], c=dots_cut[:, 2], cmap='viridis', s=90)
			fig.colorbar(scatter, ax=ax, label='Z coordinate')
			ax.view_init(elev=85, azim=-90)
			# Set the limits for x and y axes
			ax.set_xlim(0, Nx)
			ax.set_ylim(0, Ny)
			ax.set_title(f'File: {input_folder}')
			ax.set_xlabel('X')
			ax.set_ylabel('Y')
			ax.set_zlabel('Z')
			plt.show()

			# Ask for user input
			rating = input(f'({i}) Rate the plot ("=" for good, "-" for avg, "0" for bad): ')

			# Ensure the rating is valid
			while rating not in ["=", "-", "0"]:
				print('Invalid input. Please enter "=" for good, "-" for avg, "0" for bad.')
				rating = input('Rate the plot ("=" for good, "-" for avg, "0" for bad): ')

			# Append the rating to the list
			ratings.append([input_folder, rating])

			# Save to CSV every 10 ratings
			if len(ratings) % 10 == 0:
				ratings_df = pd.DataFrame(ratings, columns=['File Name', 'Rating'])
				ratings_df.to_csv(output_csv, index=False)
				print(f'Updated ratings saved to {output_csv}')
	else:
		# Iterate over all files in the input folder
		for file_name in os.listdir(input_folder):

			if file_name.endswith('.npy') and file_name not in existing_files:
				i += 1
				file_path = os.path.join(input_folder, file_name)

				# Load the dots data
				dots_cut = np.load(file_path)

				# Plot the dots in 3D with color gradient and larger size
				fig = plt.figure(figsize=(8, 8))
				ax = fig.add_subplot(111, projection='3d')
				# Color by z-coordinate and set size larger
				scatter = ax.scatter(dots_cut[:, 0], dots_cut[:, 1], dots_cut[:, 2], c=dots_cut[:, 2]
				                     , cmap='viridis', s=60)
				fig.colorbar(scatter, ax=ax, label='Z coordinate')
				ax.view_init(elev=50, azim=-90)
				# Set the limits for x and y axes
				ax.set_xlim(20, 120)
				ax.set_ylim(20, 120)
				ax.set_title(f'File: {file_name}')
				ax.set_xlabel('X')
				ax.set_ylabel('Y')
				ax.set_zlabel('Z')
				plt.show()

				# Ask for user input
				rating = input(f'({i}) Rate the plot ("=" for good, "-" for avg, "0" for bad): ')

				# Ensure the rating is valid
				while rating not in ["=", "-", "0"]:
					print('Invalid input. Please enter "=" for good, "-" for avg, "0" for bad.')
					rating = input('Rate the plot ("=" for good, "-" for avg, "0" for bad): ')

				# Append the rating to the list
				ratings.append([file_name, rating])

				# Save to CSV every 10 ratings
				if len(ratings) % 10 == 0:
					ratings_df = pd.DataFrame(ratings, columns=['File Name', 'Rating'])
					ratings_df.to_csv(output_csv, index=False)
					print(f'Updated ratings saved to {output_csv}')
	
	# Final save to ensure all ratings are saved
	ratings_df = pd.DataFrame(ratings, columns=['File Name', 'Rating'])
	ratings_df.to_csv(output_csv, index=False)
	print(f'Final ratings saved to {output_csv}')


# input_folder = '../optimized_trefoil_vs_rytov_0.025_100_1.4zR_c03_v1/data_trefoil_optimized.csv'
# output_csv = f'optimized_trefoil_vs_rytov_0.025_100_1.4zR_c03_v1.csv'
# input_folder = '../optimized_trefoil_vs_rytov_0.2_100_center_plane/data_trefoil_optimized.csv'
# output_csv = f'optimized_trefoil_vs_rytov_0.2_100_center_plane_v1.csv'


input_folder = '../optimized_trefoil_vs_rytov_0.2_100_center_plane_v3/data_trefoil_standard_12.csv'
output_csv = f'optimized_trefoil_vs_rytov_0.2_100_center_plane_v3.csv'  # 025
input_folder = '../optimized_trefoil_vs_rytov_0.2_100_center_plane_v3/data_trefoil_optimized.csv'
output_csv = f'optimized_trefoil_vs_rytov_0.2_100_center_plane_v3.csv'  # 025
input_folder = '../standard_trefoil12_vs_rytov_0.025_100_center_plane_v3/data_trefoil_standard_12.csv'
output_csv = f'standard_trefoil12_vs_rytov_0.025_100_center_plane_v3.csv'  # 025
input_folder = '../standard_vsW_trefoil_vs_rytov_0.05_100_center_plane_v3/data_trefoil_standard_105.csv'
output_csv = f'standard105_vsW_trefoil_vs_rytov_0.05_100_center_plane_v3.csv'  # 025
files_csv = True

# Example usage:
review_and_rate_dots(input_folder=input_folder, output_csv=output_csv, file_csv=True)
