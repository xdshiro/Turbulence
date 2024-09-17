import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
		ratings = existing_ratings_df.values.tolist()
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
	
	# Final save to ensure all ratings are saved
	ratings_df = pd.DataFrame(ratings, columns=['File Name', 'Rating'])
	ratings_df.to_csv(output_csv, index=False)
	print(f'Final ratings saved to {output_csv}')

folder_path = 'data'
folder_to_save = 'dots'

input_folder = f'../{folder_to_save}/{folder_path}.csv'
output_csv = f'{folder_to_save}_{folder_path}.csv'  # 025
files_csv = True

# Example usage:
review_and_rate_dots(input_folder=input_folder, output_csv=output_csv, file_csv=True)
