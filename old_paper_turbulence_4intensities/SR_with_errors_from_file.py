import numpy as np
import matplotlib.pyplot as plt

# Load the arrays from the new file
file_path = './arrays_SR.npy'
arrays = np.load(file_path)
# Rytov values (same as before)
Rytov_values = [0.025, 0.05, 0.1, 0.15, 0.2]

# Function to calculate SR value
def calculate_sr(array):
	avg_intensity = np.mean(array)
	sr_value = avg_intensity / 44730879774
	return sr_value

# Function to calculate the error in the SR value
def calculate_sr_error(array):
	N = len(array)
	sigma_avg_I = np.std(array) / np.sqrt(N)
	sr_error = sigma_avg_I / 44730879774
	return sr_error

# Calculate SR values and their errors for each array
sr_results_with_error = []
for i, array in enumerate(arrays):
	sr_value = calculate_sr(array)
	sr_error = calculate_sr_error(array)
	sr_results_with_error.append((sr_value, sr_error))
	print(f"Array {i + 1}: SR Value = {sr_value:.6f}, Error = {sr_error:.6f}")

# Extracting SR values and errors from the results
sr_values = [result[0] for result in sr_results_with_error]
sr_errors = [result[1] for result in sr_results_with_error]

# Plotting SR values vs. Rytov values
plt.errorbar(Rytov_values, sr_values, yerr=sr_errors, fmt='o', capsize=5, label='SR Value')
plt.xlabel('Rytov Values')
plt.ylabel('SR Value')
plt.title('SR Value vs. Rytov Values')
plt.grid(True)
plt.legend()
plt.show()