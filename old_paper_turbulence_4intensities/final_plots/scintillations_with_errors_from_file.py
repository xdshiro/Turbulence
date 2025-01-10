import numpy as np
import matplotlib.pyplot as plt
# Load the arrays from the file
file_path = './arrays_scin.npy'
arrays = np.load(file_path)


# Function to calculate scintillation index
def calculate_scintillation(array):
	avg_intensity = np.mean(array)
	avg_intensity_squared = np.mean(array ** 2)
	scintillation_index = avg_intensity_squared / (avg_intensity ** 2) - 1
	return scintillation_index


# Function to calculate the error in the scintillation index
def calculate_scintillation_error(array):
	N = len(array)
	
	avg_intensity = np.mean(array)
	avg_intensity_squared = np.mean(array ** 2)
	
	sigma_avg_I_squared = np.std(array) / np.sqrt(N)
	sigma_avg_I2 = np.std(array ** 2) / np.sqrt(N)
	
	# Partial derivatives
	dS_d_avg_I2 = 1 / (avg_intensity ** 2)
	dS_d_avg_I_squared = -2 * avg_intensity_squared / (avg_intensity ** 3)
	
	# Error propagation formula
	error = np.sqrt(
		(dS_d_avg_I2 * sigma_avg_I2) ** 2 +
		(dS_d_avg_I_squared * sigma_avg_I_squared) ** 2
	)
	
	return error


# Calculate scintillation index and its error for each array
scintillation_results_with_error = []
for i, array in enumerate(arrays):
	scintillation_index = calculate_scintillation(array)
	scintillation_error = calculate_scintillation_error(array)
	scintillation_results_with_error.append((scintillation_index, scintillation_error))
	print(f"Array {i + 1}: Scintillation Index = {scintillation_index:.6f}, Error = {scintillation_error:.6f}")
	# print(f"Error = {scintillation_error:.6f}")
	# print(f"Array {i + 1}: Scintillation Index = {scintillation_index:.6f}")

# If you need to use the results later, you can access them from `scintillation_results_with_error`

# Rytov values
Rytov_values = [0.025, 0.05, 0.1, 0.15, 0.2]
Cn2_values = [3.98e-15, 7.95e-15, 1.59e-14, 2.39e-14, 3.18e-14]

# Extracting scintillation indices and errors from the results
scintillation_indices = [result[0] for result in scintillation_results_with_error]
scintillation_errors = [result[1] for result in scintillation_results_with_error]

# Plotting
plt.errorbar(Rytov_values, scintillation_indices, yerr=scintillation_errors, fmt='o', capsize=5, label='Scintillation Index')
plt.xlabel('Rytov Values')
plt.ylabel('Scintillation Index')
plt.title('Scintillation Index vs. Rytov Values with C_n^2 Annotations')
plt.grid(True)

# Adding C_n^2 annotations
for i, (rytov, scintillation, cn2) in enumerate(zip(Rytov_values, scintillation_indices, Cn2_values)):
    plt.annotate(f'$C_n^2$={cn2:.2e}', (rytov, scintillation), textcoords="offset points", xytext=(40,0), ha='center')

plt.legend()
plt.show()