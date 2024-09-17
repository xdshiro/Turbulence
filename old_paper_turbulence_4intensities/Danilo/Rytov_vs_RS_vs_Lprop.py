import numpy as np
import matplotlib.pyplot as plt




# Function to calculate SR value
def calculate_sr(array):
    avg_intensity = np.mean(array)
    sr_value = avg_intensity / 15073364168
    return sr_value


# Function to calculate the error in the SR value
def calculate_sr_error(array):
    N = len(array)
    sigma_avg_I = np.std(array) / np.sqrt(N)
    sr_error = sigma_avg_I / 15073364168
    return sr_error

def get_SR_and_Errors_from_file(file_path):
    arrays = np.load(file_path)
    SR = []
    SR_delta = []
    for i, array in enumerate(arrays):
        sr_value = calculate_sr(array)
        sr_error = calculate_sr_error(array)
        SR.append(sr_value)
        SR_delta.append(sr_error)
        print(f"Array {i + 1}: SR Value = {sr_value:.6f}, Error = {sr_error:.6f}")
    return SR, SR_delta


# Data for L=135m
file_path = './arrays_SR_L135.0.npy'
SR_135, SR_135_delta = get_SR_and_Errors_from_file(file_path)
Rytov_135 = [0.03, 0.052, 0.091]
stability_135 = [80, 50, 20]
stability_135_delta = [3, 4, 3]

# Data for L=270m
SR_270 = [88, 80, 70]
SR_270_delta = [7, 7, 8]
file_path = './arrays_SR_L270.npy'
SR_270, SR_270_delta = get_SR_and_Errors_from_file(file_path)
Rytov_270 = [0.05, 0.1, 0.15]
stability_270 = [78, 48, 18]
stability_270_delta = [4, 3, 3]

# Data for L=540m
SR_540 = [89, 81, 69]
SR_540_delta = [7, 8, 7]
file_path = './arrays_SR_L540.npy'
SR_540, SR_540_delta = get_SR_and_Errors_from_file(file_path)
Rytov_540 = [0.086, 0.161, 0.28]
stability_540 = [75, 45, 15]
stability_540_delta = [3, 3, 4]

# Define plot styles
marker_size = 8
line_width = 2
error_bar_capsize = 5
error_bar_width = 1.5

# Plot 1: Rytov variance vs SR values for L = 135m, 270m, 540m
plt.figure(figsize=(10, 6))
plt.errorbar(SR_135, Rytov_135, yerr=None, fmt='-o', markersize=marker_size, capsize=error_bar_capsize,
             elinewidth=error_bar_width, linewidth=line_width, label='L = 135m')
plt.errorbar(SR_270, Rytov_270, yerr=None, fmt='-s', markersize=marker_size, capsize=error_bar_capsize,
             elinewidth=error_bar_width, linewidth=line_width, label='L = 270m')
plt.errorbar(SR_540, Rytov_540, yerr=None, fmt='-^', markersize=marker_size, capsize=error_bar_capsize,
             elinewidth=error_bar_width, linewidth=line_width, label='L = 540m')
plt.title('Rytov Variance vs SR Values', fontsize=16)
plt.xlabel('SR Values', fontsize=14)
plt.ylabel('Rytov Variance', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()

# Plot 2: SR values vs Rytov variance with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(Rytov_135, SR_135, yerr=SR_135_delta, fmt='-o', markersize=marker_size, capsize=error_bar_capsize,
             elinewidth=error_bar_width, linewidth=line_width, label='L = 135m')
plt.errorbar(Rytov_270, SR_270, yerr=SR_270_delta, fmt='-s', markersize=marker_size, capsize=error_bar_capsize,
             elinewidth=error_bar_width, linewidth=line_width, label='L = 270m')
plt.errorbar(Rytov_540, SR_540, yerr=SR_540_delta, fmt='-^', markersize=marker_size, capsize=error_bar_capsize,
             elinewidth=error_bar_width, linewidth=line_width, label='L = 540m')
plt.title('SR Values vs Rytov Variance (with Errors)', fontsize=16)
plt.xlabel('Rytov Variance', fontsize=14)
plt.ylabel('SR Values', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()

# Plot 3: Optical trefoil knot stability vs SR values with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(SR_135, stability_135, yerr=stability_135_delta, fmt='-o', markersize=marker_size,
             capsize=error_bar_capsize,
             elinewidth=error_bar_width, linewidth=line_width, label='L = 135m')
plt.errorbar(SR_270, stability_270, yerr=stability_270_delta, fmt='-s', markersize=marker_size,
             capsize=error_bar_capsize,
             elinewidth=error_bar_width, linewidth=line_width, label='L = 270m')
plt.errorbar(SR_540, stability_540, yerr=stability_540_delta, fmt='-^', markersize=marker_size,
             capsize=error_bar_capsize,
             elinewidth=error_bar_width, linewidth=line_width, label='L = 540m')
plt.title('Optical Trefoil Knot Stability vs SR Values (with Errors)', fontsize=16)
plt.xlabel('SR Values', fontsize=14)
plt.ylabel('Stability (%)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()

# Plot 4: Optical trefoil knot stability vs Rytov variance with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(Rytov_135, stability_135, yerr=stability_135_delta, fmt='-o', markersize=marker_size,
             capsize=error_bar_capsize,
             elinewidth=error_bar_width, linewidth=line_width, label='L = 135m')
plt.errorbar(Rytov_270, stability_270, yerr=stability_270_delta, fmt='-s', markersize=marker_size,
             capsize=error_bar_capsize,
             elinewidth=error_bar_width, linewidth=line_width, label='L = 270m')
plt.errorbar(Rytov_540, stability_540, yerr=stability_540_delta, fmt='-^', markersize=marker_size,
             capsize=error_bar_capsize,
             elinewidth=error_bar_width, linewidth=line_width, label='L = 540m')
plt.title('Optical Trefoil Knot Stability vs Rytov Variance (with Errors)', fontsize=16)
plt.xlabel('Rytov Variance', fontsize=14)
plt.ylabel('Stability (%)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()

# Show all plots
plt.show()
