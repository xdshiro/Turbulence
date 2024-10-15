import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm



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

def confidence_interval(p, n, confidence_level=0.95):
    z = norm.ppf((1 + confidence_level) / 2)  # Calculate the z-value dynamically based on confidence level
    p = np.array(p) / 100  # Convert percentage to proportion
    se = np.sqrt((p * (1 - p)) / n)  # Standard error
    ci = z * se  # Confidence interval
    return ci * 100  # Convert back to percentage

# Number of samples and confidence level
n_samples = 125
confidence_level = 0.95

SR_135 = [81, 75, 64]
SR_135_delta = [3, 7, 8]
# Data for L=135m
# file_path = './arrays_SR_L135.0.npy'
# SR_135, SR_135_delta = get_SR_and_Errors_from_file(file_path)
Rytov_135 = [0.03, 0.052, 0.091]
stability_135 = [89, 70, 46]
stability_135_delta = [3, 4, 3]

# Data for L=270m
SR_270 = [88, 80, 70]
SR_270_delta = [7, 7, 8]
# file_path = './arrays_SR_L270.npy'
# SR_270, SR_270_delta = get_SR_and_Errors_from_file(file_path)
Rytov_270 = [0.05, 0.1, 0.15]
stability_270 = [78, 48, 18]
stability_270_delta = [4, 3, 3]

# Data for L=540m
SR_540 = [89, 81, 69]
SR_540_delta = [7, 8, 7]
# file_path = './arrays_SR_L540.npy'
# SR_540, SR_540_delta = get_SR_and_Errors_from_file(file_path)
Rytov_540 = [0.086, 0.161, 0.28]
stability_540 = [32, 7, 0]
stability_540_delta = [3, 3, 4]


stability_135_ci = confidence_interval(stability_135, n_samples, confidence_level)
stability_270_ci = confidence_interval(stability_270, n_samples, confidence_level)
stability_540_ci = confidence_interval(stability_540, n_samples, confidence_level)

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
# plt.errorbar(SR_135, stability_135, yerr=stability_135_delta, fmt='-o', markersize=marker_size,
#              capsize=error_bar_capsize,
#              elinewidth=error_bar_width, linewidth=line_width, label='L = 135m')
# plt.errorbar(SR_270, stability_270, yerr=stability_270_delta, fmt='-s', markersize=marker_size,
#              capsize=error_bar_capsize,
#              elinewidth=error_bar_width, linewidth=line_width, label='L = 270m')
# plt.errorbar(SR_540, stability_540, yerr=stability_540_delta, fmt='-^', markersize=marker_size,
#              capsize=error_bar_capsize,
#              elinewidth=error_bar_width, linewidth=line_width, label='L = 540m')
plt.fill_between(SR_135, np.array(stability_135) - stability_135_ci,
                 np.array(stability_135) + stability_135_ci, alpha=0.2, color='blue', label='L = 135m')
plt.plot(SR_135, stability_135, '-o', color='blue', label='L = 135m', linewidth=2)

plt.fill_between(SR_270, np.array(stability_270) - stability_270_ci,
                 np.array(stability_270) + stability_270_ci, alpha=0.2, color='green', label='L = 270m')
plt.plot(SR_270, stability_270, '-s', color='green', label='L = 270m', linewidth=2)

plt.fill_between(SR_540[:-1], np.array(stability_540[:-1]) - stability_540_ci[:-1],
                 np.array(stability_540[:-1]) + stability_540_ci[:-1], alpha=0.2, color='red', label='L = 540m')
plt.plot(SR_540[:-1], stability_540[:-1], '-^', color='red', label='L = 540m', linewidth=2)

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
# plt.errorbar(Rytov_135, stability_135, yerr=stability_135_delta, fmt='-o', markersize=marker_size,
#              capsize=error_bar_capsize,
#              elinewidth=error_bar_width, linewidth=line_width, label='L = 135m')
# plt.errorbar(Rytov_270, stability_270, yerr=stability_270_delta, fmt='-s', markersize=marker_size,
#              capsize=error_bar_capsize,
#              elinewidth=error_bar_width, linewidth=line_width, label='L = 270m')
# plt.errorbar(Rytov_540, stability_540, yerr=stability_540_delta, fmt='-^', markersize=marker_size,
#              capsize=error_bar_capsize,
#              elinewidth=error_bar_width, linewidth=line_width, label='L = 540m')
plt.fill_between(Rytov_135, np.array(stability_135) - stability_135_ci,
                 np.array(stability_135) + stability_135_ci, alpha=0.2, color='blue', label='L = 135m')
plt.plot(Rytov_135, stability_135, '-o', color='blue', label='L = 135m', linewidth=2)

plt.fill_between(Rytov_270, np.array(stability_270) - stability_270_ci,
                 np.array(stability_270) + stability_270_ci, alpha=0.2, color='green', label='L = 270m')
plt.plot(Rytov_270, stability_270, '-s', color='green', label='L = 270m', linewidth=2)

plt.fill_between(Rytov_540[:-1], np.array(stability_540[:-1]) - stability_540_ci[:-1],
                 np.array(stability_540[:-1]) + stability_540_ci[:-1], alpha=0.2, color='red', label='L = 540m')
plt.plot(Rytov_540[:-1], stability_540[:-1], '-^', color='red', label='L = 540m', linewidth=2)


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
