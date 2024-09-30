import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Data for L=135m
SR_135 = [87, 79, 71]
Rytov_135 = [0.03, 0.052, 0.091]
stability_135 = [80, 50, 20]

# Data for L=270m
SR_270 = [88, 80, 70]
Rytov_270 = [0.05, 0.1, 0.15]
stability_270 = [78, 48, 18]

# Data for L=540m
SR_540 = [89, 81, 69]
Rytov_540 = [0.086, 0.161, 0.28]
stability_540 = [75, 45, 15]

# Confidence interval calculation function
def confidence_interval(p, n, confidence_level=0.95):
    z = norm.ppf((1 + confidence_level) / 2)  # Calculate the z-value dynamically based on confidence level
    p = np.array(p) / 100  # Convert percentage to proportion
    se = np.sqrt((p * (1 - p)) / n)  # Standard error
    ci = z * se  # Confidence interval
    return ci * 100  # Convert back to percentage

# Number of samples and confidence level
n_samples = 100
confidence_level = 0.95

# Calculate confidence intervals for stability
stability_135_ci = confidence_interval(stability_135, n_samples, confidence_level)
stability_270_ci = confidence_interval(stability_270, n_samples, confidence_level)
stability_540_ci = confidence_interval(stability_540, n_samples, confidence_level)

# Define plot styles
marker_size = 8
line_width = 2
error_bar_capsize = 5
error_bar_width = 1.5

# Plot 1: Optical trefoil knot stability vs SR values with confidence intervals
plt.figure(figsize=(10, 6))
plt.errorbar(SR_135, stability_135, yerr=stability_135_ci, fmt='-o', markersize=marker_size, capsize=error_bar_capsize,
             elinewidth=error_bar_width, linewidth=line_width, label='L = 135m')
plt.errorbar(SR_270, stability_270, yerr=stability_270_ci, fmt='-s', markersize=marker_size, capsize=error_bar_capsize,
             elinewidth=error_bar_width, linewidth=line_width, label='L = 270m')
plt.errorbar(SR_540, stability_540, yerr=stability_540_ci, fmt='-^', markersize=marker_size, capsize=error_bar_capsize,
             elinewidth=error_bar_width, linewidth=line_width, label='L = 540m')
plt.title('Optical Trefoil Knot Stability vs SR Values (with Confidence Intervals)', fontsize=16)
plt.xlabel('SR Values', fontsize=14)
plt.ylabel('Stability (%)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()

# Plot 2: Optical trefoil knot stability vs Rytov variance with confidence intervals
plt.figure(figsize=(10, 6))
plt.errorbar(Rytov_135, stability_135, yerr=stability_135_ci, fmt='-o', markersize=marker_size, capsize=error_bar_capsize,
             elinewidth=error_bar_width, linewidth=line_width, label='L = 135m')
plt.errorbar(Rytov_270, stability_270, yerr=stability_270_ci, fmt='-s', markersize=marker_size, capsize=error_bar_capsize,
             elinewidth=error_bar_width, linewidth=line_width, label='L = 270m')
plt.errorbar(Rytov_540, stability_540, yerr=stability_540_ci, fmt='-^', markersize=marker_size, capsize=error_bar_capsize,
             elinewidth=error_bar_width, linewidth=line_width, label='L = 540m')
plt.title('Optical Trefoil Knot Stability vs Rytov Variance (with Confidence Intervals)', fontsize=16)
plt.xlabel('Rytov Variance', fontsize=14)
plt.ylabel('Stability (%)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()

# Show all plots
plt.show()