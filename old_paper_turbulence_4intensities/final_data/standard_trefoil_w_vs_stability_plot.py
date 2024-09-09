w = [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.5]
stab_005 = [0 + 0, 15, 9 + 13 + 14, 13, 19 + 6,  9 + 7, 0 + 1]
stab_005s = [0 + 1, 3, 11 + 6 + 7, 7, 4 + 7, 1 + 4, 2 + 1]
stab_0025 = [0 + 0 + 0, 39 + 36, 36 + 43, 32, 18 + 22 + 24, 24 + 22 + 22, 4 + 2]
stab_0025s = [0 + 1 + 0, 6 + 10, 6 + 8, 8, 10 + 3 + 4, 4 + 4 + 4, 1 + 3]
w = [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.5]
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Data
stability_rytov005_final = [0, 0, 1, 13, 16, 23, 9, 1]
stability_rytov0025_final = [0, 0, 10, 24, 36, 23, 26, 4]


stability_rytov005_final = [0, 16, (14 + 16 + 18) / 3, 16.33333333, (21 + 6) / 2, (9 + 9) / 2, (1 + 1) / 2]
stability_rytov0025_final = [0, (42 + 41) / 2, (39 + 47) / 2,
                             35.33333333, (23 + 23 + 26) / 3, (22 + 24 + 24) / 3, (4 + 4) / 2]
w_values = [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.5]  # Assuming these are the values for parameter w  # Assuming these are the values for parameter w

print("stabilities in Rytov 0.05 vs w: ", stability_rytov005_final)
print("stabilities in Rytov 0.025 vs w: ", stability_rytov0025_final)
print("w values: ", w_values)
def confidence_interval(p, n, confidence_level=0.95):
    z = norm.ppf((1 + confidence_level) / 2)  # Calculate the z-value dynamically based on confidence level
    p = np.array(p) / 100  # Convert percentage to proportion
    se = np.sqrt((p * (1 - p)) / n)  # Standard error
    ci = z * se  # Confidence interval
    return ci * 100  # Convert back to percentage
# Parameters
n_samples = 300
confidence_level = 0.95
# Plotting
# Calculate confidence intervals for both datasets
stability_rytov005_ci = confidence_interval(stability_rytov005_final, n_samples, confidence_level)
stability_rytov0025_ci = confidence_interval(stability_rytov0025_final, n_samples, confidence_level)

# Plotting
plt.figure(figsize=(10, 6))

# Plot the central lines for each dataset
plt.plot(w_values, stability_rytov005_final, marker='o', label='Rytov 0.05 (SR=0.88)')
plt.plot(w_values, stability_rytov0025_final, marker='s', label='Rytov 0.025 (SR=0.94)')

# Fill areas between the confidence intervals
plt.fill_between(w_values,
                 np.array(stability_rytov005_final) - stability_rytov005_ci,
                 np.array(stability_rytov005_final) + stability_rytov005_ci,
                 color='blue', alpha=0.2)

plt.fill_between(w_values,
                 np.array(stability_rytov0025_final) - stability_rytov0025_ci,
                 np.array(stability_rytov0025_final) + stability_rytov0025_ci,
                 color='green', alpha=0.2)

# Titles and labels
plt.title('Stability of Optical Trefoil vs Parameter w', fontsize=16)
plt.xlabel('Parameter w', fontsize=14)
plt.ylabel('Percentage of Knots Preserved Their Topology', fontsize=14)

# Font settings for ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Grid and legend
plt.grid(True)
plt.legend(fontsize=12)

# Show plot
plt.show()