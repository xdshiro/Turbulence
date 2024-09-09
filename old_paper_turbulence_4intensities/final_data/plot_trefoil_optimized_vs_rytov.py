import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
def confidence_interval(p, n, confidence_level=0.95):
    z = norm.ppf((1 + confidence_level) / 2)  # Calculate the z-value dynamically based on confidence level
    p = np.array(p) / 100  # Convert percentage to proportion
    se = np.sqrt((p * (1 - p)) / n)  # Standard error
    ci = z * se  # Confidence interval
    return ci * 100  # Convert back to percentage
# Data 300
stability_optimized_center = np.array([92 + 97 + 93, 61 + 66 + 62, 27 + 37 + 35, 17 + 18 + 22, 8 + 4 + 5])
stability_optimized_center_delta = np.array([6 + 1 + 3, 10 + 8 + 8, 9 + 9 + 6, 4 + 7 + 5, 5 + 4 + 3])
stability_optimized = np.array([69, 40, 17, 4, 3])
stability_optimized_delta = np.array([9, 12, 8, 3, 1])
stability_standard_12 = np.array([32 + 25 + 39, 13 + 17 + 12, 4 + 7 + 4, 1 + 1 + 1, 0 + 0 + 1])
stability_standard_12_delta = np.array([8 + 9 + 4, 7 + 3 + 4, 0 + 2 + 1, 0 + 3 + 2, 0 + 1 + 1])
stability_standard_115 = np.array([43 * 3, 16 * 3, 4 + 7 + 4, 1 + 1 + 1, 0 + 0 + 1])
stability_standard_115_delta = np.array([0 * 3, 0 * 3, 0 + 2 + 1, 0 + 3 + 2, 0 + 1 + 1])
stability_dennis = np.array([75 + 71 + 80, 42 + 43 + 30, 8 + 8 + 13, 8 + 4 + 1, 1 + 2 + 3])
stability_dennis_delta = np.array([4 + 7 + 2, 7 + 4 + 5, 3 + 5 + 5, 3 + 1 + 5, 0 + 1 + 2])
SR_values = np.array([0.9394, 0.8846, 0.7929, 0.6974, 0.6396])

sr_values = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]

stability_optimized_center_final = (stability_optimized_center + stability_optimized_center_delta // 2) / 3
stability_standard_115_final = (stability_standard_115 + stability_standard_115_delta // 2) / 3
stability_dennis_final = (stability_dennis + stability_dennis_delta // 2) / 3

print('optimized: ', stability_optimized_center_final)
print('standard_1.15: ', stability_standard_115_final)
print('dennis:', stability_dennis_final)
# stability_optimized_center_final = [95, 66, 31, 19, 10]
stability_optimized_final = [73, 46, 21, 5, 3]
# stability_standard_12_final = [36, 16, 4, 1, 0]
# stability_dennis_final = [77, 45, 9, 9, 1]
proposed_simulation = [79, 39, 16, 7.5, 5.2, 2.5]  # OLD not presize

Rytov_values = [0.025, 0.05, 0.1, 0.15, 0.2]  # Assuming these are the values for parameter w

n_samples = 300
confidence_level = 0.95
stability_optimized_center_ci = confidence_interval(stability_optimized_center_final, n_samples, confidence_level)
stability_optimized_ci = confidence_interval(stability_optimized_final, n_samples, confidence_level)
stability_standard_115_ci = confidence_interval(stability_standard_115_final, n_samples, confidence_level)
stability_dennis_ci = confidence_interval(stability_dennis_final, n_samples, confidence_level)

# Plotting
plt.figure(figsize=(10, 6))
# plt.plot(Rytov_values, stability_optimized_final, ls='--', marker='o', label='start of the knot')

# plt.plot(Rytov_values, stability_optimized_center_final, marker='s', label='center of the knot')
# plt.plot(Rytov_values, stability_standard_12_final, marker='x', label='standard_12')
# plt.plot(Rytov_values, stability_dennis_final, marker='o', label='dennis')

# Plot the central line for each dataset
plt.plot(Rytov_values, stability_optimized_center_final, marker='s', label='center of the knot')
# plt.plot(Rytov_values, stability_optimized_final, marker='s', label='start of the knot')
plt.plot(Rytov_values, stability_standard_115_final, marker='x', label='standard_1.15')
plt.plot(Rytov_values, stability_dennis_final, marker='o', label='dennis')


plt.fill_between(Rytov_values,
                 np.array(stability_optimized_center_final) - stability_optimized_center_ci,
                 np.array(stability_optimized_center_final) + stability_optimized_center_ci,
                 color='green', alpha=0.2)

# plt.fill_between(Rytov_values,
#                  np.array(stability_optimized_final) - stability_optimized_ci,
#                  np.array(stability_optimized_final) + stability_optimized_ci,
#                  color='red', alpha=0.2)

plt.fill_between(Rytov_values,
                 np.array(stability_standard_115_final) - stability_standard_115_ci,
                 np.array(stability_standard_115_final) + stability_standard_115_ci,
                 color='red', alpha=0.2)

plt.fill_between(Rytov_values,
                 np.array(stability_dennis_final) - stability_dennis_ci,
                 np.array(stability_dennis_final) + stability_dennis_ci,
                 color='orange', alpha=0.2)

# Titles and labels
plt.title('Stability of Optical Trefoil vs Rytov', fontsize=16)
plt.xlabel('Rytov', fontsize=14)
plt.ylabel('Percentage of Knots Preserved Their Topology', fontsize=14)

# Font settings for ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Grid and legend
plt.grid(True)
plt.legend(fontsize=12)

# Show plot
plt.ylim(0, 100)
plt.show()
exit()

plt.figure(figsize=(10, 6))
plt.plot(SR_values[::-1], stability_optimized_final[::-1], marker='o', label='start of the knot')
plt.plot(SR_values[::-1], stability_optimized_center_final[::-1], marker='s', label='center of the knot')
plt.plot(sr_values, proposed_simulation, marker='x', label='arXiv simulation')
# Titles and labels
plt.title('Stability of Optical Trefoil vs SR', fontsize=16)
plt.xlabel('SR', fontsize=14)
plt.ylabel('Percentage of Knots Preserved Their Topology', fontsize=14)

# Font settings for ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.gca().invert_xaxis()
# Grid and legend
plt.grid(True)
plt.legend(fontsize=12)

# Show plot
plt.show()