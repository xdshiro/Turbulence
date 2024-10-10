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
stability_optimized_center_3int = np.array([
	(84 + 69) / 2, 85 / 2, 26 / 2, 5/2, 2/2
])
stability_optimized_center_3int = np.array([
	100, 92, 41, 21
])
stability_optimized_center_delta_3int = np.array([
	0, 0, 0, 0
])

# stability_optimized_center_3int = np.array([
# 	70, 19
# ])
# stability_optimized_center_delta_3int = np.array([
# 	11, 17
# ])

stability_optimized_center_final_3int = (stability_optimized_center_3int
                                    + stability_optimized_center_delta_3int // 2) / 1

stability_optimized_center = np.array([92 + 97 + 93, 61 + 66 + 62, 27 + 37 + 35, 17 + 18 + 22, 8 + 4 + 5])
stability_optimized_center_delta = np.array([6 + 1 + 3, 10 + 8 + 8, 9 + 9 + 6, 4 + 7 + 5, 5 + 4 + 3])


stability_optimized_center = np.array([92 + 97 + 93, 61 + 66 + 62, 17 + 18 + 22, 8 + 4 + 5])
stability_optimized_center_delta = np.array([6 + 1 + 3, 10 + 8 + 8, 4 + 7 + 5, 5 + 4 + 3])

#
stability_optimized_center_final = (stability_optimized_center + stability_optimized_center_delta // 2) / 3
print(stability_optimized_center_final)
print('optimized: ', stability_optimized_center_final_3int)

Rytov_values = [0.025, 0.05, 0.15, 0.2]  # Assuming these are the values for parameter w
Rytov_values_short = [0.025, 0.05, 0.15, 0.2]  # Assuming these are the values for parameter w

n_samples = 300
n_samples_short = 150
confidence_level = 0.95
stability_optimized_center_ci_3int = confidence_interval(stability_optimized_center_final_3int, n_samples_short, confidence_level)
stability_optimized_center_ci = confidence_interval(stability_optimized_center_final, n_samples, confidence_level)

# Plotting
plt.figure(figsize=(10, 6))
# plt.plot(Rytov_values, stability_optimized_final, ls='--', marker='o', label='start of the knot')

# plt.plot(Rytov_values, stability_optimized_center_final, marker='s', label='center of the knot')
# plt.plot(Rytov_values, stability_standard_12_final, marker='x', label='standard_12')
# plt.plot(Rytov_values, stability_dennis_final, marker='o', label='dennis')

# Plot the central line for each dataset
plt.plot(Rytov_values_short, stability_optimized_center_final_3int, marker='s', label='New optimization')
plt.plot(Rytov_values, stability_optimized_center_final, marker='s', label='Standard')

plt.fill_between(Rytov_values_short,
                 np.array(stability_optimized_center_final_3int) - stability_optimized_center_ci_3int,
                 np.array(stability_optimized_center_final_3int) + stability_optimized_center_ci_3int,
                 color='blue', alpha=0.2)
plt.fill_between(Rytov_values,
                 np.array(stability_optimized_center_final) - stability_optimized_center_ci,
                 np.array(stability_optimized_center_final) + stability_optimized_center_ci,
                 color='red', alpha=0.2)
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
