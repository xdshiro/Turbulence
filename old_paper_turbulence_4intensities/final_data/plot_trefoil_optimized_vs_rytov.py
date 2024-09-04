import matplotlib.pyplot as plt

# Data
stability_optimized_center = [92, 61, 27, 17, 8]
stability_optimized_center_delta = [6, 10, 9, 4, 5]
stability_optimized = [69, 40, 17, 4, 3]
stability_optimized_delta = [9, 12, 8, 3, 1]
stability_standard_12 = [32, 13, 00, 00, 00]
stability_standard_12_delta = [8, 7, 00, 00, 00]
stability_dennis = [00, 00, 00, 00, 00]
stability_dennis_delta = [00, 00, 00, 00, 00]


stability_optimized_center_final = [95, 66, 31, 19, 10]
stability_optimized_final = [73, 46, 21, 5, 3]
stability_standard_12_final = [36, 16, 00, 00, 00]
stability_dennis_final = [00, 00, 00, 00, 00]
proposed_simulation = [79, 39, 16, 7.5, 5.2, 2.5]  # OLD not presize

Rytov_values = [0.025, 0.05, 0.1, 0.15, 0.2]  # Assuming these are the values for parameter w

SR_values = [0.9394, 0.8846, 0.7929, 0.6974, 0.6396]

sr_values = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(Rytov_values, stability_optimized_final, marker='o', label='start of the knot')
plt.plot(Rytov_values, stability_optimized_center_final, marker='s', label='center of the knot')
plt.plot(Rytov_values, stability_standard_12_final, marker='x', label='standard_12')
plt.plot(Rytov_values, stability_dennis_final, marker='o', label='dennis')

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
plt.show()

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