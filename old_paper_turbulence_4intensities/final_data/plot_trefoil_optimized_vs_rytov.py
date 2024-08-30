import matplotlib.pyplot as plt

# Data
stability_optimized_center = [92, 61, 27]
stability_optimized_center_delta = [6, 10, 9]
stability_optimized = [69, 40, 16, 4, 3]
stability_optimized_delta = [9, 12, 8, 3, 1]

stability_optimized_center_final = [95, 64, 31]
stability_optimized_final = [74, 23, 9, 6, 3]
Rytov_values = [0.025, 0.05, 0.1, 0.15, 0.2]  # Assuming these are the values for parameter w
Rytov_values_center = [0.025, 0.05, 0.1]  # Assuming these are the values for parameter w
SR_values = [0.9394629805988903, 0.8846710259527795, 0.7929402897279878, 0.6974844551466605, 0.6396120144621134]
SR_values_center = [0.9394629805988903, 0.8846710259527795, 0.7929402897279878]
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(Rytov_values, stability_optimized_final, marker='o', label='before')
plt.plot(Rytov_values_center, stability_optimized_center_final, marker='s', label='center')

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
plt.plot(SR_values, stability_optimized_final, marker='o', label='before')
plt.plot(SR_values_center, stability_optimized_center_final, marker='s', label='center')

# Titles and labels
plt.title('Stability of Optical Trefoil vs SR', fontsize=16)
plt.xlabel('SR', fontsize=14)
plt.ylabel('Percentage of Knots Preserved Their Topology', fontsize=14)

# Font settings for ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Grid and legend
plt.grid(True)
plt.legend(fontsize=12)

# Show plot
plt.show()