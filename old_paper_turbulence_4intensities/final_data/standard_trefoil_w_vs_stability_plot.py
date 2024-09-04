w = [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.5]
stab_005 = [0, 0, 9, 13, 19,  9, 0]
stab_005s = [0, 2, 11, 7, 4, 1, 2]
stab_0025 = [0, 5, 23, 32, 18, 24, 4]
stab_0025s = [0, 10, 3, 8, 10, 4, 1]
w = [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.5]

import matplotlib.pyplot as plt

# Data
stability_rytov005_final = [0, 0, 1, 13, 16, 23, 9, 1]
stability_rytov0025_final = [0, 0, 10, 24, 36, 23, 26, 4]
w_values = [1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.5]  # Assuming these are the values for parameter w  # Assuming these are the values for parameter w

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(w_values, stability_rytov005_final, marker='o', label='Rytov 0.05 (SR=0.88)')
plt.plot(w_values, stability_rytov0025_final, marker='s', label='Rytov 0.025 (SR=0.94)')

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
