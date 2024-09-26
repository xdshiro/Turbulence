import matplotlib.pyplot as plt

# Data
length = [270, 130, 50]
SR = [0.78, 0.66, 0.44]
Scintillation = [0.043, 0.055, 0.089]

# Create a figure and a set of subplots
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot SR on the primary y-axis (left)
ax1.plot(length, SR, marker='o', color='b', label='SR')
ax1.set_xlabel('Length')
ax1.set_ylabel('SR', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True)  # Add grid to the primary y-axis

# Create a second y-axis sharing the same x-axis
ax2 = ax1.twinx()
ax2.plot(length, Scintillation, marker='o', color='r', label='Scintillation')
ax2.set_ylabel('Scintillation', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Adding titles and legends
plt.title('SR and Scintillation vs Length')

# Combine legends from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='best')

# Show the plot
plt.show()