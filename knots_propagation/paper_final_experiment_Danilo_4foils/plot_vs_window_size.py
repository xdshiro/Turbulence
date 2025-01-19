import matplotlib.pyplot as plt

# Data
accuracies = [20.7, 23.0, 24.6, 25.4, 26.7, 27.0, 27.8, 28.5, 29.3, 28.9, 30.4, 27.6]
window_size = [4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5]
accuracies = [20.7, 23.0, 24.0, 24.6, 24.2, 22.7, 21.5, 19.7]
window_size = [4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]
measured_window_size = 4.6

# Find the index of the measured window size
measured_index = window_size.index(measured_window_size)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(window_size, accuracies, marker='o', label="Accuracies", color='blue', linewidth=2)

# Highlight the measured window size point
plt.scatter(
    window_size[measured_index], accuracies[measured_index],
    color='red', label=f'Measured Window Size ({measured_window_size})', s=100, zorder=5
)

# Labels and title
plt.xlabel("Window Size", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=14)
plt.title("Accuracy vs Window Size", fontsize=16)

# Add grid
plt.grid(visible=True, linestyle='--', alpha=0.6)

# Add legend
plt.legend(fontsize=12)

# Show plot
plt.tight_layout()
plt.show()
