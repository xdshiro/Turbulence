import matplotlib.pyplot as plt
import numpy as np

# Real Data
Rytov = [0.05, 0.075]
proposed_simulation_ps1 = [20, 5.5]
proposed_simulation_ps3 = [26, 14.5]
proposed_simulation_ps5 = [37.5, 19.5]
proposed_simulation_ps20 = [34, 17.5]

# Function to calculate confidence interval
def confidence_interval(p, n, z=1.96):
    p = p / 100  # Convert percentage to proportion
    se = np.sqrt((p * (1 - p)) / n)
    ci = z * se
    return ci * 100  # Convert back to percentage

# Number of samples
n_samples = 150

# Calculate confidence intervals
proposed_simulation_ci1 = [confidence_interval(p, n_samples) for p in proposed_simulation_ps1]
proposed_simulation_ci3 = [confidence_interval(p, n_samples) for p in proposed_simulation_ps3]
proposed_simulation_ci5 = [confidence_interval(p, n_samples) for p in proposed_simulation_ps5]
proposed_simulation_ci20 = [confidence_interval(p, n_samples) for p in proposed_simulation_ps20]

# Prepare data for plotting
data_0_05 = [proposed_simulation_ps1[0], proposed_simulation_ps3[0], proposed_simulation_ps5[0], proposed_simulation_ps20[0]]
data_0_075 = [proposed_simulation_ps1[1], proposed_simulation_ps3[1], proposed_simulation_ps5[1], proposed_simulation_ps20[1]]

ci_0_05 = [proposed_simulation_ci1[0], proposed_simulation_ci3[0], proposed_simulation_ci5[0], proposed_simulation_ci20[0]]
ci_0_075 = [proposed_simulation_ci1[1], proposed_simulation_ci3[1], proposed_simulation_ci5[1], proposed_simulation_ci20[1]]

labels = ['Simulation_ps1', 'Simulation_ps3', 'Simulation_ps5', 'Simulation_ps20']
positions = np.arange(len(labels)) + 1

# Define colors for each simulation type
colors = ['red', 'blue', 'green', 'purple']

# Create the figure and the axes
fig, axes = plt.subplots(1, 2, figsize=(9, 6), sharey=True)

# Plot the points with error bars and horizontal lines for Rytov = 0.05
for i, (data, ci, color) in enumerate(zip(data_0_05, ci_0_05, colors)):
    axes[0].errorbar(positions[i], data, yerr=ci, fmt='o', color=color, capsize=5, capthick=2, label=labels[i])
    axes[0].hlines(data, positions[i] - 1.2, positions[i] + 1.2, colors=color, lw=0.3)

axes[0].set_title('Rytov = 0.05')
axes[0].set_xticks(positions)
axes[0].set_xticklabels(labels, rotation=45)
axes[0].set_ylabel('Recovered Knots (%)')

# Plot the points with error bars and horizontal lines for Rytov = 0.075
for i, (data, ci, color) in enumerate(zip(data_0_075, ci_0_075, colors)):
    axes[1].errorbar(positions[i], data, yerr=ci, fmt='o', color=color, capsize=5, capthick=2, label=labels[i])
    axes[1].hlines(data, positions[i] - 1.2, positions[i] + 1.2, colors=color, lw=0.3)

axes[1].set_title('Rytov = 0.075')
axes[1].set_xticks(positions)
axes[1].set_xticklabels(labels, rotation=45)

# Add a main title to the figure
fig.suptitle('Recovered Optimized Trefoils Comparison for Different Rytov Values')

# Add legends
axes[0].legend()
axes[1].legend()

# Show the plot
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.ylim([0, 50])
plt.show()