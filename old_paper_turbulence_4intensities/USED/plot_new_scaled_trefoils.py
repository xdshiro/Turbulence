import matplotlib.pyplot as plt
import numpy as np

# Real Data
# sr_values = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
Rytov = [0.025, 0.05, 0.075]
# Cn2 = [4.7e-15, 9.3e-15, 2.3e-14, 4.7e-14, 9.3e-14]

proposed_simulation = [65, 26, 13]  # values are not precise
proposed_simulation_scaled = [63, 23, 14]  # values are not precise


# Function to calculate confidence interval
def confidence_interval(p, n, z=1.96):
    p = p / 100  # Convert percentage to proportion
    se = np.sqrt((p * (1 - p)) / n)
    ci = z * se
    return ci * 100  # Convert back to percentage

# Number of samples
n_samples = 150

# Calculate confidence intervals
proposed_simulation_ci = [confidence_interval(p, n_samples) for p in proposed_simulation]
proposed_simulation_scaled_ci = [confidence_interval(p, n_samples) for p in proposed_simulation_scaled]


plt.figure(figsize=(8, 6))

plt.plot(Rytov, proposed_simulation, 'ro-', label='Proposed Method (100m)')
plt.fill_between(Rytov, np.array(proposed_simulation) - proposed_simulation_ci,
                 np.array(proposed_simulation) + proposed_simulation_ci, color='red', alpha=0.2)

plt.plot(Rytov, proposed_simulation_scaled, 'bo-', label='Proposed Method (1.98m)')
plt.fill_between(Rytov, np.array(proposed_simulation_scaled) - proposed_simulation_scaled_ci,
                 np.array(proposed_simulation_scaled) + proposed_simulation_scaled_ci, color='blue', alpha=0.2)


plt.xlabel('SR')
plt.ylabel('Recovered Knots (%)')
plt.title('Recovered Optimized Trefoils vs SR')
plt.legend()
plt.grid(True)
plt.ylim(0, 100)
# Reverse the x-axis
# plt.gca().invert_xaxis()
plt.tight_layout()
# Show the plot
plt.show()