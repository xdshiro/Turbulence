import matplotlib.pyplot as plt
import numpy as np

# Real Data
sr_values_new = [0.97, 0.94, 0.85, 0.76, 0.68]
sr_values = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
Rytov = [0.01, 0.02, 0.05, 0.1, 0.15]
Cn2 = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
proposed_simulation_new = [94, 82, 100-34*2, 100-41*2, 100-48*2]  # values are not precise
proposed_simulation = [57, 30.5, 17, 9.5, 3, 1.5]  # values are not precise


# Function to calculate confidence interval
def confidence_interval(p, n, z=1.96):
    p = p / 100  # Convert percentage to proportion
    se = np.sqrt((p * (1 - p)) / n)
    ci = z * se
    return ci * 100  # Convert back to percentage

# Number of samples
n_samples = 300
n_samples_new = 50

# Calculate confidence intervals
proposed_simulation_new_ci = [confidence_interval(p, n_samples_new) for p in proposed_simulation_new]
proposed_simulation_ci = [confidence_interval(p, n_samples) for p in proposed_simulation]

plt.figure(figsize=(8, 6))

plt.plot(sr_values_new, proposed_simulation_new, 'go-', label='Proposed Method (Simulation New)')
plt.fill_between(sr_values_new, np.array(proposed_simulation_new) - proposed_simulation_new_ci,
                 np.array(proposed_simulation_new) + proposed_simulation_new_ci, color='green', alpha=0.2)

plt.plot(sr_values, proposed_simulation, 'ro-', label='Proposed Method (Simulation)')
plt.fill_between(sr_values, np.array(proposed_simulation) - proposed_simulation_ci,
                 np.array(proposed_simulation) + proposed_simulation_ci, color='red', alpha=0.2)

plt.xlabel('SR')
plt.ylabel('Recovered Knots (%)')
plt.title('Recovered Optimized Hopfs vs SR')
plt.legend()
plt.grid(True)

# Reverse the x-axis
plt.gca().invert_xaxis()
plt.tight_layout()
# Show the plot
plt.show()