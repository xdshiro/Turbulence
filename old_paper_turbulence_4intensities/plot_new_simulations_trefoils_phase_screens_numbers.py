import matplotlib.pyplot as plt
import numpy as np

# Real Data
# sr_values = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
Rytov = [0.05, 0.075]
# Cn2 = [4.7e-15, 9.3e-15, 2.3e-14, 4.7e-14, 9.3e-14]
proposed_simulation_ps1 = [20, 5.5]  # values are not precise
proposed_simulation_ps3 = [26, 13.5]  # values are not precise
proposed_simulation_ps5 = [37.5, 20.5]  # values are not precise.
proposed_simulation_ps20 = [34, 17.5]  # values are not precise

# Function to calculate confidence interval
def confidence_interval(p, n, z=1.96):
    p = p / 100  # Convert percentage to proportion
    se = np.sqrt((p * (1 - p)) / n)
    ci = z * se
    return ci * 100  # Convert back to percentage

# Number of samples
n_samples = 100

# Calculate confidence intervals
proposed_simulation_ci1 = [confidence_interval(p, n_samples) for p in proposed_simulation_ps1]
proposed_simulation_ci3 = [confidence_interval(p, n_samples) for p in proposed_simulation_ps3]
proposed_simulation_ci5 = [confidence_interval(p, n_samples) for p in proposed_simulation_ps5]
proposed_simulation_ci20 = [confidence_interval(p, n_samples) for p in proposed_simulation_ps20]

plt.figure(figsize=(8, 6))

plt.plot(Rytov, proposed_simulation_ps1, 'ro-', label='Proposed Method (Simulation_ps1)')
plt.fill_between(Rytov, np.array(proposed_simulation_ps1) - proposed_simulation_ci1,
                 np.array(proposed_simulation_ps1) + proposed_simulation_ci1, color='red', alpha=0.2)

plt.plot(Rytov, proposed_simulation_ps3, 'bo-', label='Proposed Method (Simulation_ps3)')
plt.fill_between(Rytov, np.array(proposed_simulation_ps3) - proposed_simulation_ci3,
                 np.array(proposed_simulation_ps3) + proposed_simulation_ci3, color='blue', alpha=0.2)

plt.plot(Rytov, proposed_simulation_ps5, 'go-', label='Proposed Method (Simulation_ps5)')
plt.fill_between(Rytov, np.array(proposed_simulation_ps5) - proposed_simulation_ci5,
                 np.array(proposed_simulation_ps5) + proposed_simulation_ci5, color='green', alpha=0.2)

plt.plot(Rytov, proposed_simulation_ps20, 'yo-', label='Proposed Method (Simulation_ps20)')
plt.fill_between(Rytov, np.array(proposed_simulation_ps20) - proposed_simulation_ci20,
                 np.array(proposed_simulation_ps20) + proposed_simulation_ci20, color='yellow', alpha=0.2)

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