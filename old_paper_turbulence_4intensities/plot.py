import matplotlib.pyplot as plt
import numpy as np

# Real Data
sr_values = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
proposed_simulation = [79, 39, 16, 7.5, 5.2, 2.5]
mr_dennis_simulation = [41, 16, 5, 2, 0, 0]
proposed_experiment = [71, 32, 10, 6, 2, 0]
mr_dennis_experiment = [42, 14, 1, 0, 0, 0]

# New data for the slow camera assumption
sr_values_new = [0.95, 0.9, 0.85, 0.8]
slow_camera_simulation = [56.2, 20.6, 7.41, 3.37]

# Function to calculate confidence interval
def confidence_interval(p, n, z=1.96):
    p = p / 100  # Convert percentage to proportion
    se = np.sqrt((p * (1 - p)) / n)
    ci = z * se
    return ci * 100  # Convert back to percentage

# Number of samples
n_samples = 300
n_samples_new = 300

# Calculate confidence intervals
proposed_simulation_ci = [confidence_interval(p, n_samples) for p in proposed_simulation]
mr_dennis_simulation_ci = [confidence_interval(p, n_samples) for p in mr_dennis_simulation]
proposed_experiment_ci = [confidence_interval(p, n_samples) for p in proposed_experiment]
mr_dennis_experiment_ci = [confidence_interval(p, n_samples) for p in mr_dennis_experiment]
slow_camera_simulation_ci = [confidence_interval(p, n_samples_new) for p in slow_camera_simulation]

# Plot
plt.figure(figsize=(8, 6))

plt.plot(sr_values, proposed_simulation, 'ro-', label='Proposed Method (Simulation)')
plt.fill_between(sr_values, np.array(proposed_simulation) - proposed_simulation_ci,
                 np.array(proposed_simulation) + proposed_simulation_ci, color='red', alpha=0.2)

plt.plot(sr_values, mr_dennis_simulation, 'bo-', label='M.R. Dennis et al. (Simulation)')
plt.fill_between(sr_values, np.array(mr_dennis_simulation) - mr_dennis_simulation_ci,
                 np.array(mr_dennis_simulation) + mr_dennis_simulation_ci, color='blue', alpha=0.2)

plt.plot(sr_values, proposed_experiment, 'o--', color='orange', label='Proposed Method (Experiment)')
plt.fill_between(sr_values, np.array(proposed_experiment) - proposed_experiment_ci,
                 np.array(proposed_experiment) + proposed_experiment_ci, color='orange', alpha=0.2)

plt.plot(sr_values, mr_dennis_experiment, 'o--', color='green', label='M.R. Dennis et al. (Experiment)')
plt.fill_between(sr_values, np.array(mr_dennis_experiment) - mr_dennis_experiment_ci,
                 np.array(mr_dennis_experiment) + mr_dennis_experiment_ci, color='green', alpha=0.2)

plt.plot(sr_values_new, slow_camera_simulation, 'mo-', label='Slow Camera (Simulation)')
plt.fill_between(sr_values_new, np.array(slow_camera_simulation) - slow_camera_simulation_ci,
                 np.array(slow_camera_simulation) + slow_camera_simulation_ci, color='magenta', alpha=0.2)

# Customize the plot
plt.xlabel('SR')
plt.ylabel('Recovered Knots (%)')
plt.title('Recovered Knots vs SR')
plt.legend()
plt.grid(True)

# Reverse the x-axis
plt.gca().invert_xaxis()
plt.tight_layout()
# Show the plot
plt.show()