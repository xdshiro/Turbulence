import matplotlib.pyplot as plt
import numpy as np

# Scintillation - it should be \Sigma_I^2
# Rytov variance - it should be \Sigma_R^2

def Cn2(Cn2_initial, Rytov_initial, Rytov_final):
    return Cn2_initial * np.array(Rytov_final) / np.array(Rytov_initial)

def Rytov_initial(Cn2, Cn2_initial, Rytov_final):
    return np.array(Rytov_final) * np.array(Cn2_initial) / np.array(Cn2)

Scintillation_simulations = [0.009699, 0.018936, 0.039765, 0.066424, 0.090167]
Rytov_values = [0.025, 0.05, 0.1, 0.15, 0.2]
r0 = [0.08297, 0.05474, 0.03611, 0.02831, 0.02383]
w0_2_r0 = [0.10227, 0.15501, 0.23495, 0.29966, 0.35612] # (2w0/r0)
Cn2_values_simulations = [3.98e-15, 7.95e-15, 1.59e-14, 2.39e-14, 3.18e-14]

Rytov_values_simulations = [0.025, 0.05, 0.1, 0.15, 0.2]
# Error = 0.006277
# Error = 0.008857
# Error = 0.013176
# Error = 0.017393
# Error = 0.020631
Scintillation_experiment = [0.0068, 0.0168, 0.0272, 0.0418, 0.0541, 0.0682, 0.0794, 0.0931]
# Rytov_values_estimated_experiment_from_simulations = [0.017154, 0.044219, 0.069838, 0.103817, 0.126886, 0.162444, 0.188511, 0.236568]
Rytov_values_estimated_experiment_from_simulations = [
	0.017, 0.044, 0.070, 0.104, 0.127, 0.154, 0.177, 0.207
]

Cn2_values_experiment_from_simulations_precise = Cn2(
    Cn2_values_simulations[-1], Rytov_values_simulations[-1], Rytov_values_estimated_experiment_from_simulations
)
Cn2_values_experiment_from_simulations = [2.70e-15, 7.0e-15, 1.11e-14, 1.65e-14, 2.0e-14, 2.45e-14,
 2.80e-14, 3.30e-14
]

print('scintillation', np.array(Scintillation_experiment) * Cn2_values_experiment_from_simulations / Cn2_values_experiment_from_simulations_precise)
print((Cn2_values_experiment_from_simulations /
       Cn2_values_experiment_from_simulations_precise * 100
       ))
# Plot 1: Scintillation vs. Rytov variance
plt.figure(figsize=(10, 6))

# Plot Rytov vs. Scintillation (simulations)
plt.plot(Scintillation_simulations, Rytov_values_simulations, 'o-', label='Simulations (\Sigma_R^2 vs \Sigma_I^2)')

# Plot Rytov vs. Scintillation (experiment)
plt.plot(Scintillation_experiment, Rytov_values_estimated_experiment_from_simulations, 's-', label='Experiment (\Sigma_R^2 vs \Sigma_I^2)')

# Labels and legend
plt.xlabel('$\Sigma_I^2$ (Scintillation)', fontsize=14)
plt.ylabel('$\Sigma_R^2$ (Rytov variance)', fontsize=14)
plt.title('Rytov Variance vs. Scintillation', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)

# Show plot
plt.tight_layout()
plt.show()
#
# Plot 2: Scintillation vs. Cn2
# Create figure
plt.figure(figsize=(10, 6))

# Plot Scintillation vs. Cn2 (experiment)
plt.plot(Scintillation_experiment, Cn2_values_experiment_from_simulations, 'o-', markersize=8)

# Labels and legend
plt.xlabel('$σ_I^2$ (Scintillation)', fontsize=14)
plt.ylabel('$C_n^2$', fontsize=14)
plt.title('Experimental values of $C_n^2$ using numerical estimation', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
print(Scintillation_experiment)
print(Cn2_values_experiment_from_simulations)
# Increase font size for ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Adjust layout and show plot
plt.tight_layout()
plt.show()


exit()
import matplotlib.pyplot as plt

# Scintillation - it should be \Sigma_I^2
# Rytov variance - it should be \Sigma_R^2

def Cn2(Cn2_initial, Rytov_initial, Rytov_final):
    return Cn2_initial * np.array(Rytov_final) / np.array(Rytov_initial)

Scintillation_simulations = [0.009699, 0.018936, 0.039765, 0.066424, 0.090167]
Rytov_values_simulations = [0.025, 0.05, 0.1, 0.15, 0.2]
Cn2_values_simulations = [3.98e-15, 7.95e-15, 1.59e-14, 2.39e-14, 3.18e-14]
Scintillation_experiment = [0.0068, 0.0168, 0.0272, 0.0418, 0.0541, 0.0682, 0.0794, 0.0931]
Rytov_values_estimated_experiment_from_simulations = [0.017154, 0.044219, 0.069838, 0.103817, 0.126886, 0.162444, 0.188511, 0.236568]
Cn2_values_experiment_from_simulations = Cn2(
Cn2_values_simulations, Rytov_values_simulations, Rytov_values_estimated_experiment_from_simulations
)

# Updated data
# Scintillation_Sigma_I_squared = [0.01, 0.025, 0.5, 0.075, 0.1]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(Scintillation_Sigma_I_squared, Cn2_values_plane_wave_approximation, marker='o', label='Plane Wave Approximation')
plt.plot(Scintillation_Sigma_I_squared, Cn2_values_simulation_based, marker='s', label='Simulation-Based')

# Labels and title
plt.xlabel('Scintillation Index (σ_I^2)', fontsize=12)
plt.ylabel('C_n^2 (Refractive Index Structure Parameter)', fontsize=12)
plt.title('Comparison of C_n^2 Values: Plane Wave vs. Simulation-Based', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()

# Show plot
plt.show()