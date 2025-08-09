import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Set global font to Times New Roman
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 18
})

# Data
w_values = [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.5]
stability_rytov005_final = [0, 16, (14 + 16 + 18) / 3, 16.33333333, (21 + 6) / 2, (9 + 9) / 2, (1 + 1) / 2]
stability_rytov0025_final = [0, (42 + 41) / 2, (39 + 47) / 2,
                             35.33333333, (23 + 23 + 26) / 3, (22 + 24 + 24) / 3, (4 + 4) / 2]

# Print values for verification
print("Stabilities for Rytov variance 0.05 vs w:", stability_rytov005_final)
print("Stabilities for Rytov variance 0.025 vs w:", stability_rytov0025_final)
print("w values:", w_values)

# Function to calculate confidence intervals
def confidence_interval(p, n, confidence_level=0.95):
    z = norm.ppf((1 + confidence_level) / 2)
    p = np.array(p) / 100
    se = np.sqrt((p * (1 - p)) / n)
    ci = z * se
    return ci * 100

# Parameters
n_samples = 300
confidence_level = 0.95

# Calculate confidence intervals
ci_005 = confidence_interval(stability_rytov005_final, n_samples, confidence_level)
ci_0025 = confidence_interval(stability_rytov0025_final, n_samples, confidence_level)

# Plotting
plt.figure(figsize=(10, 6))

# Plot with wider lines and larger markers
plt.plot(w_values, stability_rytov005_final, marker='o', markersize=8, linewidth=2.5, color='royalblue', label='Rytov variance 0.05')
plt.plot(w_values, stability_rytov0025_final, marker='s', markersize=8, linewidth=2.5, color='seagreen', label='Rytov variance 0.025')

# Confidence interval shading
plt.fill_between(w_values,
                 np.array(stability_rytov005_final) - ci_005,
                 np.array(stability_rytov005_final) + ci_005,
                 color='royalblue', alpha=0.2)

plt.fill_between(w_values,
                 np.array(stability_rytov0025_final) - ci_0025,
                 np.array(stability_rytov0025_final) + ci_0025,
                 color='seagreen', alpha=0.2)

# Labels
plt.xlabel('Parameter $w$', fontsize=18)
plt.ylabel('Percentage of Knots Preserving Their Topology', fontsize=18)

# Ticks
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# Grid and legend
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=18)

plt.tight_layout()
plt.show()