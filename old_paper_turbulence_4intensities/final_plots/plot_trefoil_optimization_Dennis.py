import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Set global font to Times New Roman
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 18
})

def confidence_interval(p, n, confidence_level=0.95):
    z = norm.ppf((1 + confidence_level) / 2)
    p = np.array(p) / 100
    se = np.sqrt((p * (1 - p)) / n)
    ci = z * se
    return ci * 100

# Data preparation
stability_optimized_center_final_3int = np.array([100, 92, 41, 21])
stability_optimized_center_delta_3int = np.array([0, 0, 0, 0])
stability_optimized_center_final_3int = (stability_optimized_center_final_3int + stability_optimized_center_delta_3int // 2) / 1

stability_optimized_center = np.array([92 + 97 + 93, 61 + 66 + 62, 17 + 18 + 22, 8 + 4 + 5])
stability_optimized_center_delta = np.array([6 + 1 + 3, 10 + 8 + 8, 4 + 7 + 5, 5 + 4 + 3])
stability_optimized_center_final = (stability_optimized_center + stability_optimized_center_delta // 2) / 3

Rytov_values = [0.025, 0.05, 0.15, 0.2]
n_samples = 300
n_samples_short = 150
confidence_level = 0.95

ci_3int = confidence_interval(stability_optimized_center_final_3int, n_samples_short, confidence_level)
ci_standard = confidence_interval(stability_optimized_center_final, n_samples, confidence_level)

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(Rytov_values, stability_optimized_center_final_3int, marker='o', markersize=8, linewidth=2.5,
         color='royalblue', label='7-modes optimization')
plt.plot(Rytov_values, stability_optimized_center_final, marker='s', markersize=8, linewidth=2.5,
         color='seagreen', label='5-modes optimization')

plt.fill_between(Rytov_values,
                 np.array(stability_optimized_center_final_3int) - ci_3int,
                 np.array(stability_optimized_center_final_3int) + ci_3int,
                 color='royalblue', alpha=0.2)

plt.fill_between(Rytov_values,
                 np.array(stability_optimized_center_final) - ci_standard,
                 np.array(stability_optimized_center_final) + ci_standard,
                 color='seagreen', alpha=0.2)

plt.xlabel('Rytov variance', fontsize=18)
plt.ylabel('Percentage of Knots Preserving Their Topology', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=18)
plt.ylim(0, 100)
plt.tight_layout()
plt.show()