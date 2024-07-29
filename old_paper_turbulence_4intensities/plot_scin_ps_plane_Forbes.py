import matplotlib.pyplot as plt
import numpy as np

# Data
phase_screens_amount = np.array([1, 2, 3, 4, 5, 10])

Rytov_0075_5mm_scin = [
    0.16961953439813637, 0.1490646691416204, 0.12502159158039293,
    0.11534039808767433, 0.11229567658550683, 0.11035388766122955

]

# Plotting
fig, axs = plt.subplots(1, 1, figsize=(5, 5))

# Line styles and colors
line_styles = ['-', '--']
colors = ['b', 'g']

# 0.25
axs.plot(phase_screens_amount, Rytov_0075_5mm_scin, linestyle=line_styles[0], color=colors[0], marker='o', linewidth=4, label='SR_5mm_025')
axs.set_ylim(0.1, 0.18)
axs.set_title('beam w=3mm. Rytov=0.15. L=100m')
axs.set_xlabel('# Phase Screens')
axs.set_ylabel('Scintillation Value')





plt.tight_layout()
plt.show()