import matplotlib.pyplot as plt
import numpy as np

# Data
phase_screens_amount = np.array([1, 2, 3, 4, 5, 10])

SR_015_5mm_scin = [
    0.6117426393689431, 0.6183665650751051, 0.6137916935819625,
    0.6180608426055308, 0.6171194145261484, 0.6181364097702257

]

# Plotting
fig, axs = plt.subplots(1, 1, figsize=(5, 5))

# Line styles and colors
line_styles = ['-', '--']
colors = ['b', 'g']

# 0.25
axs.plot(phase_screens_amount, SR_015_5mm_scin, linestyle=line_styles[0], color=colors[1], marker='o', linewidth=4, label='SR_5mm_025')
# axs.set_ylim(0.1, 0.18)
axs.set_ylim(0.0, 1)
axs.set_title('Plane wave. Rytov=0.15. L=100m')
axs.set_xlabel('# Phase Screens')
axs.set_ylabel('SR')





plt.tight_layout()
plt.show()