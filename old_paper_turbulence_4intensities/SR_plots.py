import matplotlib.pyplot as plt

L = [500, 200]  # propagation length
Cn2 = [[1e-13, 5e-14, 1e-14, 1e-15], [5e-13, 1e-13, 5e-14, 1e-14]]  # turbulence strength
s = [[5, 3.5, 1.4, 0.4], [4.9, 2, 1.4, 0.6]]  # screens required
s2 = [[4, 1, 1, 1], [3, 1, 1, 1]]  # screens required Forbes
SR = [[0.157, 0.3, 0.7, 0.96], [0.087, 0.38, 0.55, 0.87]]  # propagation length
Rytov = [[1.95, 0.975, 0.194, 0.0194], [1.81, 0.36, 0.181, 0.036]]  # Rytov variance


# Plotting
# Plotting
# Plotting

# Font size settings
title_fontsize = 22
label_fontsize = 18
legend_fontsize = 16
fig, axs = plt.subplots(2, 2, figsize=(14, 14))

# Plot 1: Cn2 vs SR
for i, l_value in enumerate(L):
    axs[0, 0].plot(SR[i], Cn2[i], marker='o', label=f'Cn2 L = {l_value}')
axs[0, 0].set_title('Turbulence Strength (Cn2)', fontsize=title_fontsize)
axs[0, 0].set_xlabel('Strehl Ratio (SR)', fontsize=label_fontsize)
axs[0, 0].set_ylabel('Cn2', fontsize=label_fontsize)
axs[0, 0].legend(fontsize=legend_fontsize)

# Plot 2: Screens Required vs SR
for i, l_value in enumerate(L):
    axs[0, 1].plot(SR[i], s[i], marker='o', label=f'Screens Required L = {l_value}')
axs[0, 1].set_title('Screens Required (s)', fontsize=title_fontsize)
axs[0, 1].set_xlabel('Strehl Ratio (SR)', fontsize=label_fontsize)
axs[0, 1].set_ylabel('Screens Required', fontsize=label_fontsize)
axs[0, 1].legend(fontsize=legend_fontsize)

# Plot 3: Screens Required Forbes vs SR
for i, l_value in enumerate(L):
    axs[1, 0].plot(SR[i], s2[i], marker='o', label=f'Screens Required Forbes L = {l_value}')
axs[1, 0].set_title('Screens Required Forbes (s2)', fontsize=title_fontsize)
axs[1, 0].set_xlabel('Strehl Ratio (SR)', fontsize=label_fontsize)
axs[1, 0].set_ylabel('Screens Required Forbes', fontsize=label_fontsize)
axs[1, 0].legend(fontsize=legend_fontsize)

# Plot 4: Rytov Variance vs SR
for i, l_value in enumerate(L):
    axs[1, 1].plot(SR[i], Rytov[i], marker='o', label=f'Rytov Variance L = {l_value}')
axs[1, 1].set_title('Rytov Variance', fontsize=title_fontsize)
axs[1, 1].set_xlabel('Strehl Ratio (SR)', fontsize=label_fontsize)
axs[1, 1].set_ylabel('Rytov Variance', fontsize=label_fontsize)
axs[1, 1].legend(fontsize=legend_fontsize)

plt.tight_layout()
plt.show()