import matplotlib.pyplot as plt
import numpy as np

# Data
phase_screens_amount = np.array([1, 2, 3, 4, 5, 10])

SR_50mm_025 = [
    0.0417468325846051,
    0.033453670315102935,
    0.029868671042974748,
    0.027572917761715443,
    0.027131000257867743,
    0.027480150118669

]
SR_150mm_025 = [
    0.04208969462578005,
    0.030089742102403783,
    0.03100773267462409,
    0.02754827029761353,
    0.02557319874055919,
    0.024460300981074212

]
SR_50mm_075 = [
    0.1273105039097715,
    0.10647713433932493,
    0.09211038040511776,
    0.09162299392745066,
    0.08354686358329766,
    0.07713684801744303

]
SR_150mm_075 = [
    0.1254484973659571,
    0.11467143652958423,
    0.08508494401798217,
    0.07973986230977492,
    0.09267010887611038,
    0.07109404772001926

]

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Line styles and colors
line_styles = ['-', '--']
colors = ['b', 'g']

# 0.25
axs[0, 0].plot(phase_screens_amount, SR_50mm_025, linestyle=line_styles[0], color=colors[0], marker='o', linewidth=2, label='SR_5mm_025')
# axs[0, 0].set_ylim(0.8, 1)
axs[0, 0].set_title('SR_50mm_025')
axs[0, 0].set_xlabel('Phase Screens Amount')
axs[0, 0].set_ylabel('Scintillation Value')

axs[0, 1].plot(phase_screens_amount, SR_150mm_025, linestyle=line_styles[0], color=colors[1], marker='o', linewidth=2, label='Scintillation_5mm_025')
# axs[0, 1].set_ylim(0.01, 0.025)
axs[0, 1].set_title('Scintillation_150mm_025')
axs[0, 1].set_xlabel('Phase Screens Amount')
axs[0, 1].set_ylabel('Scintillation Value')


# 0.75
axs[1, 0].plot(phase_screens_amount, SR_50mm_075, linestyle=line_styles[1], color=colors[0], marker='o', linewidth=2, label='SR_5mm_075')
# axs[1, 0].set_ylim(0.65, 0.85)
axs[1, 0].set_title('SR_50mm_075')
axs[1, 0].set_xlabel('Phase Screens Amount')
axs[1, 0].set_ylabel('Scintillation Value')

axs[1, 1].plot(phase_screens_amount, SR_150mm_075, linestyle=line_styles[1], color=colors[1], marker='o', linewidth=2, label='Scintillation_5mm_075')
# axs[1, 1].set_ylim(0.03, 0.08)
axs[1, 1].set_title('Scintillation_150mm_075')
axs[1, 1].set_xlabel('Phase Screens Amount')
axs[1, 1].set_ylabel('Scintillation Value')


plt.tight_layout()
plt.show()