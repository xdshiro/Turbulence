import matplotlib.pyplot as plt
import numpy as np

# Data
phase_screens_amount = np.array([1, 2, 3, 4, 5, 10])

SR_5mm_025 = [
    0.9120438935154593, 0.9124942274950268, 0.9115760699614358,
    0.9116483128214228, 0.9138565101774505, 0.9122835438967303
]
Scintillation_5mm_025 = [
    0.01265978431990078, 0.012855020325206246, 0.01250484084668746,
    0.012169152765099112, 0.011766120955761794, 0.011343704917041952
]
Scintillation_reversed_5mm_025 = [
    0.02201770648042678, 0.018097858657311727, 0.017173921749081833,
    0.016227399200144932, 0.01563171400839458, 0.015020382518927233
]

SR_5mm_075 = [
    0.7708178505516435, 0.7732716611066133, 0.7714463231380416,
    0.7703953047653963, 0.7719633947421266, 0.7739741780827618
]
Scintillation_5mm_075 = [
    0.04375691561286543, 0.040994561520499406, 0.03903582613650136,
    0.03852364574296718, 0.03646093375844006, 0.03562656162722555
]
Scintillation_reversed_5mm_075 = [
    0.07592948487995743, 0.05926989616908429, 0.05509881695824137,
    0.050916200164129455, 0.050259463523326886, 0.046544945809794624
]

SR_5mm_15 = [
    0.6117426393689431, 0.6183665650751051, 0.6137916935819625,
    0.6180608426055308, 0.6171194145261484, 0.6181364097702257
]
Scintillation_5mm_15 = [
    0.09693537731856128, 0.08820333291253113, 0.08458423906226464,
    0.08076524937018315, 0.07831132502445293, 0.07580043594444574
]
Scintillation_reversed_5mm_15 = [
    0.16961953439813637, 0.1290646691416204, 0.11502159158039293,
    0.10974039808767433, 0.1029567658550683, 0.10035388766122955
]

# Plotting
fig, axs = plt.subplots(3, 3, figsize=(15, 15))

# Line styles and colors
line_styles = ['-', '--', '-.']
colors = ['b', 'g', 'r']

# 0.25
axs[0, 0].plot(phase_screens_amount, SR_5mm_025, linestyle=line_styles[0], color=colors[0], marker='o', linewidth=2, label='SR_5mm_025')
axs[0, 0].set_ylim(0.8, 1)
axs[0, 0].set_title('SR_5mm_025')
axs[0, 0].set_xlabel('Phase Screens Amount')
axs[0, 0].set_ylabel('SR Value')

axs[0, 1].plot(phase_screens_amount, Scintillation_5mm_025, linestyle=line_styles[0], color=colors[1], marker='o', linewidth=2, label='Scintillation_5mm_025')
axs[0, 1].set_ylim(0.01, 0.025)
axs[0, 1].set_title('Scintillation_5mm_025')
axs[0, 1].set_xlabel('Phase Screens Amount')
axs[0, 1].set_ylabel('Scintillation Value')

axs[0, 2].plot(phase_screens_amount, Scintillation_reversed_5mm_025, linestyle=line_styles[0], color=colors[2], marker='o', linewidth=2, label='Scintillation_Reversed_5mm_025')
axs[0, 2].set_ylim(0.01, 0.025)
axs[0, 2].set_title('Scintillation_Reversed_5mm_025')
axs[0, 2].set_xlabel('Phase Screens Amount')
axs[0, 2].set_ylabel('Scintillation Reversed Value')

# 0.75
axs[1, 0].plot(phase_screens_amount, SR_5mm_075, linestyle=line_styles[1], color=colors[0], marker='o', linewidth=2, label='SR_5mm_075')
axs[1, 0].set_ylim(0.65, 0.85)
axs[1, 0].set_title('SR_5mm_075')
axs[1, 0].set_xlabel('Phase Screens Amount')
axs[1, 0].set_ylabel('SR Value')

axs[1, 1].plot(phase_screens_amount, Scintillation_5mm_075, linestyle=line_styles[1], color=colors[1], marker='o', linewidth=2, label='Scintillation_5mm_075')
axs[1, 1].set_ylim(0.03, 0.08)
axs[1, 1].set_title('Scintillation_5mm_075')
axs[1, 1].set_xlabel('Phase Screens Amount')
axs[1, 1].set_ylabel('Scintillation Value')

axs[1, 2].plot(phase_screens_amount, Scintillation_reversed_5mm_075, linestyle=line_styles[1], color=colors[2], marker='o', linewidth=2, label='Scintillation_Reversed_5mm_075')
axs[1, 2].set_ylim(0.03, 0.08)
axs[1, 2].set_title('Scintillation_Reversed_5mm_075')
axs[1, 2].set_xlabel('Phase Screens Amount')
axs[1, 2].set_ylabel('Scintillation Reversed Value')

# 1.5
axs[2, 0].plot(phase_screens_amount, SR_5mm_15, linestyle=line_styles[2], color=colors[0], marker='o', linewidth=2, label='SR_5mm_15')
axs[2, 0].set_ylim(0.5, 0.7)
axs[2, 0].set_title('SR_5mm_15')
axs[2, 0].set_xlabel('Phase Screens Amount')
axs[2, 0].set_ylabel('SR Value')

axs[2, 1].plot(phase_screens_amount, Scintillation_5mm_15, linestyle=line_styles[2], color=colors[1], marker='o', linewidth=2, label='Scintillation_5mm_15')
axs[2, 1].set_ylim(0.07, 0.18)
axs[2, 1].set_title('Scintillation_5mm_15')
axs[2, 1].set_xlabel('Phase Screens Amount')
axs[2, 1].set_ylabel('Scintillation Value')

axs[2, 2].plot(phase_screens_amount, Scintillation_reversed_5mm_15, linestyle=line_styles[2], color=colors[2], marker='o', linewidth=2, label='Scintillation_Reversed_5mm_15')
axs[2, 2].set_ylim(0.07, 0.18)
axs[2, 2].set_title('Scintillation_Reversed_5mm_15')
axs[2, 2].set_xlabel('Phase Screens Amount')
axs[2, 2].set_ylabel('Scintillation Reversed Value')

plt.tight_layout()
plt.show()