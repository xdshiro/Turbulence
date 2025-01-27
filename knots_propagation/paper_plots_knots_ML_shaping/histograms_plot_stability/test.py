import numpy as np
import matplotlib.pyplot as plt

# ----- Global font settings -----
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# Sample data
knots = ['H1','H2','H3','H4','H5','H6','H7','H8','H9','T1','T2']
stabilities_005 = np.array([45, 60, 75, 35, 50, 66, 80, 62, 55, 16, 67])
err_005 = np.array([5, 3, 4, 2, 1, 7, 5, 3, 4, 1, 5])

stabilities_015 = np.array([25, 40, 49, 60, 72, 58, 33, 45, 56,  2, 21])
err_015 = np.array([3, 2, 4, 6, 3, 2, 5, 7, 3, 1, 2])

stabilities_025 = np.array([10, 20, 30, 18, 19, 23, 27, 30,  5,  1,  9])
err_025 = np.array([2, 2, 3, 1, 3, 4, 2, 2, 1, 1, 3])

N = len(knots)
ind = np.arange(N)
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5))

colors = ['#deebf7', '#9ecae1', '#3182bd']

# --- Bar plots with error bars ---
bar1 = ax.bar(ind - width, stabilities_005, width,
              yerr=err_005, capsize=5, color=colors[0],
              edgecolor='black', label="σ²=0.05")

bar2 = ax.bar(ind, stabilities_015, width,
              yerr=err_015, capsize=5, color=colors[1],
              edgecolor='black', label="σ²=0.15")

bar3 = ax.bar(ind + width, stabilities_025, width,
              yerr=err_025, capsize=5, color=colors[2],
              edgecolor='black', label="σ²=0.25")


def add_value_labels(rects, errors, extra_offset=1.0):
    """
    Place a text label on top of each bar at (height + error + extra_offset).
    """
    for rect, err in zip(rects, errors):
        height = rect.get_height()
        x_pos = rect.get_x() + rect.get_width() / 2
        label_pos = height + err + extra_offset  # bar top + error + offset
        ax.text(x_pos, label_pos, f"{height:.0f}",
                ha='center', va='bottom', fontweight='bold')


# Add numeric labels above each bar, offset by the bar's error + 1
add_value_labels(bar1, err_005, extra_offset=.5)
add_value_labels(bar2, err_015, extra_offset=.5)
add_value_labels(bar3, err_025, extra_offset=.5)

# Axis labels, ticks, legend
ax.set_xlabel('Knot Type')
ax.set_ylabel('Recovered Knots (%)')
ax.set_xticks(ind)
ax.set_xticklabels(knots)
ax.legend()

ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()