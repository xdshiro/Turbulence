import matplotlib.pyplot as plt
import numpy as np

Rytov = [0.025, 0.05, 0.1, 0.15, 0.2]
cases = [[95 + 2/3, 3 + 2/3, 2/3],
         [67 + 1/3, 22, 10 + 2/3],
         [37, 27.5, 35.5],
         [21 + 2/3, 22 + 1/3, 56],
         [7 + 2/3, 23 + 1/3, 69]]

# Convert cases to a numpy array for easy manipulation
cases_array = np.array(cases)

# Plotting the histogram with three columns for each Rytov value
width = 0.01  # Width of each bar
Rytov = np.array(Rytov)

# Create an offset for each column
offsets = [-width, 0, width]

# Plot each column for the cases
for i in range(cases_array.shape[1]):
    plt.bar(Rytov + offsets[i], cases_array[:, i], width=width, label=f'Case {i+1}')

# Add labels and title
plt.xlabel('Rytov Values')
plt.ylabel('Values')
plt.title('Cases for each Rytov Value')
plt.legend()

# Display the plot
plt.show()

# Printing values of cases
for i, case in enumerate(cases):
    print(f"Rytov {Rytov[i]}: {case}")
for case in cases:
	print(sum(case))
# print((92+1/3)*(67/(24+67)))
# # 24/31
# 22/55
# 24/67