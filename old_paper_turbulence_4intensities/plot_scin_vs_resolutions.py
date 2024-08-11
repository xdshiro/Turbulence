import matplotlib.pyplot as plt

# Data for each resolution
data = [
    [0.2671356108870644, 0.18596756850018825, 0.1702352306778874, 0.15466790825706847],
    [0.2604182217489988, 0.19403056125077475, 0.1676797552796741, 0.14756371599098216],
    [0.2593291994527771, 0.17597381747632123, 0.16218701742490065, 0.14801091678420075],
    [0.22751738044544711, 0.1876695925977077, 0.17372221876310132, 0.16371765854717174],
    [0.24337109400359921, 0.17397365578212082, 0.1608527676473767, 0.1605244062930797],
    [0.2740699567128081, 0.18496788886154142, 0.1560146100738776, 0.1614917404839303],
    [0.24361066804263687, 0.19324758395996858, 0.17998018670759128, 0.1537620797302066]
]

# Number of phase screens
phase_screens = [1, 3, 5, 10]

# Resolutions for different data sets
resolutions = [101, 301, 501, 701, 901, 1101, 1301]

# Plot
plt.figure(figsize=(12, 8))

# Colors for different resolution lines
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

for i, values in enumerate(data):
    plt.plot(phase_screens, values, marker='o', linestyle='-', label=f'Resolution {resolutions[i]}',
             color=colors[i % len(colors)], lw=3)

# Labels and title
plt.xlabel('Number of Phase Screens')
plt.ylabel('Scintillation')
plt.title('Scintillation vs. Number of Phase Screens')

# Display legend
plt.legend()

# Display the plot
plt.grid(True)
plt.show()