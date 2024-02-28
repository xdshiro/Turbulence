import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('x300_y375_imagEy.csv', skiprows=7)
x = df.iloc[:, 0].values
y = df.iloc[:, 1].values
values = df.iloc[:, 3].values
print(x)
# unique_x = np.unique(x)
# unique_y = np.unique(y)
X, Y = np.meshgrid(x, y)
# df_sorted = df.sort_values(by=[df.columns[1], df.columns[0]])
# sorted_values = df_sorted.iloc[:, 3].values
print(len(y), np.sqrt(len(y)))
Z = values.reshape(np.sqrt(len(y)), np.sqrt(len(y)))
plt.imshow(Z)
plt.show()