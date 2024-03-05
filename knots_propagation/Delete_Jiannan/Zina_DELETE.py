import numpy as np
import matplotlib.pyplot as plt
r_min = 2.82685032
E = 98085.134325306
sigma = 2.51843733
r = np.linspace(2.5, 8.02, 100)
V = 4 * E * ((sigma / r) ** 12 - (sigma / r) ** 6)
plt.figure(figsize=(10, 6))
plt.plot(r, V, label='Lennard-Jones Potential')
plt.title('Potential curve')
plt.xlabel('Distance, r (Å)')
plt.ylabel('Potential Energy, V (eV)')
# plt.ylim(-98085.0, -98084.9)
plt.grid(True)
plt.legend()
plt.gca().ticklabel_format(style='plain', axis='y', useOffset=False)
plt.show()