import numpy as np
import matplotlib.pyplot as plt

# Constants
c = 3e8  # Speed of light (m/s)
dx = 1e-6  # Spatial step (m)
dt = dx / (2 * c)  # Time step (Courant condition)
size = 200  # Size of the simulation grid
steps = 500  # Number of time steps

# Field arrays
Ez = np.zeros(size)  # Electric field
Hy = np.zeros(size)  # Magnetic field

# Pulse parameters
pulse_pos = 50  # Initial position of the pulse
pulse_width = 20  # Width of the pulse

# Initialize electric field with a Gaussian pulse
Ez[pulse_pos] = np.exp(-((np.arange(size) - pulse_pos) ** 2) / (2 * pulse_width ** 2))

# FDTD loop
for n in range(steps):
	# Update magnetic field (Hy)
	Hy[:-1] += (Ez[1:] - Ez[:-1]) * dt / (dx * 4 * np.pi * 1e-7)
	
	# Update electric field (Ez)
	Ez[1:] += (Hy[1:] - Hy[:-1]) * dt / (dx * 8.85e-12)
	
	# Plot the electric field at selected time steps
	if n % 50 == 0:
		plt.plot(Ez, label=f'Time step: {n}')
		plt.xlabel('Position')
		plt.ylabel('Electric Field (Ez)')
		plt.title('1D FDTD Simulation of Pulse Propagation')

plt.legend()
plt.show()