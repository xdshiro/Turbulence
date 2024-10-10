import numpy as np

import matplotlib.pyplot as plt

import math
from scipy.special import assoc_laguerre

from functions.all_knots_functions import *
def Gaussian(x, y, z=0, w0 = 1, k0 = 1, x0=0, y0=0, z0=0):
	"""
	Classic Gaussian beam
	:param x: X coordinate
	:param y: Y coordinate
	:param z: Z coordinate
	:param w: beam waist
	:param k0: wave number
	:param x0: center of the beam in x
	:param y0: center of the beam in y
	:param z0: center of the beam in z
	:return: complex field
	"""

	x = x - x0
	y = y - y0
	z = z - z0
	zR = (k0 * w0 ** 2)/2
	wz = w0 * np.sqrt(1+(z/zR)**2)
	R = (z ** 2 + zR ** 2)/z
	phi = -np.arctan2(z, zR)

	E_gaussian = (w0/wz) * np.exp(-(x ** 2 + y ** 2) / (wz ** 2)) * np.exp(
		-1j * k0 * (x ** 2 + y ** 2)/(2*R))*np.exp(-1j*phi)

	return E_gaussian


def plot_field_both(E_gaussian, extend=None):
	fig, ax = plt.subplots(1, 2, figsize=(13, 6))
	im0 = ax[0].imshow(np.abs(E_gaussian).T, extent=extend, cmap='magma')
	ax[0].set_title('|Amplitude|')
	fig.colorbar(im0, ax=ax[0], fraction=0.04, pad=0.02)
	
	im1 = ax[1].imshow(np.angle(E_gaussian).T, extent=extend, cmap='hsv')
	ax[1].set_title('Phase')
	fig.colorbar(im1, ax=ax[1], fraction=0.04, pad=0.02)
	plt.tight_layout()
	plt.show()


def field_propagator(field, k0, dz, dx, dy):
	"""
	Propagate the input field over a distance dz using the angular spectrum method.

	Parameters:
	field (numpy.ndarray): The input field in the spatial domain (2D array).
	k0 (float): The wavenumber (2 * pi / wavelength).
	dz (float): The propagation distance.
	dx (float): The resolution in the x direction.
	dy (float): The resolution in the y direction.

	Returns:
	numpy.ndarray: The propagated field in the spatial domain.
	"""
	# Get the number of points along x and y axes
	nx, ny = field.shape
	wavelength = 2 * np.pi / k0
	
	# Create the spatial frequency grids for the angular spectrum method
	# kx = np.arange(-nx // 2, nx // 2) / (nx * dx)
	# ky = np.arange(-ny // 2, ny // 2) / (ny * dy)
	# these 2 lines below are equivalent to the lines above
	kx = np.fft.fftshift(np.fft.fftfreq(nx, dx))  # Frequency coordinates along x
	ky = np.fft.fftshift(np.fft.fftfreq(ny, dy))  # Frequency coordinates along y
	# print(kx[0], kx[1])
	Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
	
	# Compute the transfer function for free-space propagation (angular spectrum)
	H = np.exp(1j * np.pi * (Kx**2 + Ky**2) * wavelength * dz)
	
	# Compute the Fourier transform of the initial field
	field_ft = np.fft.fftshift(np.fft.fft2(field))
	
	# Propagate the field in the Fourier domain
	field_ft_prop = field_ft * H
	
	# Perform the inverse Fourier transform to get the propagated field in the spatial domain
	field_prop = np.fft.ifft2(np.fft.ifftshift(field_ft_prop))
	##############
	# Return the propagated field
	return field_prop


width0 = 5e-3  # Beam waist
wavelength = 532e-9  # Wavelength of the beam in meters (532 nm corresponds to green light)
k0 = 2 * np.pi / wavelength  # Wave number (related to the wavelength of the beam)

# Defining boundaries for the 3D space where the beam is calculated
# These limits define the range of x, y, and z coordinates in the 3D mesh.
x_lim_3D = (-20.0e-3, 20.0e-3)  # X boundaries (in meters, representing the transverse dimension of the beam)
y_lim_3D = (-20.0e-3, 20.0e-3)  # Y boundaries (in meters, same as x for a square domain)
z_lim_3D = (-140, 140)  # Z boundaries (in arbitrary units, typically represents propagation distance)

# Resolution or the number of points along each axis
res_x_3D = 321  # Number of points along the X-axis
res_y_3D = 321  # Number of points along the Y-axis
res_z_3D = 121  # Number of points along the Z-axis

# Creating a linearly spaced array of z-coordinates for the 3D mesh
z_3D = np.linspace(*z_lim_3D, res_z_3D)  # 1D array of z-values between -100 and 100, with 40 points

# Creating 1D arrays of x and y coordinates for the 3D mesh
x_3D = np.linspace(*x_lim_3D, res_x_3D)  # 1D array of x-values between -20e-3 and 20e-3, with 120 points
y_3D = np.linspace(*y_lim_3D, res_y_3D)  # 1D array of y-values between -20e-3 and 20e-3, with 120 points

# Creating a 3D mesh grid using the x, y, and z coordinate arrays.
# 'indexing' specifies how the indices are interpreted ('ij' corresponds to matrix-style indexing).
mesh_3D = np.meshgrid(x_3D, y_3D, z_3D, indexing='ij')

# Generate the field using the custom function LG_simple (which likely generates a Laguerre-Gaussian beam).
# This function takes the 3D mesh grid and the beam parameters.
# 'l' (azimuthal index), 'p' (radial index), 'width' (beam width), 'k0' (wave number),
# 'x0', 'y0', 'z0' are the center coordinates of the beam.
#field = LG_simple(*mesh_3D, l=l, p=p, width=width0, k0=k0, x0=0, y0=0, z0=0)
field = Gaussian(*mesh_3D, w0=width0, k0=k0, x0=0, y0=0, z0=0)

# Plotting the field at the middle plane along the z-axis.
# The middle plane is taken by selecting the slice at res_z_3D // 2 (which is the middle of the z-resolution).
# plot_field_both is a custom plotting function that plots the intensity and phase of the field.
plot_field_both(field[:, :, 0])

# Plotting the field in the very last Z plane
# This will show the intensity and phase of the field at the end of the propagation range.
plot_field_both(field[:, :, -1])

L_prop = z_lim_3D[1] - z_lim_3D[0]  # propagation length
dx = x_3D[1] - x_3D[0]  # Resolution in x
dy = y_3D[1] - y_3D[0]  # Resolution in y
pixel_scale = (x_lim_3D[-1] - x_lim_3D[0]) / (res_x_3D - 1)

field_after_propagation = field_propagator(
	field[:, :, 0],  # The field at the middle z-plane
	k0, dz=L_prop, dx=dx, dy=dy
)

# Plotting the field after propagation.
# This will show the intensity and phase of the beam after it has propagated a distance of length_along_z.
plot_field_both(field_after_propagation)

# Initialize an array to store the propagated field along the XZ plane
field_xz = np.zeros((res_x_3D, res_z_3D), dtype=complex)


# Propagate the field over the Z range
field_current = field[:, :, 0]  # Start from the middle plane

field_xz[:, 0] = field_current[:, res_y_3D // 2]
for i in range(res_z_3D - 1):
	# Propagate the current field a step forward in Z
	field_current = field_propagator(field_current, k0, dz=L_prop / (res_z_3D - 1), dx=dx, dy=dy)
	# Store the intensity at the center of the Y axis (for the XZ plane)
	field_xz[:, i + 1] = field_current[:, res_y_3D // 2]

# plotting zx cross-section of the theoretical field
plot_field_both(field[:, res_y_3D // 2, :])
# plotting zx cross-section of the simulated field
plot_field_both(field_xz)





length_along_z = z_lim_3D[1] - z_lim_3D[0]
# Number of phase screens (used in turbulence simulations, will be 1 for now)
screens_num = 1

# Multiplier that scales the distance between phase screens, not used in this example but kept for future use
multiplier = 1

# Beam parameters used for propagation: (x0, y0, beam width, wavelength).
beam_par = (0, 0, width0, wavelength)  # The first two parameters are x0 and y0, but are set to 0 for simplicity.

# Pixel scale is the physical size of each pixel in the transverse plane (x, y).
# It is calculated by dividing the x-axis range by the number of resolution points in the x-axis.
pixel_scale = (x_lim_3D[-1] - x_lim_3D[0]) / (res_x_3D - 1)

# Propagation parameters for a phase screen model.
# Since there's no turbulence, the large 1e100 numbers are placeholders for distance and turbulence parameters.
psh_par_0 = (
	1e100, res_x_3D, pixel_scale, 1e100, 1e100
)  # Resolution in x, pixel scale, and placeholders for other parameters

# Propagating the field from the middle z-plane over a distance of length_along_z.
# propagation_ps is a function that handles the propagation with any phase screens.
# The function will return the field after propagation.
field_after_propagation = propagation_ps(
	field[:, :, 0],  # The field at the middle z-plane
	beam_par, psh_par_0, L_prop=length_along_z, screens_num=1,
	multiplier=[1]  # Scaling factor for phase screens, set to 1 for simplicity
)

plot_field_both(field_after_propagation)


