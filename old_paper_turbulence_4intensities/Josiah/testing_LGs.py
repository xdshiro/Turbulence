import numpy as np
from functions.all_knots_functions import *  # Importing the custom knot functions library

# Parameters
width0 = 5e-3  # Beam width (related to the beam waist or size of the beam)
l = 1  # Azimuthal index (the orbital angular momentum of the beam, 'l' defines the number of twists in the wavefront)
p = 0  # Radial index (related to the number of rings in the intensity distribution of the beam)
wavelength = 532e-9  # Wavelength of the beam in meters (532 nm corresponds to green light)
k0 = 2 * np.pi / wavelength  # Wave number (related to the wavelength of the beam)

# Defining boundaries for the 3D space where the beam is calculated
# These limits define the range of x, y, and z coordinates in the 3D mesh.
x_lim_3D = (-20.0e-3, 20.0e-3)  # X boundaries (in meters, representing the transverse dimension of the beam)
y_lim_3D = (-20.0e-3, 20.0e-3)  # Y boundaries (in meters, same as x for a square domain)
z_lim_3D = (-100, 100)  # Z boundaries (in arbitrary units, typically represents propagation distance)

# Resolution or the number of points along each axis
res_x_3D = 120  # Number of points along the X-axis
res_y_3D = 120  # Number of points along the Y-axis
res_z_3D = 40  # Number of points along the Z-axis

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
field = LG_simple(*mesh_3D, l=l, p=p, width=width0, k0=k0, x0=0, y0=0, z0=0)

# Plotting the field at the middle plane along the z-axis.
# The middle plane is taken by selecting the slice at res_z_3D // 2 (which is the middle of the z-resolution).
# plot_field_both is a custom plotting function that plots the intensity and phase of the field.
plot_field_both(field[:, :, res_z_3D // 2])

# Plotting the field in the very last Z plane
# This will show the intensity and phase of the field at the end of the propagation range.
plot_field_both(field[:, :, -1])

# Propagating the field along the z-axis starting from the middle plane.

length_along_z = 100  # Propagation distance along the z-axis, typically representing the total distance traveled by the beam

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
	field[:, :, res_z_3D // 2],  # The field at the middle z-plane
	beam_par, psh_par_0, L_prop=length_along_z, screens_num=1,
	multiplier=[1]  # Scaling factor for phase screens, set to 1 for simplicity
)

# Plotting the field after propagation.
# This will show the intensity and phase of the beam after it has propagated a distance of length_along_z.
plot_field_both(field_after_propagation)