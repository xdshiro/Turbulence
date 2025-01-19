from functions.all_knots_functions import *
import itertools
import scipy.io as sio
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm
# Create a new colormap based on 'Blues' with compressed values near zero
# colors = plt.cm.Blues(np.linspace(0, 1, 256))
# colors[:, 0:3] = colors[:, 0:3] ** 1.3  # Adjust the power as needed for compression
# custom_blues = LinearSegmentedColormap.from_list("CustomBlues", colors)
# Define the original colormap and the fraction of white at the beginning
original_cmap = plt.cm.YlGnBu
white_fraction = 0.01  # 1% for white

# Extract the colors from the original colormap and add white at the beginning
colors = original_cmap(np.linspace(0, 1, 256))
colors[:int(256 * white_fraction)] = [1, 1, 1, 1]  # Set the first 1% to white

# Create the custom colormap
custom_cmap = LinearSegmentedColormap.from_list("CustomYlGnBu", colors)
custom_blues = custom_cmap
foils_all = list(itertools.product(range(3), repeat=4))

# foils = [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2)]
# foils = [(0, 0, 1, 1), (0, 0, 1, 2), (0, 0, 2, 2)]
foils = [(0, 1, 1, 1), (0, 1, 1, 2), (0, 1, 2, 2)]
foils = [(2, 0, 2, 2), (1, 1, 1, 1), (0, 1, 0, 2)]
import itertools

foils = list(itertools.product(range(3), repeat=4))
plot = False
# foils = foils_all   # to save all the combinations. Just turn of plots because there will be too many
for foil in foils:
	print(f'4foil name is {foil}')
	knot = ''.join([str(element) for element in foil])
	
	
	# # meshes and boundaries for getting a knot
	x_lim_3D_knot, y_lim_3D_knot, z_lim_3D_knot = (-7.0, 7.0), (-7.0, 7.0), (-2.0, 2.0)
	res_x_3D_knot, res_y_3D_knot, res_z_3D_knot = 256, 256, 1
	#
	# # beam
	lmbda = 532e-9  # wavelength
	L_prop = 270  # propagation distance
	knot_length = 212.58897655870774 / 2 * 1.4
	center_plane = 1
	
	width0 = 6.0e-3 / np.sqrt(2)  # beam width
	xy_lim_2D_origin = (-30.0e-3, 30.0e-3)  # window size to start with
	scale = 1
	res_xy_2D_origin = int(scale * 300) # resolution
	#
	res_z = int(scale * 64)  # resolution of the knot is res_z+1
	crop = int(scale * 185)  # for the knot propagation
	crop_3d = int(scale * 100)  # for the knot
	
	k0 = 2 * np.pi / lmbda  # wave number
	
	# # propagation parameters
	z0 = knot_length * (1 - center_plane) + L_prop  # the source position
	prop1 = L_prop  # z0-prop1 - detector position
	prop2 = knot_length * (1 - center_plane)  # z0-prop1-pro2 - knot center (assumed)
	
	beam_par = (0, 0, width0, lmbda)
	
	# # extra values (simulations)
	x_3D_knot, y_3D_knot = np.linspace(*x_lim_3D_knot, res_x_3D_knot), np.linspace(*y_lim_3D_knot, res_y_3D_knot)
	if res_z_3D_knot != 1:
		z_3D_knot = np.linspace(*z_lim_3D_knot, res_z_3D_knot)
	else:
		z_3D_knot = 0
	mesh_3D_knot = np.meshgrid(x_3D_knot, y_3D_knot, z_3D_knot, indexing='ij')
	x_2D_origin = np.linspace(*xy_lim_2D_origin, res_xy_2D_origin)
	y_2D_origin = np.linspace(*xy_lim_2D_origin, res_xy_2D_origin)
	mesh_2D_original = np.meshgrid(x_2D_origin, y_2D_origin, indexing='ij')
	
	pxl_scale = (xy_lim_2D_origin[1] - xy_lim_2D_origin[0]) / (res_xy_2D_origin - 1)
	
	values = unknot_4_any(mesh_3D_knot, braid_func=braid, plot=False,
	                      angle_size=foil, cmap=custom_blues)
	print_coeff = True
	# printing values
	if print_coeff:
		values_print = np.real(values['weight']) / np.sqrt(np.sum(np.real(values['weight']) ** 2)) * 10
		values_print_formatted = [f'{val:.2f}' for val in values_print]
		print(f'vales l: {values["l"]}, p: {values["p"]}, weights: {values_print_formatted}')
	# sio.savemat(f'values_{foil}.mat', values)
	# building the knot from the coefficients
	field_before_prop = field_knot_from_weights(
		values, mesh_2D_original, width0, k0=k0, x0=0, y0=0, z0=z0
	)
	if plot:
		plot_field_both(field_before_prop, phase_phi=True)
	
	psh_par_0 = (1 * 1e100, res_xy_2D_origin, pxl_scale, 1 * 1e100, 1 * 1e100)
	field_after_prop = propagation_ps(
	                field_before_prop, beam_par, psh_par_0, prop1, multiplier=[1], screens_num=1, seed=0
	            )
	
	if plot:
		plot_field_both(field_after_prop, phase_phi=True)
	
	field_z_crop = field_after_prop[
	                           res_xy_2D_origin // 2 - crop // 2: res_xy_2D_origin // 2 + crop // 2,
	                           res_xy_2D_origin // 2 - crop // 2: res_xy_2D_origin // 2 + crop // 2,
	                           ]
	if plot:
		plot_field_both(field_z_crop, phase_phi=True)