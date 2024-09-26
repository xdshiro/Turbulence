import matplotlib.pyplot as plt

from functions.all_knots_functions import *
import os
import pickle
import csv
import json
from tqdm import trange


import scipy.io
from scipy.ndimage import center_of_mass, shift, zoom
import numpy as np

# Load the .mat file
mat_data = scipy.io.loadmat('trefoil_noturb_fullslm_test3.mat')
print(mat_data.keys())

# Extract the complex field from the loaded data
complex_field = mat_data['U'].T

# Plot the original field
plot_field_both(complex_field)

# Scaling coefficient for new resolution
scaling_factor = np.sqrt(270 / 0.64)
print(f'coefficient of the size scaling: {scaling_factor}')

# 1. Calculate the intensity of the complex field
intensity = np.abs(complex_field)**2

# 2. Find the center of intensity
center = center_of_mass(intensity)

# Get the shape of the complex field
shape = complex_field.shape

# Calculate the target center (which is the center of the array)
target_center = (shape[0] // 2, shape[1] // 2)

# 3. Calculate the shift required to move the center of intensity to the target center
shift_amount = np.array(target_center) - np.array(center)

# 4. Shift the complex field so the center of intensity is in the middle
shifted_field = shift(complex_field, shift_amount, mode='constant', cval=0)
plot_field_both(shifted_field)
pad = 100
shifted_field = np.pad(shifted_field, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)

# Separate the real and imaginary parts
real_part = np.real(shifted_field)
imag_part = np.imag(shifted_field)

# Desired size X x X
X = 300

# Calculate the zoom factor based on the new desired size
zoom_factor = (X / real_part.shape[0], X / real_part.shape[1])

# Step 1: Create a mask where values are zero in the original arrays
zero_mask = (real_part == 0) & (imag_part == 0)

# Step 2: Resize both the real and imaginary parts using zoom
real_resized = zoom(real_part, zoom_factor, mode='reflect')
imag_resized = zoom(imag_part, zoom_factor, mode='reflect')

# Step 3: Resize the zero mask to match the zoomed size, using nearest interpolation to maintain zeros
zero_mask_resized = zoom(zero_mask.astype(float), zoom_factor, mode='constant', cval=0, order=0).astype(bool)

# Step 4: Combine the resized real and imaginary parts into a complex field
complex_resized = real_resized + 1j * imag_resized

# Step 5: Apply the mask to ensure zero values are preserved
complex_resized[zero_mask_resized] = 0

# plot_field_both(complex_resized)

# exit()
SAMPLES = 1
indx_plus = 0

plot = 0
plot_3d = 1
print_coeff = 0

print_values = 0
centering = 0
seed = None  # does work with more than 1 phase screen
no_last_plane = True


# folder = 'data_basis_delete'
# folder = 'data_no_centers_32114'
# folder = 'data_low_10'

spectrum_save = 1
no_turb = 0

# meshes and boundaries for getting a knot
x_lim_3D_knot, y_lim_3D_knot, z_lim_3D_knot = (-7.0, 7.0), (-7.0, 7.0), (-2.0, 2.0)
res_x_3D_knot, res_y_3D_knot, res_z_3D_knot = 256, 256, 1

# beam
lmbda = 532e-9  # wavelength
L_prop = 270 * 1  # propagation distance
knot_length = 212.58897655870774 / 2 * 1.8  # we need RALEYIG!!!!!!!!  # 1000 how far is detector from the knot center
center_plane = 0.3
center_plane = 1
########################################################
width0 = 6.0e-3 / np.sqrt(2)  # beam width
xy_lim_2D_origin = (-45.0e-3 * (X + pad)/X, 45.0e-3 * (X + pad)/X)  # window size to start with
scale = 1.0
res_xy_2D_origin = int(scale * 300) # resolution

res_z = int(scale * 80)  # resolution of the knot is res_z+1
crop = int(scale * 185)  # for the knot propagation
crop_3d = int(scale * 100)  # for the knot
new_resolution = (int(scale * 100), int(scale * 100))  # resolution of the knot to save

screens_num1 = 3
multiplier1 = [1] * screens_num1
screens_num2 = 1
multiplier2 = [1] * screens_num2

# turbulence
# Cn2 = 1.35e-13  # turbulence strength  is basically in the range of 10−17–10−12 m−2/3
# Cn2 = 3.21e-14
# Cn2s = [5e-15, 1e-14, 5e-15, 1e-13]
# Cn2 = Cn2s[0]
Rytovs = [0.03, 0.052, 0.091]  # 135
Rytovs = [0.086, 0.161, 0.28]  # 540
Rytovs = [0.05]#, 0.2]
Rytovs = [1e-30]#, 0.2]
# Rytovs = [0.025, 0.05]
for Rytov in Rytovs:
    # folder = f'standard_vs_WWW_trefoil_vs_rytov_{Rytov}_100_1.4zR_c03_v1'
    folder = f'experiment_{Rytov}_100_center_plane_v1'
    # folder = f'optimized_trefoil_vs_rytov_{Rytov}_100_center_plane_v2'
    k0 = 2 * np.pi / lmbda  # wave number
    Cn2 = Cn2_from_Rytov(Rytov, k0, L_prop)
    # # # # # Cn2 = 3.21e-15
    # Cn2 = 3.21e-40
    # https://www.mdpi.com/2076-3417/11/22/10548
    L0 = 9 * 1e10 # outer scale
    l0 = 5e-3 *1e-10 # inner scale
    
    # propagation parameters
    z0 = knot_length * (1 - center_plane) + L_prop  # the source position
    prop1 = L_prop  # z0-prop1 - detector position
    prop2 = knot_length * (1 - center_plane)  # z0-prop1-pro2 - knot center (assumed)
    
    # extra values (physical)
    
    beam_par = (0, 0, width0, lmbda)
    
    # extra values (simulations)
    x_3D_knot, y_3D_knot = np.linspace(*x_lim_3D_knot, res_x_3D_knot), np.linspace(*y_lim_3D_knot, res_y_3D_knot)
    if res_z_3D_knot != 1:
        z_3D_knot = np.linspace(*z_lim_3D_knot, res_z_3D_knot)
    else:
        z_3D_knot = 0
    mesh_3D_knot = np.meshgrid(x_3D_knot, y_3D_knot, z_3D_knot, indexing='ij')
    x_2D_origin = np.linspace(*xy_lim_2D_origin, res_xy_2D_origin)
    y_2D_origin = np.linspace(*xy_lim_2D_origin, res_xy_2D_origin)
    mesh_2D_original = np.meshgrid(x_2D_origin, y_2D_origin, indexing='ij')
    # boundary_3D = [[0, 0, 0], [res_x_3D, res_y_3D, res_z_3D]]  # boundaries for 3d knot
    extend = [*xy_lim_2D_origin, *xy_lim_2D_origin]  # boundaries for 2d plot
    xy_lim_2D_crop = list(np.array(xy_lim_2D_origin) / res_xy_2D_origin * crop)
    extend_crop = [*xy_lim_2D_crop, *xy_lim_2D_crop]  # boundaries for 2d plot after crop
    xy_lim_2D_crop3d = list(np.array(xy_lim_2D_crop) / crop * crop_3d)
    extend_crop3d = [*xy_lim_2D_crop3d, *xy_lim_2D_crop3d]  # boundaries for 2d plot after crop3d
    pxl_scale = (xy_lim_2D_origin[1] - xy_lim_2D_origin[0]) / (res_xy_2D_origin - 1)
    D_window = (xy_lim_2D_origin[1] - xy_lim_2D_origin[0])
    perfect_scale = lmbda * np.sqrt(L_prop ** 2 + (D_window / 2) ** 2) / D_window
    if print_values:
        print(f'dx={pxl_scale * 1e6: .2f}um, perfect={perfect_scale * 1e6: .2f}um,'
              f' resolution required={math.ceil(D_window / perfect_scale + 1)}')
    
    # extra values (turbulence)
    r0 = r0_from_Cn2(Cn2=Cn2, k0=k0, dz=L_prop)
    if print_values:
        print(f'r0 parameter: {r0}, 2w0/r0={2 * width0 / r0}')
    screens_num = screens_number(Cn2, k0, dz=L_prop)
    if print_values:
        print(f'Number of screen required: {screens_num}')
    ryt = rytov(Cn2, k0, L_prop)
    if print_values:
        print(f'SR={np.exp(-ryt)} (Rytov), Rytov={ryt}')
    zR = (k0 * width0 ** 2)
    if print_values:
        print(f'Rayleigh Range (Zr) = {zR} (m)')
    psh_par = (r0, res_xy_2D_origin, pxl_scale, L0, l0)
    psh_par_0 = (r0 * 1e100, res_xy_2D_origin, pxl_scale, L0, l0 * 1e100)
    
    if no_turb:
        psh_par = psh_par_0
    
    knot_types = {
        'optimized': hopf_optimized,  # 6
        'dennis': hopf_dennis,
        'trefoil_optimized': trefoil_optimized,
        'trefoil_standard_105': trefoil_standard_105,
        'trefoil_standard_11': trefoil_standard_11,
        'trefoil_standard_115': trefoil_standard_115,
        'trefoil_standard_12': trefoil_standard_12,
        'trefoil_standard_125': trefoil_standard_12,
        'trefoil_standard_13': trefoil_standard_13,
        'trefoil_standard_15': trefoil_standard_15,
        'trefoil_dennis': trefoil_dennis,
        'trefoil_optimized_math_5': trefoil_optimized_math_5,
        'trefoil_optimized_math_many': trefoil_optimized_math_many,
        'trefoil_optimized_math_many_095': trefoil_optimized_math_many,
    }
    knots = [
        'standard_14', 'standard_16', 'standard_18', '30both', '30oneZ',
        'optimized', 'pm_03_z', '4foil', '6foil', 'stand4foil',
        '30oneX'
    ]
    knots = [
        # 'trefoil_standard_105',
        # 'trefoil_standard_11',
        # 'trefoil_standard_115',
        # 'trefoil_standard_12',
        # 'trefoil_standard_13',
        # 'trefoil_standard_125',
        # 'trefoil_standard_15',
    ]
    knots = [
        # 'trefoil_optimized_math_5',
        # 'trefoil_optimized_math_many',
        'trefoil_optimized_math_many_095',
    ]
    knots = [
        'trefoil_optimized'
    ]
    folder_path = os.path.join("..", folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    folder_path = os.path.join("..", folder, "fields")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # getting the knot
    for knot in knots:
        print(knot)
        if os.path.exists(f'../{folder}\{knot}.pkl'):
            with open(f'../{folder}\{knot}.pkl', 'rb') as file:
                values = pickle.load(file)
        else:
            values = knot_types[knot](mesh_3D_knot, braid_func=braid, plot=True)
            with open(f'../{folder}\{knot}.pkl', 'wb') as file:
                pickle.dump(values, file)
        # printing values
        if print_coeff:
            values_print = np.real(values['weight']) / np.sqrt(np.sum(np.real(values['weight']) ** 2)) * 10
            values_print_formatted = [f'{val:.2f}' for val in values_print]
            print(f'vales l: {values["l"]}, p: {values["p"]}, weights: {values_print_formatted}')

        # building the knot from the coefficients
        field_before_prop = field_knot_from_weights(
            values, mesh_2D_original, width0, k0=k0, x0=0, y0=0, z0=z0
        )
    # plot_field_both(field_before_prop)
    # exit()
    field_before_prop = complex_resized
# center plane
    if plot:
        plot_field_both(field_before_prop)
    for indx in trange(SAMPLES, desc="Progress"):
        # propagating in the turbulence prop1
        field_after_turb = propagation_ps(
            field_before_prop, beam_par, psh_par, prop1, multiplier=multiplier1, screens_num=screens_num1, seed=seed
        )
        if plot:
            plot_field_both(field_after_turb, extend=extend)
        if center_plane == 1:
            field_center = field_after_turb
        else:
            field_center = propagation_ps(
                field_after_turb, beam_par, psh_par_0, prop2, multiplier=multiplier2, screens_num=screens_num2,
                seed=seed
            )
        ###########################
        field_center = field_center / np.sqrt(np.sum(np.abs(field_center) ** 2))
       
        ###########################
        # if plot:
        plot_field_both(field_center, extend=extend)

        field_z_crop = field_center[
                       res_xy_2D_origin // 2 - crop // 2: res_xy_2D_origin // 2 + crop // 2,
                       res_xy_2D_origin // 2 - crop // 2: res_xy_2D_origin // 2 + crop // 2,
                       ]
        # if 1:
        #
        #     filename = f'../{folder}\\fields\\data_{knot}_{indx + indx_plus}.npy'  # l rows, p columns
        #
        #     np.save(filename, field_z_crop)
        if plot:
            plot_field_both(field_z_crop, extend=extend_crop)

        if spectrum_save:
            if centering:
                x_cent_R_big, y_cent_R_big = find_center_of_intensity(field_center)
                x_cent_big, y_cent_big = x_cent_R_big, y_cent_R_big

                x_cent_big_r = x_2D_origin[0] + (x_2D_origin[-1] - x_2D_origin[0]) / res_xy_2D_origin * round(x_cent_big)

                y_cent_big_r = y_2D_origin[0] + (y_2D_origin[-1] - y_2D_origin[0]) / res_xy_2D_origin * round(y_cent_big)
            else:
                x_cent_big_r = 0
                y_cent_big_r = 0

            moments = {'p': (0, 6), 'l': (-6, 6)}
            # mesh_2D =
            spectrum = cbs.LG_spectrum(
                field_center, **moments, mesh=mesh_2D_original, plot=False, width=width0, k0=k0,
                functions=LG_simple, x0=x_cent_big_r, y0=y_cent_big_r
            )

            # if 1:
            #     filename = f'../{folder}\data_{knot}_spectr.csv'  # l rows, p columns
            #     spectrum_list = (
            #             [moments['l'][0], moments['l'][1], moments['p'][0], moments['p'][1]] + [indx + indx_plus] +
            #             [[x.real, x.imag] for x in spectrum.flatten()]
            #     )
            #     dots_json = json.dumps(spectrum_list)
            #     with open(filename, 'a', newline='') as file:
            #         writer = csv.writer(file)
            #         writer.writerow([dots_json])
        # plot_field_both(field_center)
        # exit()
        field_3d = beam_expander(field_z_crop, beam_par, psh_par_0, distance_both=knot_length, steps_one=res_z // 2)
        plot_field_both(field_3d[:, :, -1], extend=extend)
        if centering:
            x_cent_R, y_cent_R = find_center_of_intensity(field_z_crop)
            x_cent, y_cent = int(x_cent_R), int(y_cent_R)
        else:
            x_cent, y_cent = crop // 2, crop // 2

        if print_values:
            print(f'Centers: {x_cent}, {y_cent} out of {crop}')

        field_3d_crop = field_3d[
                        x_cent - crop_3d // 2: x_cent + crop_3d // 2,
                        y_cent - crop_3d // 2: y_cent + crop_3d // 2,
                        :
                        ]
        if no_last_plane:
            field_3d_crop = field_3d_crop[:, :, :-1]
        if plot_3d:
            plot_field_both(field_3d_crop[:, :, res_z // 2], extend=extend_crop3d)

        dots_init_dict, dots_init = sing.get_singularities(np.angle(field_3d_crop), axesAll=False, returnDict=True)

        dots_cut_non_unique = cut_circle_dots(dots_init, crop_3d // 2, crop_3d // 2, crop_3d // 2)

        # check if there is no same points
        # Creating a view with a compound data type
        view = np.ascontiguousarray(dots_cut_non_unique).view(
            np.dtype((np.void, dots_cut_non_unique.dtype.itemsize * dots_cut_non_unique.shape[1]))
        )
        # Using np.unique with the view
        _, idx = np.unique(view, return_index=True)
        dots_cut = dots_cut_non_unique[idx]

        # decreasing the resolution of knots
        # knot_resolution = [crop_3d, crop_3d, res_z + 1]
        original_resolution = (crop_3d, crop_3d)

        scale_x = new_resolution[0] / original_resolution[0]
        scale_y = new_resolution[1] / original_resolution[1]
        xy = dots_cut[:, :2]  # First two columns (x and y)
        z = dots_cut[:, 2]  # Third column
        scaled_xy = xy * [scale_x, scale_y]
        scaled_xy = np.rint(scaled_xy).astype(int)
        scaled_data = np.column_stack((scaled_xy, z))
        if plot_3d:
            dots_bound = [
                [0, 0, 0],
                [crop_3d, crop_3d, res_z + 1],
            ]
            pl.plotDots(dots_cut, dots_bound, color='black', show=True, size=10)

        if no_last_plane:
            knot_resolution = [new_resolution[0], new_resolution[0], res_z]
        else:
            knot_resolution = [new_resolution[0], new_resolution[0], res_z + 1]
        dots_cut_modified = np.vstack([[indx + indx_plus, 0, 0], knot_resolution, scaled_data])
        # if 0:
        #     filename = f'../{folder}\data_{knot}.csv'
        #     dots_json = json.dumps(dots_cut_modified.tolist())
        #     with open(filename, 'a', newline='') as file:
        #         writer = csv.writer(file)
        #         writer.writerow([dots_json])


