import matplotlib.pyplot as plt

from functions.all_knots_functions import *
import os
import pickle
import csv
import json
from tqdm import trange
from scipy.ndimage import zoom
import scipy.io

SAMPLES = 200
indx_plus = 0

plot = 0
plot_3d = 0
print_coeff = 0


def process_knot_fields4(fields, X, reso=256, pad_factor=16, crop_factor=1, plot=True):
    # Load the .mat file containing the absolute values
    # Extract the variable containing the absolute values
    # max_amp = max([np.max(np.abs(field_)) for field_ in fields3])
    # size_interf = np.shape(fields3[0])
    # print(max_amp)
    phases = [0, np.pi/2, np.pi, 3*np.pi/2]
    k_x = 1e7
    # Convert the MATLAB cell array to a list of NumPy arrays
    # absolute_fields_list = [abs(field + np.max(np.abs(field)) * np.exp(1j * phases[i] + 1j * X * k_x)) for i, field in enumerate(fields)]
    # absolute_fields_list = [abs(field + np.max(np.abs(field)) * np.exp(1j * phases[i] + 1j * X * k_x)) for i, field in enumerate(fields)]
    max_amp = max([np.max(np.abs(field_)) for field_ in fields])

    # Convert the MATLAB cell array to a list of NumPy arrays
    absolute_fields_list = [abs(field + max_amp * np.exp(1j * phases[i] + 1j * X * k_x)) ** 2 for i, field in
                            enumerate(fields)]

    # Access individual fields
    I1 = absolute_fields_list[0]
    I2 = absolute_fields_list[1]
    I3 = absolute_fields_list[2]
    I4 = absolute_fields_list[3]
    # plot_field_both(fields3[0])
    # plot_field_both(fields3[1])
    # plot_field_both(fields3[2])
    # plot_field_both(fields3[3])
    plot_field_both(I1)
    # plot_field_both(I2)
    # plot_field_both(I3)
    # plot_field_both(I4)
    # Compute the phase and signal
    phase = np.arctan2(I4 - I2, I1 - I3)
    signal = (I1 - I3) ** 2 + (I4 - I2) ** 2
    U_f = np.sqrt(signal) * np.exp(1j * phase)

    # Upscale U_f to specified resolution
    target_size = (reso, reso)
    # U_f_real = np.real(U_f)
    # U_f_imag = np.imag(U_f)
    #
    # U_f_real_resized = zoom(U_f_real, (target_size[0] / U_f.shape[0], target_size[1] / U_f.shape[1]), order=3)
    # U_f_imag_resized = zoom(U_f_imag, (target_size[0] / U_f.shape[0], target_size[1] / U_f.shape[1]), order=3)
    #
    # U_f_resized = U_f_real_resized + 1j * U_f_imag_resized
    U_f_resized = U_f

    if plot:
        plot_field_both(U_f_resized)

    # Control pad size
    pad_size = (int(U_f_resized.shape[0] * pad_factor), int(U_f_resized.shape[1] * pad_factor))
    pad_width = ((pad_size[0] // 2, pad_size[0] // 2), (pad_size[1] // 2, pad_size[1] // 2))
    U_f_padded = np.pad(U_f_resized, pad_width, mode='constant', constant_values=0)

    # Take the 2D FFT
    U_f_fft = np.fft.fftshift(np.fft.fft2(U_f_padded))

    if plot:
        plot_field_both(U_f_fft)

    # Find the maximum value and its position
    max_idx = np.unravel_index(np.argmax(np.abs(U_f_fft)), U_f_fft.shape)

    # Define the size of the cropped region around the maximum value
    crop_size = int(reso * crop_factor)
    half_crop_size = crop_size // 2

    # Crop the region around the maximum value
    start_row = max(0, max_idx[0] - half_crop_size)
    end_row = min(U_f_fft.shape[0], max_idx[0] + half_crop_size)
    start_col = max(0, max_idx[1] - half_crop_size)
    end_col = min(U_f_fft.shape[1], max_idx[1] + half_crop_size)

    U_f_fft_cropped = U_f_fft[start_row:end_row, start_col:end_col]

    if plot:
        plot_field_both(U_f_fft)

    # Pad the cropped FFT
    pad_size2 = (int(U_f_fft_cropped.shape[0] * (pad_factor / crop_factor)),
                 int(U_f_fft_cropped.shape[1] * (pad_factor / crop_factor)))
    pad_width2 = ((pad_size2[0] // 2, pad_size2[0] // 2), (pad_size2[1] // 2, pad_size2[1] // 2))
    U_f_fft_cropped_padded = np.pad(U_f_fft_cropped, pad_width2, mode='constant', constant_values=0)

    # Compute the IFFT of the cropped field
    U_f_ifft_cropped = np.fft.ifft2(np.fft.ifftshift(U_f_fft_cropped_padded))

    if plot:
        plot_field_both(U_f_ifft_cropped)

    # Define the size of the cropped region for the IFFT result
    ifft_crop_size = reso
    half_ifft_crop_size = ifft_crop_size // 2
    print(ifft_crop_size)
    # Find the center of the IFFT result
    ifft_center = (U_f_ifft_cropped.shape[0] // 2, U_f_ifft_cropped.shape[1] // 2)

    # Crop the region around the center
    start_row_ifft = max(0, ifft_center[0] - half_ifft_crop_size)
    end_row_ifft = min(U_f_ifft_cropped.shape[0], ifft_center[0] + half_ifft_crop_size)
    start_col_ifft = max(0, ifft_center[1] - half_ifft_crop_size)
    end_col_ifft = min(U_f_ifft_cropped.shape[1], ifft_center[1] + half_ifft_crop_size)

    U_f_ifft_cropped_final = U_f_ifft_cropped[start_row_ifft:end_row_ifft, start_col_ifft:end_col_ifft]

    if plot:
        plot_field_both(U_f_ifft_cropped_final)

    return U_f_ifft_cropped_final
def process_knot_fields3(fields, X, reso=256, pad_factor=16, crop_factor=1, plot=True):
    # Load the .mat file containing the absolute values
    # Extract the variable containing the absolute values


    max_amp = max([np.max(np.abs(field_)) for field_ in fields])
    # size_interf = np.shape(fields3[0])
    # print(max_amp)
    k_x = 4e3
    # Convert the MATLAB cell array to a list of NumPy arrays
    # phases = [ -2 * np.pi/3, 0, 2*np.pi/3]
    phases = [ np.pi/4, 3*np.pi/4, 5*np.pi/4]
    absolute_fields_list = [abs(field + max_amp * np.exp(1j * phases[i] + 1j * X * k_x)) ** 2 for i, field in enumerate(fields)]

    # Access individual fields
    I1 = absolute_fields_list[0]
    I2 = absolute_fields_list[1]
    I3 = absolute_fields_list[2]
    # plot_field_both(fields[0])
    # plot_field_both(fields[1])
    # I4 = absolute_fields_list[3]
    # plot_field_both(I1)
    # plot_field_both(fields3[0])
    # plot_field_both(I2)
    # plot_field_both(I3)
    # plot_field_both(I4)
    # Compute the phase and signal
    # phase = np.arctan2(np.sqrt(3) * (I1 - I3), (2 * I1 - I2 - I3))
    # signal = np.sqrt(3 * (I1 - I3) ** 2 + (2 * I2 - I1 - I3) ** 2)
    phase = np.arctan2((I3 - I2), (I1 - I2))
    signal = np.sqrt((I3 - I2) ** 2 + (I1 - I2) ** 2)
    U_f = signal * np.exp(1j * phase)

    # Upscale U_f to specified resolution
    target_size = (reso, reso)
    # U_f_real = np.real(U_f)
    # U_f_imag = np.imag(U_f)
    #
    # U_f_real_resized = zoom(U_f_real, (target_size[0] / U_f.shape[0], target_size[1] / U_f.shape[1]), order=3)
    # U_f_imag_resized = zoom(U_f_imag, (target_size[0] / U_f.shape[0], target_size[1] / U_f.shape[1]), order=3)

    # U_f_resized = U_f_real_resized + 1j * U_f_imag_resized
    U_f_resized = U_f

    if plot:
        plot_field_both(U_f_resized)

    # Control pad size
    pad_size = (int(U_f_resized.shape[0] * pad_factor), int(U_f_resized.shape[1] * pad_factor))
    pad_width = ((pad_size[0] // 2, pad_size[0] // 2), (pad_size[1] // 2, pad_size[1] // 2))
    U_f_padded = np.pad(U_f_resized, pad_width, mode='constant', constant_values=0)

    # Take the 2D FFT
    U_f_fft = np.fft.fftshift(np.fft.fft2(U_f_padded))

    if plot:
        plot_field_both(U_f_fft)

    # Find the maximum value and its position
    max_idx = np.unravel_index(np.argmax(np.abs(U_f_fft)), U_f_fft.shape)

    # Define the size of the cropped region around the maximum value
    crop_size = int(reso * crop_factor)
    half_crop_size = crop_size // 2

    # Crop the region around the maximum value
    start_row = max(0, max_idx[0] - half_crop_size)
    end_row = min(U_f_fft.shape[0], max_idx[0] + half_crop_size)
    start_col = max(0, max_idx[1] - half_crop_size)
    end_col = min(U_f_fft.shape[1], max_idx[1] + half_crop_size)

    U_f_fft_cropped = U_f_fft[start_row:end_row, start_col:end_col]

    if plot:
        plot_field_both(U_f_fft)

    # Pad the cropped FFT
    pad_size2 = (int(U_f_fft_cropped.shape[0] * (pad_factor / crop_factor)),
                 int(U_f_fft_cropped.shape[1] * (pad_factor / crop_factor)))
    pad_width2 = ((pad_size2[0] // 2, pad_size2[0] // 2), (pad_size2[1] // 2, pad_size2[1] // 2))
    U_f_fft_cropped_padded = np.pad(U_f_fft_cropped, pad_width2, mode='constant', constant_values=0)

    # Compute the IFFT of the cropped field
    U_f_ifft_cropped = np.fft.ifft2(np.fft.ifftshift(U_f_fft_cropped_padded))

    if plot:
        plot_field_both(U_f_ifft_cropped)

    # Define the size of the cropped region for the IFFT result
    ifft_crop_size = reso
    half_ifft_crop_size = ifft_crop_size // 2


    # Find the center of the IFFT result
    ifft_center = (U_f_ifft_cropped.shape[0] // 2, U_f_ifft_cropped.shape[1] // 2)

    # Crop the region around the center
    start_row_ifft = max(0, ifft_center[0] - half_ifft_crop_size)
    end_row_ifft = min(U_f_ifft_cropped.shape[0], ifft_center[0] + half_ifft_crop_size)
    start_col_ifft = max(0, ifft_center[1] - half_ifft_crop_size)
    end_col_ifft = min(U_f_ifft_cropped.shape[1], ifft_center[1] + half_ifft_crop_size)

    U_f_ifft_cropped_final = U_f_ifft_cropped[start_row_ifft:end_row_ifft, start_col_ifft:end_col_ifft]

    if plot:
        plot_field_both(U_f_ifft_cropped_final)

    return U_f_ifft_cropped_final
print_values = 0
centering = 0
seed = None  # does work with more than 1 phase screen
no_last_plane = True

spectrum_save = 1
no_turb = 0

# meshes and boundaries for getting a knot
x_lim_3D_knot, y_lim_3D_knot, z_lim_3D_knot = (-7.0, 7.0), (-7.0, 7.0), (-2.0, 2.0)
res_x_3D_knot, res_y_3D_knot, res_z_3D_knot = 256, 256, 1

# beam
lmbda = 532e-9  # wavelength
L_prop = 270  # propagation distance
knot_length = 212.58897655870774 / 2 * 1.4  # we need RALEYIG!!!!!!!!  # 1000 how far is detector from the knot center
center_plane = 0.3
center_plane = 1
########################################################
width0 = 6.0e-3 / np.sqrt(2)  # beam width
xy_lim_2D_origin = (-35.0e-3, 35.0e-3)  # window size to start with
scale = 1.5
res_xy_2D_origin = int(scale * 300) # resolution

res_z = int(scale * 100)  # resolution of the knot is res_z+1
crop = int(scale * 185)  # for the knot propagation
crop_3d = int(scale * 100)  # for the knot
new_resolution = (int(scale * 100), int(scale * 100))  # resolution of the knot to save

screens_num1 = 3
multiplier1 = [1] * screens_num1


Rytovs = [0.00001]#, 0.2]
Rytovs = [0.025, 0.05, 0.1, 0.15, 0.2]
for Rytov in Rytovs:

    folder = f'optimized_L{L_prop}_{Rytov}_3int_200'

    k0 = 2 * np.pi / lmbda  # wave number
    Cn2 = Cn2_from_Rytov(Rytov, k0, L_prop)

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
    X = mesh_2D_original[0][
        res_xy_2D_origin // 2 - crop // 2: res_xy_2D_origin // 2 + crop // 2,
        res_xy_2D_origin // 2 - crop // 2: res_xy_2D_origin // 2 + crop // 2,
        ]
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
        if plot:
            plot_field_both(field_before_prop)
        for indx in trange(SAMPLES, desc="Progress"):
            
            
            fields3 = []
            for i in range(3):
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
                if plot:
                    plot_field_both(field_center, extend=extend)
        
                field_z_crop = field_center[
                               res_xy_2D_origin // 2 - crop // 2: res_xy_2D_origin // 2 + crop // 2,
                               res_xy_2D_origin // 2 - crop // 2: res_xy_2D_origin // 2 + crop // 2,
                               ]
                if 1:
        
                    filename = f'../{folder}\\fields\\data_{knot}_{indx + indx_plus}.npy'  # l rows, p columns
        
                    np.save(filename, field_z_crop)
                if plot:
                    plot_field_both(field_z_crop, extend=extend_crop)
                fields3.append(field_z_crop)


            # for field_z_crop in fields3:
            #     plot_field_both(field_z_crop) # + max_amp * np.exp())
            # fields3_test = [fields3[0], fields3[0],fields3[0],fields3[0],]
            field_z_crop = process_knot_fields3(fields3[0:3], X, reso=np.shape(X)[0], plot=False)
            # print(np.shape(X)[0])
            # field_z_crop = process_knot_fields4(fields3[0:4], X, reso=np.shape(X)[0], plot=True)
            # plot_field_both(fields3[0])
            # plot_field_both(field_z_crop)

            # plot_field_both(field_center)

















            field_3d = beam_expander(field_z_crop, beam_par, psh_par_0, distance_both=knot_length, steps_one=res_z // 2)
    
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
            if 1:
                filename = f'../{folder}\data_{knot}.csv'
                dots_json = json.dumps(dots_cut_modified.tolist())
                with open(filename, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([dots_json])


