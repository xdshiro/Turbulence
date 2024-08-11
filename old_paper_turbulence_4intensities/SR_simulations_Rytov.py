import numpy as np
import matplotlib.pyplot as plt
from aotools.turbulence import phasescreen
from scipy.special import laguerre
from aotools import opticalpropagation
from aotools.turbulence.phasescreen import ft_phase_screen as ps
from aotools.turbulence.phasescreen import ft_sh_phase_screen as psh
from functions.functions_turbulence import *
from functions.all_knots_functions import *
import math


def crop_field_3d(field_3d, crop_percentage):
    # Get the shape of the field_3d
    x_size, y_size, z_size = field_3d.shape

    # Calculate the center indices
    center_x, center_y = x_size // 2, y_size // 2

    # Calculate the crop size
    crop_x = int(crop_percentage * x_size / 100)
    crop_y = int(crop_percentage * y_size / 100)


    # Calculate the boundaries
    start_x = max(center_x - crop_x // 2, 0)
    end_x = min(center_x + crop_x // 2, x_size)
    start_y = max(center_y - crop_y // 2, 0)
    end_y = min(center_y + crop_y // 2, y_size)

    # Slice the array
    cropped_field = field_3d[start_x:end_x, start_y:end_y, :]

    return cropped_field, end_x - start_x, end_y - start_y
def run_simulation(L_prop, width0, xy_lim_2D, res_xy_2D, Rytov, l0, L0, screens_nums):
    # Beam parameters
    lmbda = 633e-9
    width_values = width0
    l, p = 0, 0

    beam_par = (l, p, width0, lmbda)
    k0 = 2 * np.pi / lmbda
    xy_2D = np.linspace(*xy_lim_2D, res_xy_2D)
    mesh_2D = np.meshgrid(xy_2D, xy_2D, indexing='ij')
    pxl_scale = (xy_lim_2D[1] - xy_lim_2D[0]) / (res_xy_2D - 1)
    D_window = (xy_lim_2D[1] - xy_lim_2D[0])
    perfect_scale = lmbda * np.sqrt(L_prop ** 2 + (D_window / 2) ** 2) / D_window
    print(f'dx={pxl_scale * 1e6: .2f}um, perfect={perfect_scale * 1e6: .2f}um,'
          f' resolution required={math.ceil(D_window / perfect_scale + 1)}')

    # Turbulence parameters

    Cn2 = Cn2_from_Rytov(Rytov, k0, L_prop)
    r0 = r0_from_Cn2(Cn2=Cn2, k0=k0, dz=L_prop)
    print(f'r0 parameter: {r0}, 2w0/r0={2 * width_values / r0}')

    psh_par = (r0, res_xy_2D, pxl_scale, L0, l0)
    psh_par_0 = (r0 * 1e100, res_xy_2D, pxl_scale, L0, l0 * 1e100)
    screens_num = screens_number(Cn2, k0, dz=L_prop)
    print(f'Number of screen required: {screens_num}')

    ryt = rytov(Cn2, k0, L_prop)
    print(f'SR={np.exp(-ryt)} (Rytov {ryt})')
    print(f'Cn2 from Rytov {Cn2_from_Rytov(Rytov, k0, L_prop)}')
    LG_21_2D = LG_simple(*mesh_2D, z=0, l=l, p=p, width=width0, k0=k0, x0=0, y0=0, z0=0)
    l_save = [0, 0, 0, 0, 3]
    p_save = [0, 1, 2, 3, 0]
    weight_save = [1.29, -3.95, 7.49, -3.28, -3.98]
    LG_21_2D = 0
    for i in range(5):
        LG_21_2D += weight_save[i] * LG_simple(*mesh_2D, z=-L_prop, l=l_save[i], p=p_save[i],
                                               width=width0, k0=k0, x0=0, y0=0, z0=0)
    # plot_field_both(LG_21_2D, extend=None)
    # phase_screen = psh_wrap(psh_par, seed=1)
    # plot_field(phase_screen, extend=None)
    # plot_field_both(np.exp(1j*phase_screen), extend=None)

    field_prop = propagation_ps(LG_21_2D, beam_par, psh_par, L_prop, screens_num=screens_nums)
    # plot_field_both(field_prop)
    # print(phase_screen)
    # plt.show()

    # field_3d = beam_expander(field_prop, beam_par, psh_par_0, distance_both=30, steps_one=40 // 2)
    # cropped_field_3d, end_x, end_y = crop_field_3d(field_3d, 40)
    # # print(cropped_field_3d.shape)
    # dots_init_dict, dots_init = sing.get_singularities(np.angle(cropped_field_3d), axesAll=True, returnDict=True)
    # dots_bound = [
    #     [0, 0, 0],
    #     [end_x, end_y, 40 + 1],  # Assuming z limit remains the same
    # ]
    # pl.plotDots(dots_init_dict, dots_bound, color='black', show=True, size=10)
    SR = SR_gauss_fourier(mesh_2D, L_prop, beam_par, psh_par, epochs=2000, screens_num=screens_nums, max_cut=False, pad_factor=4)
    # scin = scintillation(mesh_2D, L_prop, beam_par, psh_par, epochs=10000, screens_num=screens_nums, seed=None)
    # scin_middle = scin[res_xy_2D // 2, res_xy_2D // 2]
    # scin_f = scintillation_fourier(mesh_2D, L_prop, beam_par, psh_par, epochs=500, screens_num=screens_nums, seed=None)
    # scin_f_middle = scin_f[res_xy_2D // 2, res_xy_2D // 2]
    # scin_rev = scintillation_reversed(mesh_2D, L_prop, beam_par, psh_par, epochs=1500, screens_num=screens_nums, seed=None)
    # scin_rev_middle = np.average(scin_rev[res_xy_2D // 2 - 3:res_xy_2D // 2 + 4,
    #                              res_xy_2D // 2 - 3:res_xy_2D // 2 + 4])
    # # print(f'SCIN={scin_middle}, PS={screens_nums}')
    # # print(f'SCIN={scin_f_middle}')
    # print(f'SCIN={scin_rev_middle}')
    # # plot_field_both(scin, extend=None)
    #
    # with open('simulation_results_015_100_10_501.txt', 'a') as f:
    #     # f.write(f'{SR}, {scin_middle}, {scin_rev_middle}\n')
    #     f.write(f'{scin_rev_middle}\n')


# Define the sets of values you want to iterate over
L_prop_values = [100]
width0_values = [5e-3 / np.sqrt(2)]
# width0_values = [5e-3 / np.sqrt(2) * 10]
xy_lim_2D_values = [(-30.0e-3, 30.0e-3)]
# xy_lim_2D_values = [(-30.0e-3 * 5, 30.0e-3 * 5)]
res_xy_2D_values = [301]

# Cn2_values = [5e-15, 1e-14, 5e-14, 1e-13]
# Cn2_values = [1e-13]
Rytov_values = [0.075, 0.03, 0.05, 0.75, 0.1, 0.15]
Rytov_values = [0.025]
l0_values = [5e-3 * 1e-10]
L0_values = [10 * 1e10]
# screens_numss = [1,2,3,4,5,10]
screens_numss = [2]
# screens_numss = [10,2,3,4,5,10]

# Ensure all lists are the same length by repeating the single-element lists
max_len = max(len(L_prop_values), len(width0_values), len(xy_lim_2D_values),
              len(res_xy_2D_values), len(Rytov_values), len(l0_values), len(L0_values))


L_prop_values = L_prop_values if len(L_prop_values) > 1 else L_prop_values * max_len
width0_values = width0_values if len(width0_values) > 1 else width0_values * max_len
xy_lim_2D_values = xy_lim_2D_values if len(xy_lim_2D_values) > 1 else xy_lim_2D_values * max_len
res_xy_2D_values = res_xy_2D_values if len(res_xy_2D_values) > 1 else res_xy_2D_values * max_len
Rytov_values = Rytov_values if len(Rytov_values) > 1 else Rytov_values * max_len
l0_values = l0_values if len(l0_values) > 1 else l0_values * max_len
L0_values = L0_values if len(L0_values) > 1 else L0_values * max_len
parameter_sets = list(zip(L_prop_values, width0_values, xy_lim_2D_values, res_xy_2D_values, Rytov_values, l0_values, L0_values))

# Run simulations for each combination of values

for params in parameter_sets:
    for screens_nums in screens_numss:
        print(f'Simulation parameters: {params}, screens={screens_nums}')

        run_simulation(*params, screens_nums=screens_nums)