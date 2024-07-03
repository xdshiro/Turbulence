import numpy as np
import matplotlib.pyplot as plt
from aotools.turbulence import phasescreen
from scipy.special import laguerre
from aotools import opticalpropagation
from aotools.turbulence.phasescreen import ft_phase_screen as ps
from aotools.turbulence.phasescreen import ft_sh_phase_screen as psh
from functions.functions_turbulence import *
from functions.all_knots_functions import *
"""
r0 (float) – r0 parameter of scrn in metres
N (int) – Size of phase scrn in pxls
delta (float) – size in Metres of each pxl
L0 (float) – Size of outer-scale in metres
l0 (float) – inner scale in metres
"""
# %% Beam parameters
lmbda = 532e-9
L_prop = 150e-3
L_prop = 150e-3
width0 = 2e-3
width_values = width0 / np.sqrt(2)
l, p = 0, 0
# xy_lim_2D = (-8.0e-6, 8.0e-6)
xy_lim_2D = np.array((-10.0e-3, 10.0e-3))
res_xy_2D = 91

# Rytov variance should be smaller than 1 for each segment
beam_par = (l, p, width0, lmbda)
k0 = 2 * np.pi / lmbda
xy_2D = np.linspace(*xy_lim_2D, res_xy_2D)
mesh_2D = np.meshgrid(xy_2D, xy_2D, indexing='ij')
pxl_scale = (xy_lim_2D[1] - xy_lim_2D[0]) / (res_xy_2D - 1)
D_window = (xy_lim_2D[1] - xy_lim_2D[0])
perfect_scale = lmbda * np.sqrt(L_prop ** 2 + (D_window / 2) ** 2) / D_window
print(f'dx={pxl_scale * 1e6: .2f}um, perfect={perfect_scale * 1e6: .2f}um,'
      f' resolution required={math.ceil(D_window / perfect_scale + 1)}')
# assert res_xy_2D > D_window / perfect_scale + 1, 'Resolution is too low'


# %% turbulence parameters

Cn2 = 10e-1
Cn2 = 5e-1
Cn2 = 3.21e-14
Cn2 = 3.21e-10

r0 = r0_from_Cn2(Cn2=Cn2, k0=k0, dz=L_prop)

print(f'r0 parameter: {r0}, 2w0/r0={2 * width_values / r0}')

L0 = 5
L0 = 9
l0 = 5e-3  # !!!!!!

psh_par = (r0, res_xy_2D, pxl_scale, L0, l0)
psh_par_0 = (r0 * 1e100, res_xy_2D, pxl_scale, L0, l0 * 1e100)

# %%

screens_num = screens_number(Cn2, k0, dz=L_prop)
print(f'Number of screen required: {screens_num}')

ryt = rytov(Cn2, k0, L_prop)
print(f'SR={np.exp(-ryt)} (Rytov {ryt})')

LG_21_2D = LG_simple(*mesh_2D, z=0, l=l, p=p, width=width0, k0=k0, x0=0, y0=0, z0=0)
l_save = [0, 0, 0, 0, 3]
p_save = [0, 1, 2, 3, 0]
weight_save = [1.29, -3.95, 7.49, -3.28, -3.98]
LG_21_2D = 0
for i in range(5):
    LG_21_2D += weight_save[i] * LG_simple(*mesh_2D, z=-L_prop, l=l_save[i], p=p_save[i],
                                           width=width0, k0=k0, x0=0, y0=0, z0=0)
plot_field_both(LG_21_2D, extend=None)
phase_screen = psh_wrap(psh_par, seed=1)
# print((np.sum(np.abs(LG_21_2D * np.exp(1j * phase_screen)) ** 2) * pxl_scale ** 2) / (np.sum(np.abs(LG_21_2D) ** 2) * pxl_scale ** 2))
plot_field(phase_screen, extend=None)

field_prop = propagation_ps(LG_21_2D, beam_par, psh_par, L_prop, screens_num=8)
plot_field_both(field_prop)
plt.show()
field_3d = beam_expander(field_prop, beam_par, psh_par_0, distance_both=20, steps_one=20 // 2)
dots_init_dict, dots_init = sing.get_singularities(np.angle(field_3d), axesAll=False, returnDict=True)
dots_bound = [
                [0, 0, 0],
                [res_xy_2D, res_xy_2D, 20 + 1],
            ]
pl.plotDots(dots_init_dict, dots_bound, color='black', show=True, size=10)
# dots_cut_non_unique = cut_circle_dots(dots_init, N // 2, crop_3d // 2, crop_3d // 2)
SR_gauss_fourier(mesh_2D, L_prop, beam_par, psh_par, epochs=200, screens_num=1, max_cut=False, pad_factor=4)
def scintillation(mesh_2D, L_prop, beam_par, psh_par, epochs=100, screens_num=1, max_cut=False, seed=None):
    _, _, width0, lmbda = beam_par
    k0 = 2 * np.pi / lmbda
    r0, N, pxl_scale, L0, l0 = psh_par
    xy_array, _ = arrays_from_mesh(mesh_2D)
    # res_xy_2D = len(xy_array)
    # xy_scale = xy_array[1] - xy_array[0]
    # assert N == len(xy_array), 'Resolution of the beam isn"t equal to the phase screen N'
    # assert len(xy_array) % 2 == 0, 'Odd resolution of the beam'
    LG_00 = LG_simple(*mesh_2D, z=0, l=0, p=0, width=width0, k0=k0, x0=0, y0=0, z0=0)
    # I0 = np.abs(LG_simple(x=0, y=0, z=L_prop,
    #                       l=0, p=0, width=width0, k0=k0, x0=0, y0=0, z0=0)) ** 2
    I_avg_tot = 0
    I_sqr_avg_tot = 0

    Cn2 = Cn2_from_r0(r0, k0, L_prop)
    dL = L_prop / screens_num
    r0d = r0_from_Cn2(Cn2=Cn2, k0=k0, dz=dL)
    psh_par_dL = r0d, N, pxl_scale, L0, l0
    # E = beam_2D

    for i in range(epochs):
        E = LG_00
        for _ in range(screens_num):
            if seed is not None:
                phase_screen_i = psh_wrap(psh_par_dL, seed=seed+i)
            else:
                phase_screen_i = psh_wrap(psh_par_dL)
            E = opticalpropagation.angularSpectrum(
                E * np.exp(1j * phase_screen_i), lmbda, pxl_scale, pxl_scale, dL
            )
        current = np.abs(E) ** 2
        I_avg_tot += current
        I_sqr_avg_tot += current ** 2
        # print(current, I0)
        # if max_cut:
        #     I_avg_tot += min(current, I0)
        # else:
        #     I_avg_tot += current
        if i == 1:
            plot_field_both(E, extend=None)
    scin = (I_sqr_avg_tot / epochs) / (I_avg_tot / epochs) ** 2 - 1
    return scin
exit()
# print()
# exit()
scin = scintillation(mesh_2D, L_prop, beam_par, psh_par, epochs=1000, screens_num=2 , max_cut=False, seed=None)
print(scin[res_xy_2D // 2, res_xy_2D // 2])
plot_field_both(scin, extend=None)
exit()
print(SR_gauss(mesh_2D, L_prop, beam_par, psh_par, epochs=200, screens_num=3, max_cut=False))

# exit()
# LG_21_2D_z01 = LG_simple(*mesh_2D, z=L_prop, l=2, p=1, width=width0, k0=k0, x0=0, y0=0, z0=0)


# plot_field_both(LG_21_2D, extend=None)
# plot_field_both(LG_21_2D_z01, extend=None)


# psh_test = psh_wrap(psh_par)


# field = propagation_ps(LG_21_2D, beam_par, psh_par, L_prop, screens_num=50)
# plot_field_both(field, extend=None)
#

# 0.81
