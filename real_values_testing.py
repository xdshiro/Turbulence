import numpy as np
import matplotlib.pyplot as plt
from aotools.turbulence import phasescreen
from scipy.special import laguerre
from aotools import opticalpropagation
from aotools.turbulence.phasescreen import ft_phase_screen as ps
from aotools.turbulence.phasescreen import ft_sh_phase_screen as psh
from functions.functions_turbulence import *

"""
r0 (float) – r0 parameter of scrn in metres
N (int) – Size of phase scrn in pxls
delta (float) – size in Metres of each pxl
L0 (float) – Size of outer-scale in metres
l0 (float) – inner scale in metres
"""
# %% Beam parameters
lmbda = 633e-9
L_prop = 150  # ! точно
width0 = 15e-3  # !? может точно
l, p = 0, 0
# xy_lim_2D = (-8.0e-6, 8.0e-6)
xy_lim_2D = (-50.0e-3, 50.0e-3)
res_xy_2D = 201

# print(51 % 2)
# ar = np.arange(-(51 - 1) / 2, (51 + 1) / 2)
# print(ar, len(ar))
# exit()

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
# Cn2_real_array = [1e-17, 1e-16, 1e-15, 1e-14, 1e-13]  # weak -> strong
# 3.21 × 10−14
# 1.35 × 10−13 m−2∕3
Cn2 = 10e-1
Cn2 = 5e-1
Cn2 = 3.21e-14
# Cn2 = 1.35e-13
Cn2 = 1.35e-13
r0 = r0_from_Cn2(Cn2=Cn2, k0=k0, dz=L_prop)
print(f'r0 parameter: {r0}')
L0 = 5
L0 = 9
l0 = 10e-3  # !!!!!!
psh_par = (r0, res_xy_2D, pxl_scale, L0, l0)

# %%

screens_num = screens_number(Cn2, k0, dz=L_prop)
print(f'Number of screen required: {screens_num}')

ryt = rytov(Cn2, k0, L_prop)
print(f'SR={np.exp(-ryt)} (Rytov)')
# exit()
# exit()
LG_21_2D = LG_simple(*mesh_2D, z=0, l=l, p=p, width=width0, k0=k0, x0=0, y0=0, z0=0)
# LG_21_2D_z01 = LG_simple(*mesh_2D, z=L_prop, l=2, p=1, width=width0, k0=k0, x0=0, y0=0, z0=0)


# plot_field_both(LG_21_2D, extend=None)
# plot_field_both(LG_21_2D_z01, extend=None)


# psh_test = psh_wrap(psh_par)




# field = propagation_ps(LG_21_2D, beam_par, psh_par, L_prop, screens_num=50)
# plot_field_both(field, extend=None)
#
SR_gauss(mesh_2D, L_prop, beam_par, psh_par, epochs=100, screens_num=5, max_cut=True)
# 0.81