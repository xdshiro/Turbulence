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
lmbda = 550e-9
L_prop = 4e-5
width0 = 1e-6
l, p = 0, 0
# xy_lim_2D = (-8.0e-6, 8.0e-6)
xy_lim_2D = (-20.0e-6, 20.0e-6)
res_xy_2D = 150

beam_par = (l, p, width0, lmbda)
k0 = 2 * np.pi / lmbda
xy_2D = np.linspace(*xy_lim_2D, res_xy_2D)
mesh_2D = np.meshgrid(xy_2D, xy_2D, indexing='ij')
pxl_scale = (xy_lim_2D[1] - xy_lim_2D[0]) / (res_xy_2D - 1)
D_window = (xy_lim_2D[1] - xy_lim_2D[0])
perfect_scale = lmbda * np.sqrt(L_prop ** 2 + (D_window / 2) ** 2) / D_window
print(f'dx={pxl_scale * 1e6: .2f}um, perfect={perfect_scale * 1e6: .2f}um,'
      f' resolution required={math.ceil(D_window / perfect_scale + 1)}')
assert res_xy_2D > D_window / perfect_scale + 1, 'Resolution is too low'
# %% turbulence parameters
# Cn2_real_array = [1e-17, 1e-16, 1e-15, 1e-14, 1e-13]  # strong -> weak
Cn2 = 10e-1
Cn2 = 5e-1
Cn2 = 10e-1
r0 = r0_from_Cn2(Cn2=Cn2, k0=k0, dz=L_prop)
L0 = 5
l0 = 1e-7  # !!!!!!
psh_par = (r0, res_xy_2D, pxl_scale, L0, l0)

# %%

screens_num = screens_number(Cn2, k0, dz=L_prop)
print(f'Number of screen required: {screens_num}')
# exit()
LG_21_2D = LG_simple(*mesh_2D, z=0, l=2, p=1, width=width0, k0=k0, x0=0, y0=0, z0=0)
# LG_21_2D_z01 = LG_simple(*mesh_2D, z=L_prop, l=2, p=1, width=width0, k0=k0, x0=0, y0=0, z0=0)


# plot_field_both(LG_21_2D, extend=None)
# plot_field_both(LG_21_2D_z01, extend=None)


# psh_test = psh_wrap(psh_par)




field = propagation_ps(LG_21_2D, beam_par, psh_par, L_prop, screens_num=50)
plot_field_both(field, extend=None)

# SR_gauss(mesh_2D, L_prop, beam_par, psh_par, epochs=100, screens_num=1)
# 0.81