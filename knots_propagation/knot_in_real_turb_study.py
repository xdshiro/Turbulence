from functions.functions_turbulence import *
import my_functions.singularities as sing
import my_functions.plotings as pl
import my_functions.functions_general as fg

# %%  main parameters

lmbda = 633e-9  # wavelength
L_prop = 150  # propagation distance
width0 = 15e-3 / np.sqrt(2)  # beam width
xy_lim_2D = (-100.0e-3, 100.0e-3)  # window size to start with
res_xy_2D = 401  # resolution
# Cn2 = 1.35e-13  # turbulence strength  is basically in the range of 10−17–10−12 m−2/3
Cn2 = 3.21e-14
# https://www.mdpi.com/2076-3417/11/22/10548
L0 = 9
l0 = 5e-3  # !!!!!!

crop = 120
z0 = 1000 + 100
prop1 = 100
prop2 = 1000
# l = 0
# p = 0
# %%
# grating
extend = [*xy_lim_2D, *xy_lim_2D]
k0 = 2 * np.pi / lmbda
xy_2D = np.linspace(*xy_lim_2D, res_xy_2D)
mesh_2D = np.meshgrid(xy_2D, xy_2D, indexing='ij')
pxl_scale = (xy_lim_2D[1] - xy_lim_2D[0]) / (res_xy_2D - 1)
D_window = (xy_lim_2D[1] - xy_lim_2D[0])
perfect_scale = lmbda * np.sqrt(L_prop ** 2 + (D_window / 2) ** 2) / D_window
print(f'dx={pxl_scale * 1e6: .2f}um, perfect={perfect_scale * 1e6: .2f}um,'
      f' resolution required={math.ceil(D_window / perfect_scale + 1)}')
beam_par = (0, 0, width0, lmbda)
# turbulence
r0 = r0_from_Cn2(Cn2=Cn2, k0=k0, dz=L_prop)
print(f'r0 parameter: {r0}, 2w0/r0={2 * width0 / r0}')
screens_num = screens_number(Cn2, k0, dz=L_prop)
print(f'Number of screen required: {screens_num}')
ryt = rytov(Cn2, k0, L_prop)
print(f'SR={np.exp(-ryt)} (Rytov)')
zR = (k0 * width0 ** 2)
print(f'Rayleigh Range (Zr) = {zR} (m)')
psh_par = (r0, res_xy_2D, pxl_scale, L0, l0)
psh_par_0 = (r0 * 1e100, res_xy_2D, pxl_scale, L0, l0 * 1e100)
# %% Beam
# Denis
C00 = 1.51
C01 = -5.06
C02 = 7.23
C03 = -2.04
C30 = -3.97
C_31 = 0
# z0 = 0
# field = (
# 		C00 * LG_simple(*mesh_2D, z0=z0, l=0, p=0, width=width0, k0=k0) +
# 		C01 * LG_simple(*mesh_2D, z0=z0, l=0, p=1, width=width0, k0=k0) +
# 		C02 * LG_simple(*mesh_2D, z0=z0, l=0, p=2, width=width0, k0=k0) +
# 		C03 * LG_simple(*mesh_2D, z0=z0, l=0, p=3, width=width0, k0=k0) +
# 		C30 * LG_simple(*mesh_2D, z0=z0, l=3, p=0, width=width0, k0=k0) +
# 		C_31 * LG_simple(*mesh_2D, z0=z0, l=-3, p=1, width=width0, k0=k0)
# )

# plot_field_both(field, extend=extend)

field = (
		C00 * LG_simple(*mesh_2D, z0=z0, l=0, p=0, width=width0, k0=k0) +
		C01 * LG_simple(*mesh_2D, z0=z0, l=0, p=1, width=width0, k0=k0) +
		C02 * LG_simple(*mesh_2D, z0=z0, l=0, p=2, width=width0, k0=k0) +
		C03 * LG_simple(*mesh_2D, z0=z0, l=0, p=3, width=width0, k0=k0) +
		C30 * LG_simple(*mesh_2D, z0=z0, l=3, p=0, width=width0, k0=k0) +
		C_31 * LG_simple(*mesh_2D, z0=z0, l=-3, p=1, width=width0, k0=k0)
)

plot_field_both(field, extend=extend)

field_z = propagation_ps(field, beam_par, psh_par, prop1, multiplier=[1], screens_num=1, seed=None)
plot_field_both(field_z, extend=extend)

field_z = propagation_ps(field_z, beam_par, psh_par_0, prop2, multiplier=[1], screens_num=1, seed=None)
plot_field_both(field_z, extend=extend)

field_z_crop = field_z[
   res_xy_2D // 2 - crop // 2: res_xy_2D // 2 + crop // 2,
   res_xy_2D // 2 - crop // 2: res_xy_2D // 2 + crop // 2,
]
plot_field_both(field_z_crop, extend=extend)
def beam_expander(field, beam_par, distance_both, steps_one):
	beam_3d = np.zeros((*np.shape(field), steps_one * 2 + 1))
	print(np.shape(beam_3d))
# print(np.shape(field_z_c, rop))
beam_expander(field_z_crop, 1, 1, 20)
exit()
dots_init_dict, dots_init = sing.get_singularities(np.angle(field_z), axesAll=True, returnDict=True)
pl.plotDots(dots_init, dots_init, color='black', show=True, size=10)
