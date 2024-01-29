from functions.functions_turbulence import *
import my_functions.singularities as sing
import my_functions.plotings as pl
import my_functions.functions_general as fg
import functions.dots_processing as dp
import functions.center_beam_search as cbs

plot = True
plot_3d = True

# %%  main parameters
lmbda = 633e-9  # wavelength
L_prop = 100  # propagation distance
knot_length = 100  # 1000 how far is detector from the knot center
width0 = 15e-3 / np.sqrt(2)  # beam width
width0 = 5e-3 / np.sqrt(2)  # beam width
xy_lim_2D = (-30.0e-3, 30.0e-3)  # window size to start with
res_xy_2D = 401  # resolution
# Cn2 = 1.35e-13  # turbulence strength  is basically in the range of 10−17–10−12 m−2/3
Cn2 = 3.21e-14
# Cn2 = 3.21e-40
# https://www.mdpi.com/2076-3417/11/22/10548
L0 = 9  # outer scale
l0 = 5e-3  # inner scale

res_z = 40  # resolution of the knot is res_z+1
crop = 300  # for the knot propagation
crop_3d = 120  # for the knot

z0 = knot_length * 1 + L_prop  # the source position
prop1 = L_prop  # z0-prop1 - detector position
prop2 = knot_length * 1  # z0-prop1-pro2 - knot center (assumed)
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


field = (
        C00 * LG_simple(*mesh_2D, z0=z0, l=0, p=0, width=width0, k0=k0) +
        C01 * LG_simple(*mesh_2D, z0=z0, l=0, p=1, width=width0, k0=k0) +
        C02 * LG_simple(*mesh_2D, z0=z0, l=0, p=2, width=width0, k0=k0) +
        C03 * LG_simple(*mesh_2D, z0=z0, l=0, p=3, width=width0, k0=k0) +
        C30 * LG_simple(*mesh_2D, z0=z0, l=3, p=0, width=width0, k0=k0) +
        C_31 * LG_simple(*mesh_2D, z0=z0, l=-3, p=1, width=width0, k0=k0)
)

if plot:
    plot_field_both(field, extend=extend)

field_z = propagation_ps(field, beam_par, psh_par, prop1, multiplier=[1], screens_num=1, seed=None)
if plot:
    plot_field_both(field_z, extend=extend)

# psh_par_0 = (r0 * 1e100, res_xy_2D, pxl_scale / 100, L0, l0 * 1e100)
# beam_par = (0, 0, width0 / 100, lmbda)
# field_z = propagation_ps(field_z, beam_par, psh_par_0, prop2 / 10000, multiplier=[1], screens_num=1, seed=None)
field_z = propagation_ps(field_z, beam_par, psh_par_0, prop2, multiplier=[1], screens_num=1, seed=None)
if plot:
    plot_field_both(field_z, extend=extend)

field_z_crop = field_z[
               res_xy_2D // 2 - crop // 2: res_xy_2D // 2 + crop // 2,
               res_xy_2D // 2 - crop // 2: res_xy_2D // 2 + crop // 2,
               ]

if plot:
    plot_field_both(field_z_crop, extend=extend)

# print(np.shape(field_z_c, rop))
field_3d = beam_expander(field_z_crop, beam_par, psh_par_0, distance_both=knot_length, steps_one=res_z // 2)
#
if plot_3d and 0:
    # plot_field_both(field_3d[:, :, 0], extend=extend)
    plot_field_both(field_3d[:, :, res_z // 2], extend=extend)
# plot_field_both(field_3d[:, :, res_z // 2 + 1], extend=extend)
# plot_field_both(field_3d[:, :, res_z // 2 + 2], extend=extend)
# plot_field_both(field_3d[:, :, res_z], extend=extend)

x_cent_R, y_cent_R = find_center_of_intensity(field_z_crop)
x_cent, y_cent = int(x_cent_R), int(y_cent_R)

# phase = np.angle(x_new + 1j * y_new)
# phase_mask = (phase > A) & (phase < B)
field_3d_crop = field_3d[
                x_cent - crop_3d // 2: x_cent + crop_3d // 2,
                y_cent - crop_3d // 2: y_cent + crop_3d // 2,
                :
                ]

if plot_3d:
    # plot_field_both(field_3d_crop[:, :, 0], extend=extend)
    plot_field_both(field_3d_crop[:, :, res_z // 2], extend=extend)
# plot_field_both(field_3d_crop[:, :, res_z // 2 + 1], extend=extend)
# plot_field_both(field_3d_crop[:, :, res_z // 2 + 2], extend=extend)
# plot_field_both(field_3d_crop[:, :, res_z], extend=extend)

dots_init_dict, dots_init = sing.get_singularities(np.angle(field_3d_crop), axesAll=False, returnDict=True)

dots_cut = cut_circle_dots(dots_init, crop_3d // 2, crop_3d // 2, crop_3d // 2)

if plot_3d:
    dots_bound = [
        [0, 0, 0],
        [crop_3d, crop_3d, res_z + 1],
    ]
    # pl.plotDots(dots_init, dots_init, color='black', show=True, size=10)
    pl.plotDots(dots_cut, dots_bound, color='black', show=True, size=10)

# def find_beam_waist(field, mesh=None):
# 	"""
# 	wrapper for the beam waste finder. More details in knots_ML.center_beam_search
# 	"""
# 	shape = np.shape(field)
# 	if mesh is None:
# 		mesh = fg.create_mesh_XY(xRes=shape[0], yRes=shape[1])
# 	width = cbs.find_width(field, mesh=mesh, width=shape[1] // 8, widthStep=1, print_steps=False)
# 	return width

# print(49 * pxl_scale)
# exit()
# width = find_beam_waist(
# 	field_z_crop,
# 	mesh=fg.create_mesh_XY(xRes=np.shape(field_z_crop)[0], yRes=np.shape(field_z_crop)[1])
# )

# print(width)

# def field_interpolation(field, mesh=None, resolution=(100, 100),
#                         xMinMax_frac=(1, 1), yMinMax_frac=(1, 1), fill_value=True):
#     """
#     Wrapper for the field interpolation fg.interpolation_complex
#     :param resolution: new field resolution
#     :param xMinMax_frac: new dimension for the field. (x_dim_old * frac)
#     :param yMinMax_frac: new dimension for the field. (y_dim_old * frac)
#     """
#     shape = np.shape(field)
#     if mesh is None:
#         mesh = fg.create_mesh_XY(xRes=shape[0], yRes=shape[1])
#     interpol_field = fg.interpolation_complex(field, mesh=mesh, fill_value=fill_value)
#     xMinMax = int(-shape[0] // 2 * xMinMax_frac[0]), int(shape[0] // 2 * xMinMax_frac[1])
#     yMinMax = int(-shape[1] // 2 * yMinMax_frac[0]), int(shape[1] // 2 * yMinMax_frac[1])
#     xyMesh_interpol = fg.create_mesh_XY(
#         xRes=resolution[0], yRes=resolution[1],
#         xMinMax=xMinMax, yMinMax=yMinMax)
#     return interpol_field(*xyMesh_interpol), xyMesh_interpol
#
# field_interpol, mesh_interpol = field_interpolation(
#         field_z_crop, mesh=mesh_init, resolution=resolution_iterpol_center,
#         xMinMax_frac=xMinMax_frac_center, yMinMax_frac=yMinMax_frac_center
#     )
