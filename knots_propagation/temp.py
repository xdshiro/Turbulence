from functions.functions_turbulence import *
import my_functions.singularities as sing
import my_functions.plotings as pl
from functions.all_knots_functions import *

plot = 1
plot_3d = 1
print_coeff = 1
print_values = 1
seed = 1

# meshes and boundaries for getting a knot
x_lim_3D_knot, y_lim_3D_knot, z_lim_3D_knot = (-7.0, 7.0), (-7.0, 7.0), (-2.0, 2.0)
res_x_3D_knot, res_y_3D_knot, res_z_3D_knot = 256, 256, 1

# beam
lmbda = 633e-9  # wavelength
L_prop = 100  # propagation distance
knot_length = 100  # we need RALEYIG!!!!!!!!  # 1000 how far is detector from the knot center
width0 = 5e-3 / np.sqrt(2)  # beam width
xy_lim_2D_origin = (-30.0e-3, 30.0e-3)  # window size to start with
res_xy_2D_origin = 401  # resolution

res_z = 40  # resolution of the knot is res_z+1
crop = 250  # for the knot propagation
crop_3d = 120  # for the knot

screens_num1 = 100
multiplier1 = [1] * screens_num1
screens_num2 = 1
multiplier2 = [1] * screens_num2

# turbulence
# Cn2 = 1.35e-13  # turbulence strength  is basically in the range of 10−17–10−12 m−2/3
Cn2 = 3.21e-14
# Cn2 = 3.21e-40
# https://www.mdpi.com/2076-3417/11/22/10548
L0 = 9  # outer scale
l0 = 5e-3  # inner scale

# propagation parameters
z0 = knot_length * 1 + L_prop  # the source position
prop1 = L_prop  # z0-prop1 - detector position
prop2 = knot_length * 1  # z0-prop1-pro2 - knot center (assumed)

# extra values (physical)
k0 = 2 * np.pi / lmbda  # wave number
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
    print(f'SR={np.exp(-ryt)} (Rytov)')
zR = (k0 * width0 ** 2)
if print_values:
    print(f'Rayleigh Range (Zr) = {zR} (m)')
psh_par = (r0, res_xy_2D_origin, pxl_scale, L0, l0)
psh_par_0 = (r0 * 1e100, res_xy_2D_origin, pxl_scale, L0, l0 * 1e100)

# getting the knot
values = hopf_standard(mesh_3D_knot, braid_func=braid, plot=True)
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

# propagating in the turbulence prop1
field_after_turb = propagation_ps(
    field_before_prop, beam_par, psh_par, prop1, multiplier=multiplier1, screens_num=screens_num1, seed=seed
)
if plot:
    plot_field_both(field_after_turb, extend=extend)

# psh_par_0 = (r0 * 1e100, res_xy_2D, pxl_scale / 100, L0, l0 * 1e100)
# beam_par = (0, 0, width0 / 100, lmbda)
# field_z = propagation_ps(field_z, beam_par, psh_par_0, prop2 / 10000, multiplier=[1], screens_num=1, seed=None)
field_center = propagation_ps(
    field_after_turb, beam_par, psh_par_0, prop2, multiplier=multiplier2, screens_num=screens_num2, seed=seed
)
if plot:
    plot_field_both(field_center, extend=extend)

field_z_crop = field_center[
               res_xy_2D_origin // 2 - crop // 2: res_xy_2D_origin // 2 + crop // 2,
               res_xy_2D_origin // 2 - crop // 2: res_xy_2D_origin // 2 + crop // 2,
               ]

if plot:
    plot_field_both(field_z_crop, extend=extend_crop)

field_3d = beam_expander(field_z_crop, beam_par, psh_par_0, distance_both=knot_length, steps_one=res_z // 2)


x_cent_R, y_cent_R = find_center_of_intensity(field_z_crop)
x_cent, y_cent = int(x_cent_R), int(y_cent_R)
if print_values:
    print(f'Centers: {x_cent}, {y_cent} out of {crop}')

field_3d_crop = field_3d[
                x_cent - crop_3d // 2: x_cent + crop_3d // 2,
                y_cent - crop_3d // 2: y_cent + crop_3d // 2,
                :
                ]

if plot_3d:
    plot_field_both(field_3d_crop[:, :, 0], extend=extend_crop3d)
    plot_field_both(field_3d_crop[:, :, res_z // 2], extend=extend_crop3d)
    plot_field_both(field_3d_crop[:, :, res_z], extend=extend_crop3d)

dots_init_dict, dots_init = sing.get_singularities(np.angle(field_3d_crop), axesAll=False, returnDict=True)

dots_cut = cut_circle_dots(dots_init, crop_3d // 2, crop_3d // 2, crop_3d // 2)

if plot_3d:
    dots_bound = [
        [0, 0, 0],
        [crop_3d, crop_3d, res_z + 1],
    ]
    # pl.plotDots(dots_init, dots_init, color='black', show=True, size=10)
    pl.plotDots(dots_cut, dots_bound, color='black', show=True, size=10)
