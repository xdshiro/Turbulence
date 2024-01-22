from functions.functions_turbulence import *
import my_functions.singularities as sing
import my_functions.plotings as pl
from functions.all_knots_functions import *

# meshes and boundaries
x_lim_3D, y_lim_3D, z_lim_3D = (-7.0, 7.0), (-7.0, 7.0), (-2.0, 2.0)
res_x_3D, res_y_3D, res_z_3D = 120, 120, 50
x_3D, y_3D, z_3D = np.linspace(*x_lim_3D, res_x_3D), np.linspace(*y_lim_3D, res_y_3D), np.linspace(*z_lim_3D, res_z_3D)
mesh_3D = np.meshgrid(x_3D, y_3D, z_3D, indexing='ij')
mesh_2D = np.meshgrid(x_3D, y_3D, indexing='ij')
boundary_3D = [[0, 0, 0], [res_x_3D, res_y_3D, res_z_3D]]

res_z = 40  # resolution of the knot is res_z+1
crop = 300  # for the knot propagation
crop_3d = 120  # for the knot

# beam
lmbda = 633e-9  # wavelength
L_prop = 100  # propagation distance
knot_length = 100  # we need RALEYIG!!!!!!!!  # 1000 how far is detector from the knot center
width0 = 5e-3 / np.sqrt(2)  # beam width
xy_lim_2D = (-30.0e-3, 30.0e-3)  # window size to start with
res_xy_2D = 401  # resolution

# turbulence
# Cn2 = 1.35e-13  # turbulence strength  is basically in the range of 10−17–10−12 m−2/3
Cn2 = 3.21e-14
# Cn2 = 3.21e-40
# https://www.mdpi.com/2076-3417/11/22/10548
L0 = 9  # outer scale
l0 = 5e-3  # inner scale



z0 = knot_length * 1 + L_prop  # the source position
prop1 = L_prop  # z0-prop1 - detector position
prop2 = knot_length * 1  # z0-prop1-pro2 - knot center (assumed)


values = hopf_standard(mesh_3D, braid_func=braid)
w_real = 1.6
field_new_3D = field_knot_from_weights(values, mesh_2D, w_real, k0=1, x0=0, y0=0, z0=-1)

plot_field_both(field_new_3D)