from functions.functions_turbulence import *
import my_functions.singularities as sing
import my_functions.plotings as pl
import functions.center_beam_search as cbs
import numpy as np


def u(x, y, z):
	numerator = x ** 2 + y ** 2 + z ** 2 - 1 + 2j * z
	denominator = x ** 2 + y ** 2 + z ** 2 + 1
	return numerator / denominator


def v(x, y, z):
	numerator = 2 * (x + 1j * y)
	denominator = x ** 2 + y ** 2 + z ** 2 + 1
	return numerator / denominator


def braid(x, y, z, angle=0, pow_cos=1, pow_sin=1, theta=0, a_cos=1, a_sin=1):
	def cos_v(x, y, z, power=1):
		return (v(x, y, z) ** power + np.conj(v(x, y, z)) ** power) / 2
	
	def sin_v(x, y, z, power=1):
		return (v(x, y, z) ** power - np.conj(v(x, y, z)) ** power) / 2j
	
	angle_3D = np.ones(np.shape(z)) * angle
	a_cos_3D = np.ones(np.shape(z)) * a_cos
	a_sin_3D = np.ones(np.shape(z)) * a_sin
	
	return u(x, y, z) * np.exp(1j * theta) - (
			cos_v(x, y, z, pow_cos) / a_cos_3D + 1j
			* sin_v(x, y, z, pow_sin) / a_sin_3D) * np.exp(1j * angle_3D)


def field_of_braids_trefoil_classic(mesh_3D, braid_func=braid):
	xyz_array = [
		(mesh_3D[0], mesh_3D[1], mesh_3D[2]),
		(mesh_3D[0], mesh_3D[1], mesh_3D[2])
	]
	# starting angle for each braid
	angle_array = np.array([0, 1. * np.pi])
	# powers in cos in sin
	pow_cos_array = [1.5, 1.5]
	pow_sin_array = [1.5, 1.5]
	# conjugating the braid (in "Milnor" space)
	conj_array = [0, 0]
	# moving x+iy (same as in the paper)
	theta_array = [0.0 * np.pi, 0 * np.pi]
	# braid scaling
	a_cos_array = [1, 1]
	a_sin_array = [1, 1]
	
	if theta_array is None:
		theta_array = [0] * len(angle_array)
	if a_cos_array is None:
		a_cos_array = [1] * len(angle_array)
	if a_sin_array is None:
		a_sin_array = [1] * len(angle_array)
	ans = 1
	for i, xyz in enumerate(xyz_array):
		if conj_array[i]:
			ans *= np.conjugate(braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
			                               a_cos_array[i], a_sin_array[i]))
		else:
			ans *= braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
			                  a_cos_array[i], a_sin_array[i])
	
	return ans


def field_of_braids_hopf_classic(mesh_3D, braid_func=braid):
	xyz_array = [
		(mesh_3D[0], mesh_3D[1], mesh_3D[2]),
		(mesh_3D[0], mesh_3D[1], mesh_3D[2])
	]
	# starting angle for each braid
	angle_array = np.array([0, 1. * np.pi])
	# powers in cos in sin
	pow_cos_array = [0.5, 0.5]
	pow_sin_array = [0.5, 0.5]
	# conjugating the braid (in "Milnor" space)
	conj_array = [0, 1]
	# moving x+iy (same as in the paper)
	theta_array = [0.0 * np.pi, 0 * np.pi]
	# braid scaling
	a_cos_array = [1, 1]
	a_sin_array = [1, 1]
	
	if theta_array is None:
		theta_array = [0] * len(angle_array)
	if a_cos_array is None:
		a_cos_array = [1] * len(angle_array)
	if a_sin_array is None:
		a_sin_array = [1] * len(angle_array)
	ans = 1
	for i, xyz in enumerate(xyz_array):
		if conj_array[i]:
			ans *= np.conjugate(braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
			                               a_cos_array[i], a_sin_array[i]))
		else:
			ans *= braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
			                  a_cos_array[i], a_sin_array[i])
	return ans


x_lim_3D, y_lim_3D, z_lim_3D = (-8.0, 8.0), (-8.0, 8.0), (-1.5, 1.5)
# x_lim_3D, y_lim_3D, z_lim_3D = (-4.0, 4.0), (-4.0, 4.0), (-1.5, 1.5)
res_x_3D, res_y_3D, res_z_3D = 180, 180, 40
# res_x_3D, res_y_3D, res_z_3D = 100, 100, 100
x_3D = np.linspace(*x_lim_3D, res_x_3D)
y_3D = np.linspace(*y_lim_3D, res_y_3D)
z_3D = np.linspace(*z_lim_3D, res_z_3D)
mesh_3D = np.meshgrid(x_3D, y_3D, z_3D, indexing='ij')
mesh_2D = np.meshgrid(x_3D, y_3D, indexing='ij')
R = np.sqrt(mesh_3D[0] ** 2 + mesh_3D[1] ** 2)
boundary_3D = [[0, 0, 0], [res_x_3D, res_y_3D, res_z_3D]]
w = 1.6
# field = field_of_braids_separate_trefoil(mesh_3D, braid_func=braid)
# field = field_of_braids_trefoil_classic(mesh_3D, braid_func=braid)
# field *= (1 + R ** 2) ** 2
# field *= LG_simple(*mesh_3D[:2], 0, l=0, p=0, width=w, k0=1, x0=0, y0=0, z0=0)
#
# plot_field_both(field[:, :, res_z_3D // 2])
# plt.show()

w = 1.2
# field = field_of_braids_separate_trefoil(mesh_3D, braid_func=braid)
field = field_of_braids_hopf_classic(mesh_3D, braid_func=braid)
field *= (1 + R ** 2) ** 3
field *= LG_simple(*mesh_3D[:2], 0, l=0, p=0, width=w, k0=1, x0=0, y0=0, z0=0)

plot_field_both(field[:, :, res_z_3D // 2])
plt.show()

dots_init_dict, dots_init = sing.get_singularities(np.angle(field), axesAll=True, returnDict=True)

pl.plotDots(dots_init, dots_init, color='black', show=True, size=10)

moments = {'p': (0, 5), 'l': (-5, 5)}

values = cbs.LG_spectrum(
	field[:, :, res_z_3D // 2], **moments, mesh=mesh_2D, plot=True, width=w, k0=1,
)

field_new_3D = np.zeros((res_x_3D, res_y_3D, res_z_3D)).astype(np.complex128)
total = 0
l_save = []
p_save = []
weight_save = []
modes_cutoff = 0.001
moment0 = moments['l'][0]

for l, p_array in enumerate(values):
	for p, value in enumerate(p_array):
		if abs(value) > modes_cutoff * abs(values).max():
			total += 1
			l_save.append(l + moment0)
			p_save.append(p)
			weight_save.append(value)
			# weights_important[f'{l + moment0}, {p}'] = value
			field_new_3D += value * LG_simple(*mesh_3D, l=l + moment0, p=p,
			                                  width=w, k0=1, x0=0, y0=0, z0=0)
weights_important = {'l': l_save, 'p': p_save, 'weight': weight_save}
field_new_3D = field_new_3D / np.abs(field_new_3D).max()

plot_field_both(field_new_3D[:, :, res_z_3D // 2])

dots_init_dict, dots_init = sing.get_singularities(np.angle(field_new_3D), axesAll=True, returnDict=True)
pl.plotDots(dots_init, dots_init, color='black', show=True, size=10)