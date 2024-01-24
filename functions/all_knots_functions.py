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


def hopf_standard_16(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
	mesh_3D_new1 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
	mesh_3D_new2 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
	xyz_array = [
		(mesh_3D_new1[0], mesh_3D_new1[1], mesh_3D_new1[2]),
		(mesh_3D_new2[0], mesh_3D_new2[1], mesh_3D_new2[2])
	]
	# starting angle for each braid
	angle_array = np.array([0, 1. * np.pi])
	# powers in cos in sin
	pow_cos_array = [1, 1]
	pow_sin_array = [1, 1]
	# conjugating the braid (in "Milnor" space)
	conj_array = [0, 0]
	# moving x+iy (same as in the paper)
	theta_array = [0.0 * np.pi, 0 * np.pi]
	# braid scaling
	a_cos_array = [1, 1]
	a_sin_array = [1, 1]

	ans = 1
	for i, xyz in enumerate(xyz_array):
		if conj_array[i]:
			ans *= np.conjugate(braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
			                               a_cos_array[i], a_sin_array[i]))
		else:
			ans *= braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
			                  a_cos_array[i], a_sin_array[i])
	R = np.sqrt(mesh_3D[0] ** 2 + mesh_3D[1] ** 2)
	ans *= (1 + R ** 2) ** 2
	w = 1.6
	ans *= LG_simple(*mesh_3D[:2], 0, l=0, p=0, width=w, k0=1, x0=0, y0=0, z0=0)
	
	moments = {'p': (0, 6), 'l': (-6, 6)}

	_, _, res_z_3D = np.shape(mesh_3D_new1[0])
	x_2D = mesh_3D[0][:, :, 0]
	y_2D = mesh_3D[1][:, :, 0]
	if plot:
		plot_field_both(ans[:, :, res_z_3D // 2])
	values = cbs.LG_spectrum(
		ans[:, :, res_z_3D // 2], **moments, mesh=(x_2D, y_2D), plot=True, width=w, k0=1,
	)
	l_save = []
	p_save = []
	weight_save = []
	moment0 = moments['l'][0]
	for l, p_array in enumerate(values):
		for p, value in enumerate(p_array):
			if abs(value) > modes_cutoff * abs(values).max():
				l_save.append(l + moment0)
				p_save.append(p)
				weight_save.append(value)
	weight_save /= np.sqrt(np.sum(np.array(weight_save) ** 2)) * 100
	weights_important = {'l': l_save, 'p': p_save, 'weight': weight_save}
	return weights_important


def hopf_standard_14(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
	mesh_3D_new1 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
	mesh_3D_new2 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
	xyz_array = [
		(mesh_3D_new1[0], mesh_3D_new1[1], mesh_3D_new1[2]),
		(mesh_3D_new2[0], mesh_3D_new2[1], mesh_3D_new2[2])
	]
	# starting angle for each braid
	angle_array = np.array([0, 1. * np.pi])
	# powers in cos in sin
	pow_cos_array = [1, 1]
	pow_sin_array = [1, 1]
	# conjugating the braid (in "Milnor" space)
	conj_array = [0, 0]
	# moving x+iy (same as in the paper)
	theta_array = [0.0 * np.pi, 0 * np.pi]
	# braid scaling
	a_cos_array = [1, 1]
	a_sin_array = [1, 1]

	ans = 1
	for i, xyz in enumerate(xyz_array):
		if conj_array[i]:
			ans *= np.conjugate(braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
										   a_cos_array[i], a_sin_array[i]))
		else:
			ans *= braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
							  a_cos_array[i], a_sin_array[i])
	R = np.sqrt(mesh_3D[0] ** 2 + mesh_3D[1] ** 2)
	ans *= (1 + R ** 2) ** 2
	w = 1.4
	ans *= LG_simple(*mesh_3D[:2], 0, l=0, p=0, width=w, k0=1, x0=0, y0=0, z0=0)

	moments = {'p': (0, 6), 'l': (-6, 6)}

	_, _, res_z_3D = np.shape(mesh_3D_new1[0])
	x_2D = mesh_3D[0][:, :, 0]
	y_2D = mesh_3D[1][:, :, 0]
	if plot:
		plot_field_both(ans[:, :, res_z_3D // 2])
	values = cbs.LG_spectrum(
		ans[:, :, res_z_3D // 2], **moments, mesh=(x_2D, y_2D), plot=True, width=w, k0=1,
	)
	l_save = []
	p_save = []
	weight_save = []
	moment0 = moments['l'][0]
	for l, p_array in enumerate(values):
		for p, value in enumerate(p_array):
			if abs(value) > modes_cutoff * abs(values).max():
				l_save.append(l + moment0)
				p_save.append(p)
				weight_save.append(value)
	weight_save /= np.sqrt(np.sum(np.array(weight_save) ** 2)) * 100
	weights_important = {'l': l_save, 'p': p_save, 'weight': weight_save}
	return weights_important

def hopf_standard_18(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
	mesh_3D_new1 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
	mesh_3D_new2 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
	xyz_array = [
		(mesh_3D_new1[0], mesh_3D_new1[1], mesh_3D_new1[2]),
		(mesh_3D_new2[0], mesh_3D_new2[1], mesh_3D_new2[2])
	]
	# starting angle for each braid
	angle_array = np.array([0, 1. * np.pi])
	# powers in cos in sin
	pow_cos_array = [1, 1]
	pow_sin_array = [1, 1]
	# conjugating the braid (in "Milnor" space)
	conj_array = [0, 0]
	# moving x+iy (same as in the paper)
	theta_array = [0.0 * np.pi, 0 * np.pi]
	# braid scaling
	a_cos_array = [1, 1]
	a_sin_array = [1, 1]

	ans = 1
	for i, xyz in enumerate(xyz_array):
		if conj_array[i]:
			ans *= np.conjugate(braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
										   a_cos_array[i], a_sin_array[i]))
		else:
			ans *= braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
							  a_cos_array[i], a_sin_array[i])
	R = np.sqrt(mesh_3D[0] ** 2 + mesh_3D[1] ** 2)
	ans *= (1 + R ** 2) ** 2
	w = 1.8
	ans *= LG_simple(*mesh_3D[:2], 0, l=0, p=0, width=w, k0=1, x0=0, y0=0, z0=0)

	moments = {'p': (0, 6), 'l': (-6, 6)}

	_, _, res_z_3D = np.shape(mesh_3D_new1[0])
	x_2D = mesh_3D[0][:, :, 0]
	y_2D = mesh_3D[1][:, :, 0]
	if plot:
		plot_field_both(ans[:, :, res_z_3D // 2])
	values = cbs.LG_spectrum(
		ans[:, :, res_z_3D // 2], **moments, mesh=(x_2D, y_2D), plot=True, width=w, k0=1,
	)
	l_save = []
	p_save = []
	weight_save = []
	moment0 = moments['l'][0]
	for l, p_array in enumerate(values):
		for p, value in enumerate(p_array):
			if abs(value) > modes_cutoff * abs(values).max():
				l_save.append(l + moment0)
				p_save.append(p)
				weight_save.append(value)
	weight_save /= np.sqrt(np.sum(np.array(weight_save) ** 2)) * 100
	weights_important = {'l': l_save, 'p': p_save, 'weight': weight_save}
	return weights_important
def field_knot_from_weights(values, mesh, w_real, k0=1, x0=0, y0=0, z0=0):
	res = np.shape(mesh[0])
	field_new = np.zeros(res).astype(np.complex128)
	
	for i in range(len(values['l'])):
		l, p, weight = values['l'][i], values['p'][i], values['weight'][i]
		field_new += weight * LG_simple(*mesh, l=l, p=p,
		                                   width=w_real, k0=k0, x0=x0, y0=y0, z0=z0)
	
	field_new = field_new / np.abs(field_new).max()
	return field_new



# dots_bound = [
# 	[0, 0, 0],
# 	[res_x_3D, res_y_3D, res_z_3D],
# ]
# dots_init_dict, dots_init = sing.get_singularities(np.angle(field_new_3D), axesAll=True, returnDict=True)
# pl.plotDots(dots_init, dots_bound, color='black', show=True, size=10)
