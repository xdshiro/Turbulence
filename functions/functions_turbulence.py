import numpy as np
import matplotlib.pyplot as plt
from aotools.turbulence import phasescreen
from scipy.special import laguerre
from aotools import opticalpropagation
from aotools.turbulence.phasescreen import ft_phase_screen as ps
from aotools.turbulence.phasescreen import ft_sh_phase_screen as psh
import math
from scipy.special import assoc_laguerre


def plot_field_both(E, extend=None):
	fig, ax = plt.subplots(1, 2, figsize=(13, 6))
	im0 = ax[0].imshow(np.abs(E), extent=extend, cmap='magma')
	ax[0].set_title('|Amplitude|')
	fig.colorbar(im0, ax=ax[0], fraction=0.04, pad=0.02)
	
	im1 = ax[1].imshow(np.angle(E), extent=extend, cmap='hsv')
	ax[1].set_title('Phase')
	fig.colorbar(im1, ax=ax[1], fraction=0.04, pad=0.02)
	plt.tight_layout()
	plt.show()


def plot_field(E, extend=None):
	fig, ax = plt.subplots(1, 1, figsize=(7, 6))
	im0 = ax.imshow(E, extent=extend, cmap='jet')
	ax.set_title('Value')
	fig.colorbar(im0, ax=ax, label='Amplitude', fraction=0.04, pad=0.02)
	plt.tight_layout()
	plt.show()


def LG_simple(x, y, z=0, l=1, p=0, width=1, k0=1, x0=0, y0=0, z0=0):
	"""
    Classic LG beam
    :param x: X coordinate
    :param y: Y coordinate
    :param z: Z coordinate
    :param l: azimuthal index
    :param p: radial index
    :param width: beam waste
    :param k0: wave number
    :param x0: center of the beam in x
    :param y0: center of the beam in y
    :param z0: center of the beam in z
    :return: complex field
    """
	
	def rho(*r):
		return np.sqrt(sum(x ** 2 for x in r))
	
	def phi(x, y):
		return np.angle(x + 1j * y)
	
	def laguerre_polynomial(x, l, p):
		return assoc_laguerre(x, p, l)
	
	x = x - x0
	y = y - y0
	z = z - z0
	zR = (k0 * width ** 2)
	# zR = (k0 * width ** 2) / 2
	
	E = (np.sqrt(math.factorial(p) / (np.pi * math.factorial(np.abs(l) + p)))
	     * rho(x, y) ** np.abs(l) * np.exp(1j * l * phi(x, y))
	     / (width ** (np.abs(l) + 1) * (1 + 1j * z / zR) ** (np.abs(l) + 1))
	     * ((1 - 1j * z / zR) / (1 + 1j * z / zR)) ** p
	     * np.exp(-rho(x, y) ** 2 / (2 * width ** 2 * (1 + 1j * z / zR)))
	     * laguerre_polynomial(rho(x, y) ** 2 / (width ** 2 * (1 + z ** 2 / zR ** 2)), np.abs(l), p)
	     )
	
	return E


def r0_from_Cn2(Cn2, k0, dz):
	return (0.423 * k0 ** 2 * Cn2 * dz) ** (-3 / 5)


def Cn2_from_r0(r0, k0, dz):
	return r0 ** (-5 / 3) / (0.423 * k0 ** 2 * dz)


def SR_from_r0(r0, D):
	return np.exp(-(D / r0) ** (5 / 3))


def SR_from_Cn2(Cn2, k0, dz, D):
	r0 = r0_from_Cn2(Cn2, k0, dz)
	return np.exp(-(D / r0) ** (5 / 3))


def rytov(Cn2, k0, dz):
	return 1.23 * Cn2 * k0 ** (7 / 6) * dz ** (11 / 6)


def screens_number(Cn2, k0, dz):
	sigma_rytov = rytov(Cn2, k0, dz)
	return (10 * sigma_rytov) ** (6 / 11)


def arrays_from_mesh(mesh, indexing='ij'):
	"""
	Functions returns the tuple of x1Array, x2Array... of the mesh
	:param indexing: ij for a classic matrix
	:param mesh: no-sparse mesh, for 3D: [3][Nx, Ny, Nz]
	:return: for 3D: xArray, yArray, zArray
	"""
	xList = []
	if indexing == 'ij':
		for i, m in enumerate(mesh):
			row = [0] * len(np.shape(m))
			row[i] = slice(None, None)
			xList.append(m[tuple(row)])
	else:
		if len(np.shape(mesh[0])) == 2:
			for i, m in enumerate(mesh):
				row = [0] * len(np.shape(m))
				row[len(np.shape(m)) - 1 - i] = slice(None, None)
				xList.append(m[tuple(row)])
		elif len(np.shape(mesh[0])) == 3:
			indexing = [1, 0, 2]
			for i, m in enumerate(mesh):
				row = [0] * len(np.shape(m))
				row[indexing[i]] = slice(None, None)
				xList.append(m[tuple(row)])
		else:
			print("'xy' cannot be recreated for 4+ dimensions")
	
	xTuple = tuple(xList)
	return xTuple


def SR_gauss(mesh_2D, L_prop, beam_par, psh_par, epochs=100, screens_num=1, max_cut=False):
	_, _, width0, lmbda = beam_par
	k0 = 2 * np.pi / lmbda
	r0, N, pxl_scale, L0, l0 = psh_par
	xy_array, _ = arrays_from_mesh(mesh_2D)
	res_xy_2D = len(xy_array)
	xy_scale = xy_array[1] - xy_array[0]
	assert N == len(xy_array), 'Resolution of the beam isn"t equal to the phase screen N'
	# assert len(xy_array) % 2 == 0, 'Odd resolution of the beam'
	LG_00 = LG_simple(*mesh_2D, z=0, l=0, p=0, width=width0, k0=k0, x0=0, y0=0, z0=0)
	I0 = np.abs(LG_simple(x=0, y=0, z=L_prop,
	                      l=0, p=0, width=width0, k0=k0, x0=0, y0=0, z0=0)) ** 2
	I_avg_tot = 0
	
	Cn2 = Cn2_from_r0(r0, k0, L_prop)
	dL = L_prop / screens_num
	r0d = r0_from_Cn2(Cn2=Cn2, k0=k0, dz=dL)
	psh_par_dL = r0d, N, pxl_scale, L0, l0
	# E = beam_2D
	
	for i in range(epochs):
		E = LG_00
		for _ in range(screens_num):
			phase_screen_i = psh_wrap(psh_par_dL)
			E = opticalpropagation.angularSpectrum(
				E * np.exp(1j * phase_screen_i), lmbda, pxl_scale, pxl_scale, dL
			)
		current = np.abs(E[res_xy_2D // 2, res_xy_2D // 2]) ** 2
		# print(current, I0)
		if max_cut:
			I_avg_tot += min(current, I0)
		else:
			I_avg_tot += current
		if i == 1:
			plot_field_both(E, extend=None)
	I_avg = I_avg_tot / epochs
	
	SR = I_avg / I0
	print(f'SR={SR}')
	return SR




def SR_gauss_or(mesh_2D, L_prop, beam_par, psh_par, epochs=100, screens_num=1):
	l, p, width0, lmbda = beam_par
	k0 = 2 * np.pi / lmbda
	r0, N, pxl_scale, L0, l0 = psh_par
	xy_array, _ = arrays_from_mesh(mesh_2D)
	res_xy_2D = len(xy_array)
	xy_scale = xy_array[1] - xy_array[0]
	assert N == len(xy_array), 'Resolution of the beam isn"t equal to the phase screen N'
	assert len(xy_array) % 2 == 0, 'Odd resolution of the beam'
	LG_00 = LG_simple(*mesh_2D, z=0, l=l, p=p, width=width0, k0=k0, x0=0, y0=0, z0=0)
	I0 = np.abs(LG_simple(x=xy_scale / 2, y=xy_scale / 2, z=L_prop,
	                      l=l, p=p, width=width0, k0=k0, x0=0, y0=0, z0=0)) ** 2
	I_avg_tot = 0

	Cn2 = Cn2_from_r0(r0, k0, L_prop)
	dL = L_prop / screens_num
	r0d = r0_from_Cn2(Cn2=Cn2, k0=k0, dz=dL)
	psh_par_dL = r0d, N, pxl_scale, L0, l0
	# E = beam_2D
	
	for i in range(epochs):
		E = LG_00
		for _ in range(screens_num):
			phase_screen_i = psh_wrap(psh_par_dL)
			E = opticalpropagation.angularSpectrum(
				E * np.exp(1j * phase_screen_i), lmbda, pxl_scale, pxl_scale, dL
			)
		I_avg_tot += np.abs(E[res_xy_2D // 2, res_xy_2D // 2]) ** 2
		if i == 1:
			plot_field_both(E, extend=None)
	I_avg = I_avg_tot / epochs
	SR = I_avg / I0
	print(f'SR={SR}')
	return SR


def SR_gauss_old(mesh_2D, L_prop, beam_par, psh_par, epochs=100):
	l, p, width0, lmbda = beam_par
	k0 = 2 * np.pi / lmbda
	r0, N, delta, L0, l0 = psh_par
	xy_array, _ = arrays_from_mesh(mesh_2D)
	res_xy_2D = len(xy_array)
	xy_scale = xy_array[1] - xy_array[0]
	assert N == len(xy_array), 'Resolution of the beam isn"t equal to the phase screen N'
	assert len(xy_array) % 2 == 0, 'Odd resolution of the beam'
	LG_00 = LG_simple(*mesh_2D, z=0, l=l, p=p, width=width0, k0=k0, x0=0, y0=0, z0=0)
	I0 = np.abs(LG_simple(x=xy_scale / 2, y=xy_scale / 2, z=L_prop,
	                      l=l, p=p, width=width0, k0=k0, x0=0, y0=0, z0=0)) ** 2
	I_avg_tot = 0
	# E = beam_2D
	
	for i in range(epochs):
		psh_test = psh(r0, N, delta, L0, l0, FFT=None)
		E_current = opticalpropagation.angularSpectrum(
			LG_00 * np.exp(1j * psh_test), lmbda, xy_scale, xy_scale, L_prop
		)
		I_avg_tot += np.abs(E_current[res_xy_2D // 2, res_xy_2D // 2]) ** 2
	# if i == 1:
	# 	plot_field_both(E_current, extend=None)
	I_avg = I_avg_tot / epochs
	SR = I_avg / I0
	print(f'SR={SR}')
	return SR


def psh_wrap(psh_par, seed=None):
	r0, N, delta, L0, l0 = psh_par
	return psh(r0, N, delta, L0, l0, seed=seed)


def propagation_ps_simple(beam_2D, beam_par, psh_par, L_prop, screens_num=1):
	l, p, width0, lmbda = beam_par
	r0, res_xy_2D, pxl_scale, L0, l0 = psh_par
	k0 = 2 * np.pi / lmbda
	Cn2 = Cn2_from_r0(r0, k0, L_prop)
	dL = L_prop / screens_num
	r0 = r0_from_Cn2(Cn2=Cn2, k0=k0, dz=dL)
	psh_par_dL = r0, res_xy_2D, pxl_scale, L0, l0
	E = beam_2D
	for i in range(screens_num):
		phase_screen_i = psh_wrap(psh_par_dL)
		E = opticalpropagation.angularSpectrum(
			E * np.exp(1j * phase_screen_i), lmbda, pxl_scale, pxl_scale * 10, dL
		)
	return E

def propagation_ps(beam_2D, beam_par, psh_par, L_prop, screens_num=1, multiplier=1, seed=None):
	_, _, _, lmbda = beam_par
	r0, res_xy_2D, pxl_scale, L0, l0 = psh_par
	k0 = 2 * np.pi / lmbda
	Cn2 = Cn2_from_r0(r0, k0, L_prop)
	dL = L_prop / screens_num
	if type(multiplier) is not list:
		dMult = [multiplier ** (1 / screens_num)] * screens_num
	else:
		dMult = multiplier
	r0 = r0_from_Cn2(Cn2=Cn2, k0=k0, dz=dL)


	E = beam_2D
	current_scale = 1
	for i in range(screens_num):
		psh_par_dL = r0, res_xy_2D, pxl_scale / current_scale, L0, l0
		phase_screen_i = psh_wrap(psh_par_dL, seed=seed)
		E = opticalpropagation.angularSpectrum(
			E * np.exp(1j * phase_screen_i), lmbda,
			pxl_scale / current_scale, pxl_scale / (current_scale * dMult[i]), dL
		)
		# plot_field_both(E, extend=None)
		current_scale *= dMult[i]
	return E