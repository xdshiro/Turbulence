"""
This Python script provides a comprehensive set of functions for manipulating and analyzing optical knots, with specific emphasis on generating and exploring various braid configurations and their corresponding topological structures. It is part of a larger framework for studying singularities and turbulence in optical fields, utilizing advanced mathematical and computational techniques. The script includes several key components:

## Import Statements

- External Libraries:
  - numpy: For numerical operations.
  - itertools: For creating combinations and products of lists.

- Custom Modules:
  - functions.functions_turbulence: Functions related to turbulence analysis.
  - my_functions.singularities: Functions for handling singularities.
  - my_functions.plotings: Functions for plotting.
  - functions.center_beam_search: Functions for beam search.

## Functions

### u(x, y, z)
Computes a specific complex function of the input coordinates.

### v(x, y, z)
Computes another complex function of the input coordinates.

### braid(x, y, z, angle=0, pow_cos=1, pow_sin=1, theta=0, a_cos=1, a_sin=1)
Generates a braid function based on the input coordinates and parameters.

### hopf_standard_16(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False)
Applies a specific braid function and performs operations to produce a standard Hopf link with 16-fold symmetry.

### hopf_standard_14(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False)
Similar to hopf_standard_16 but for a 14-fold symmetry Hopf link.

### hopf_standard_18(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False)
Similar to hopf_standard_16 but for an 18-fold symmetry Hopf link.

### hopf_30both(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False)
Generates a Hopf link with 30-fold symmetry by rotating the mesh grid.

### hopf_30oneZ(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False)
Generates a Hopf link with 30-fold symmetry focusing on the Z-axis rotation.

### hopf_optimized(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False)
Returns precomputed optimized weights for a specific Hopf link configuration.

### hopf_pm_03_z(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False)
Generates a Hopf link with specific rotations and adjustments in the Z-axis.

### hopf_4foil(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False)
Generates a 4-foil knot using specific braid function parameters.

### hopf_6foil(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False)
Generates a 6-foil knot using specific braid function parameters.

### hopf_stand4foil(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False)
Generates a standard 4-foil knot using specific braid function parameters.

### hopf_30oneX(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False)
Generates a Hopf link with 30-fold symmetry focusing on the X-axis rotation.

### hopf_15oneZ(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False)
Generates a Hopf link with 15-fold symmetry focusing on the Z-axis rotation.

### hopf_dennis(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False)
Returns precomputed optimized weights for a specific Forbes-type Hopf link configuration.

### lobe_remove(mesh, angle1, angle2, rot_x, rot_y, rot_z)
Removes a lobe from the mesh grid within specified angle ranges and rotations.

### lobe_smaller(mesh, angle1, angle2, rot_x, rot_y, rot_z)
Reduces the size of a lobe in the mesh grid within specified angle ranges and rotations.

### unknot_6(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False)
Generates an unknot with 6-fold symmetry.

### unknot_4(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False)
Generates an unknot with 4-fold symmetry.

### unknot_4_any(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False, angle_size=(2, 2, 2, 2))
Generates a 4-fold unknot with customizable angle sizes.

### borromean(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False)
Generates a Borromean ring configuration.

### loops_x2(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False)
Generates a double loop configuration with specified rotations and shifts.

### whitehead(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False)
Generates a Whitehead link configuration with specific lobe rotations.

## Helper Functions

### rotate_meshgrid(*args)
Rotates the given mesh grid by specified angles.

### LG_simple(*args)
Generates a Laguerre-Gaussian beam profile with specified parameters.

### plot_field_both(*args)
Plots the given field data using both 2D and 3D visualizations.

### cbs.LG_spectrum(*args)
Calculates the spectrum of the Laguerre-Gaussian beam for the given field.

This script is essential for researchers working on the topological manipulation of optical fields, providing a versatile toolkit for creating and analyzing complex knot structures.
"""
import matplotlib.pyplot as plt

from functions.functions_turbulence import *
import my_functions.singularities as sing
import my_functions.plotings as pl
import functions.center_beam_search as cbs
import numpy as np

from itertools import combinations, product


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


def hopf_standard_16(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False, cmap='jet'):
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
        ans[:, :, res_z_3D // 2], **moments, mesh=(x_2D, y_2D), plot=True, width=w, k0=1, cmap=cmap
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


def hopf_30both(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    mesh_3D_new1 = rotate_meshgrid(*mesh_3D, np.radians(30), np.radians(00), np.radians(0))
    mesh_3D_new2 = rotate_meshgrid(*mesh_3D, np.radians(-30), np.radians(00), np.radians(0))
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


def hopf_30oneZ(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    mesh_3D_new1 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(30))
    mesh_3D_new2 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    xyz_array = [
        (mesh_3D_new1[0], mesh_3D_new1[1], mesh_3D_new1[2] - 0.3),
        (mesh_3D_new2[0], mesh_3D_new2[1], mesh_3D_new2[2] + 0.3)
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


def hopf_optimized(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    l_save = [0, 0, 0, 2]
    p_save = [0, 1, 2, 0]
    weight_save = [2.96, -6.23, 4.75, -5.49]
    weight_save /= np.sqrt(np.sum(np.array(weight_save) ** 2)) * 100
    weights_important = {'l': l_save, 'p': p_save, 'weight': weight_save}
    return weights_important


def hopf_pm_03_z(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    mesh_3D_new1 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    mesh_3D_new2 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    xyz_array = [
        (mesh_3D_new1[0], mesh_3D_new1[1], mesh_3D_new1[2] - 0.3),
        (mesh_3D_new2[0], mesh_3D_new2[1], mesh_3D_new2[2] + 0.3)
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


def hopf_4foil(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    mesh_3D_new1 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    mesh_3D_new2 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    xyz_array = [
        (mesh_3D_new1[0], mesh_3D_new1[1], mesh_3D_new1[2]),
        (mesh_3D_new2[0], mesh_3D_new2[1], mesh_3D_new2[2])
    ]
    # starting angle for each braid
    angle_array = np.array([0, 1. * np.pi])
    # powers in cos in sin
    pow_cos_array = [2, 2]
    pow_sin_array = [2, 2]
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
    ans *= (1 + R ** 2) ** 4
    w = 1.0
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


def hopf_6foil(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    mesh_3D_new1 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    mesh_3D_new2 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    xyz_array = [
        (mesh_3D_new1[0], mesh_3D_new1[1], mesh_3D_new1[2]),
        (mesh_3D_new2[0], mesh_3D_new2[1], mesh_3D_new2[2])
    ]
    # starting angle for each braid
    angle_array = np.array([0, 1. * np.pi])
    # powers in cos in sin
    pow_cos_array = [3, 3]
    pow_sin_array = [3, 3]
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
    ans *= (1 + R ** 2) ** 6
    w = 0.75
    ans *= LG_simple(*mesh_3D[:2], 0, l=0, p=0, width=w, k0=1, x0=0, y0=0, z0=0)

    moments = {'p': (0, 7), 'l': (-7, 7)}

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


def hopf_stand4foil(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    mesh_3D_new1 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    mesh_3D_new2 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    xyz_array = [
        (mesh_3D_new1[0], mesh_3D_new1[1], mesh_3D_new1[2]),
        (mesh_3D_new2[0], mesh_3D_new2[1], mesh_3D_new2[2])
    ]
    # starting angle for each braid
    angle_array = np.array([0, 1. * np.pi])
    # powers in cos in sin
    pow_cos_array = [1, 2]
    pow_sin_array = [1, 2]
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
    ans *= (1 + R ** 2) ** 3
    w = 1.3
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


def hopf_30oneX(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    mesh_3D_new1 = rotate_meshgrid(*mesh_3D, np.radians(20), np.radians(00), np.radians(0))
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


def hopf_15oneZ(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False, cmap='jet'):
    mesh_3D_new1 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(15))
    mesh_3D_new2 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    xyz_array = [
        (mesh_3D_new1[0], mesh_3D_new1[1], mesh_3D_new1[2] - 0.3),
        (mesh_3D_new2[0], mesh_3D_new2[1], mesh_3D_new2[2] + 0.3)
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
        ans[:, :, res_z_3D // 2], **moments, mesh=(x_2D, y_2D), plot=True, width=w, k0=1, cmap=cmap
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


def hopf_dennis(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    l_save = [0, 0, 0, 2]
    p_save = [0, 1, 2, 0]
    weight_save = [2.63, -6.32, 4.21, -5.95]
    weight_save /= np.sqrt(np.sum(np.array(weight_save) ** 2)) * 100
    weights_important = {'l': l_save, 'p': p_save, 'weight': weight_save}
    return weights_important


def lobe_remove(mesh, angle1, angle2, rot_x, rot_y, rot_z):
    A, B = angle1, angle2
    phase = np.angle(mesh[0] + 1j * mesh[1])
    phase_mask = (phase > A) & (phase <= B)

    mesh2_flat = rotate_meshgrid(
        mesh[0][phase_mask],
        mesh[1][phase_mask],
        mesh[2][phase_mask],
        np.radians(rot_x), np.radians(rot_y), np.radians(rot_z)
    )
    mesh2 = np.copy(mesh)
    mesh2[0][phase_mask] = mesh2_flat[0] * 100
    mesh2[1][phase_mask] = mesh2_flat[1] * 100
    mesh2[2][phase_mask] = mesh2_flat[2]
    return mesh2


def lobe_smaller(mesh, angle1, angle2, rot_x, rot_y, rot_z):
    A, B = angle1, angle2
    phase = np.angle(mesh[0] + 1j * mesh[1])
    phase_mask = (phase > A) & (phase <= B)

    mesh2_flat = rotate_meshgrid(
        mesh[0][phase_mask],
        mesh[1][phase_mask],
        mesh[2][phase_mask],
        np.radians(rot_x), np.radians(rot_y), np.radians(rot_z)
    )
    mesh2 = np.copy(mesh)
    mesh2[0][phase_mask] = mesh2_flat[0] * 1.5
    mesh2[1][phase_mask] = mesh2_flat[1] * 1.5
    mesh2[2][phase_mask] = mesh2_flat[2]
    return mesh2


def unknot_6(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    mesh_3D_new1 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))

    A, B = -np.pi / 6, np.pi / 6
    mesh_3D_new2 = lobe_smaller(mesh_3D_new1, A, B, rot_x=0, rot_y=0, rot_z=0)
    A, B = 3 * np.pi / 6, 5 * np.pi / 6
    mesh_3D_new2 = lobe_remove(mesh_3D_new2, A, B, rot_x=0, rot_y=0, rot_z=0)
    # A, B = -3 * np.pi / 6, -1 * np.pi / 6
    # mesh_3D_new2 = rotate_part(mesh_3D_new2, A, B, rot_x=0, rot_y=0, rot_z=0)
    xyz_array = [
        (mesh_3D_new2[0], mesh_3D_new2[1], mesh_3D_new2[2]),
    ]
    xyz_array = [
        (mesh_3D[0], mesh_3D[1], mesh_3D[2]),
    ]
    # starting angle for each braid
    angle_array = np.array([0])
    # powers in cos in sin
    power = 6
    pow_cos_array = [power]
    pow_sin_array = [power]
    # conjugating the braid (in "Milnor" space)
    conj_array = [0]
    # moving x+iy (same as in the paper)
    theta_array = [0.0 * np.pi, 0 * np.pi]
    # braid scaling
    a_cos_array = [1]
    a_sin_array = [1]

    ans = 1
    for i, xyz in enumerate(xyz_array):
        if conj_array[i]:
            ans *= np.conjugate(braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                                           a_cos_array[i], a_sin_array[i]))
        else:
            ans *= braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                              a_cos_array[i], a_sin_array[i])
    R = np.sqrt(mesh_3D[0] ** 2 + mesh_3D[1] ** 2)
    ans *= (1 + R ** 2) ** power
    ws = {
        0: 3,
        1: 2.6,
        # 2: 1.6,
        2: 2.6 ** (1 / 2),
        3: 1.2,
        4: 0.9,
        5: 0.75,
        6: 0.65,
    }
    w = ws[power]
    ans *= LG_simple(*mesh_3D[:2], 0, l=0, p=0, width=w, k0=1, x0=0, y0=0, z0=0)

    moments = {'p': (0, 10), 'l': (-10, 10)}

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


def unknot_4(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    mesh_3D_new1 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))

    A, B = -np.pi / 4, np.pi / 4
    mesh_3D_new2 = lobe_smaller(mesh_3D_new1, A, B, rot_x=0, rot_y=0, rot_z=0)
    A, B = -4 * np.pi / 4, -3 * np.pi / 4
    mesh_3D_new2 = lobe_remove(mesh_3D_new2, A, B, rot_x=0, rot_y=0, rot_z=0)
    A, B = 3 * np.pi / 4, 4 * np.pi / 4
    mesh_3D_new2 = lobe_remove(mesh_3D_new2, A, B, rot_x=0, rot_y=0, rot_z=0)
    # A, B = -3 * np.pi / 6, -1 * np.pi / 6
    # mesh_3D_new2 = rotate_part(mesh_3D_new2, A, B, rot_x=0, rot_y=0, rot_z=0)
    xyz_array = [
        (mesh_3D_new2[0], mesh_3D_new2[1], mesh_3D_new2[2]),
    ]
    # starting angle for each braid
    angle_array = np.array([0])
    # powers in cos in sin
    power = 4
    pow_cos_array = [power]
    pow_sin_array = [power]
    # conjugating the braid (in "Milnor" space)
    conj_array = [0]
    # moving x+iy (same as in the paper)
    theta_array = [0.0 * np.pi, 0 * np.pi]
    # braid scaling
    a_cos_array = [1]
    a_sin_array = [1]

    ans = 1
    for i, xyz in enumerate(xyz_array):
        if conj_array[i]:
            ans *= np.conjugate(braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                                           a_cos_array[i], a_sin_array[i]))
        else:
            ans *= braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                              a_cos_array[i], a_sin_array[i])
    R = np.sqrt(mesh_3D[0] ** 2 + mesh_3D[1] ** 2)
    ans *= (1 + R ** 2) ** power
    ws = {
        0: 3,
        1: 2.6,
        # 2: 1.6,
        2: 2.6 ** (1 / 2),
        3: 1.2,
        4: 0.9,
        5: 0.75,
        6: 0.65,
    }
    w = ws[power]
    ans *= LG_simple(*mesh_3D[:2], 0, l=0, p=0, width=w, k0=1, x0=0, y0=0, z0=0)

    moments = {'p': (0, 10), 'l': (-10, 10)}

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



def borromean(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    mesh_3D_new1 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    mesh_3D_new2 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(30))
    xyz_array = [
        (mesh_3D_new1[0], mesh_3D_new1[1], mesh_3D_new1[2]),
        (mesh_3D_new2[0], mesh_3D_new2[1], mesh_3D_new2[2]),
        (mesh_3D_new1[0], mesh_3D_new1[1], mesh_3D_new1[2]),
    ]
    # starting angle for each braid
    angle_array = np.array([0, 2. * np.pi / 3, 4. * np.pi / 3])
    # powers in cos in sin
    pow_cos_array = [1, 1, 1]
    pow_sin_array = [1, 1, 1]
    # conjugating the braid (in "Milnor" space)
    conj_array = [0, 0, 0]
    # moving x+iy (same as in the paper)
    theta_array = [0.0 * np.pi, 0 * np.pi, 0]
    # braid scaling
    a_cos_array = [1, 1, 1]
    a_sin_array = [1, 1, 1]

    ans = 1
    for i, xyz in enumerate(xyz_array):
        if conj_array[i]:
            ans *= np.conjugate(braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                                           a_cos_array[i], a_sin_array[i]))
        else:
            ans *= braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                              a_cos_array[i], a_sin_array[i])
    R = np.sqrt(mesh_3D[0] ** 2 + mesh_3D[1] ** 2)
    ans *= (1 + R ** 2) ** 3
    w = 1.4
    ans *= LG_simple(*mesh_3D[:2], 0, l=0, p=0, width=w, k0=1, x0=0, y0=0, z0=0)

    moments = {'p': (0, 7), 'l': (-7, 7)}

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


def loops_x2(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    def lobe_rotate(mesh, angle1, angle2, rot_x=0, rot_y=0, rot_z=0, xshift=0, yshift=0, zshift=0):
        A, B = angle1, angle2
        phase = np.angle(mesh[0] + 1j * mesh[1])
        phase_mask = (phase > A) & (phase <= B)

        mesh2_flat = rotate_meshgrid(
            mesh[0][phase_mask] + xshift,
            mesh[1][phase_mask] + yshift,
            mesh[2][phase_mask] + xshift,
            np.radians(rot_x), np.radians(rot_y), np.radians(rot_z)
        )
        mesh2 = np.copy(mesh)
        mesh2[0][phase_mask] = mesh2_flat[0]
        mesh2[1][phase_mask] = mesh2_flat[1]
        mesh2[2][phase_mask] = mesh2_flat[2]
        return mesh2

    mesh_3D_1 = rotate_meshgrid(*mesh_3D, np.radians(-40), np.radians(00), np.radians(0))
    mesh_3D_2 = rotate_meshgrid(*mesh_3D, np.radians(40), np.radians(00), np.radians(0))
    mesh_3D_3 = rotate_meshgrid(*mesh_3D, np.radians(0), np.radians(0), np.radians(0))
    ang = np.pi / 6
    angle1, angle2 = -np.pi / 2 - ang, -np.pi / 2 + ang
    angle1, angle2 = -np.pi / 2, np.pi / 2
    mesh_3D_new1 = lobe_rotate(mesh_3D_1, angle1, angle2, yshift=0, rot_y=50)
    mesh_3D_new2 = lobe_rotate(mesh_3D_2, angle1, angle2, yshift=0, rot_y=-50)
    # angle1, angle2 = -np.pi, -np.pi / 2
    # mesh_3D_new1 = lobe_rotate(mesh_3D_new1, angle1, angle2, yshift=-0.7)
    # mesh_3D_new2 = lobe_rotate(mesh_3D_new2, angle1, angle2, yshift=-0.7)
    # angle1, angle2 = np.pi / 2, np.pi
    # mesh_3D_new1 = lobe_rotate(mesh_3D_new1, angle1, angle2, yshift=-0.7)
    # mesh_3D_new2 = lobe_rotate(mesh_3D_new2, angle1, angle2, yshift=-0.7)
    angle1, angle2 = -np.pi / 2 - ang, -np.pi / 2 + ang
    mesh_3D_new3 = lobe_rotate(mesh_3D_3, angle1, angle2, 0, 0, 0)
    alpha = 1.
    shift = 0.3
    xyz_array = [
        (mesh_3D_new1[0] + shift, mesh_3D_new1[1], mesh_3D_new1[2]),
        (mesh_3D_new2[0] * alpha - shift, mesh_3D_new2[1] * alpha, mesh_3D_new2[2] * alpha),
        #  (mesh_3D_new3[0], mesh_3D_new3[1], mesh_3D_new3[2]),

    ]
    shift1 = 0.0
    shift2 = 0.0
    shift3 = 0.0
    shift_scale = np.cos(np.pi / 6) * 1

    # starting angle for each braid
    angle_array = np.array([0, -np.pi, 1. * np.pi / 2])
    # angle_array = np.array([- np.pi / 6, np.pi + np.pi / 6 , 1. * np.pi / 2])
    # powers in cos in sin
    pow_cos_array = [1, 1, 1]
    pow_sin_array = [1, 1, 1]
    # conjugating the braid (in "Milnor" space)
    conj_array = [0, 0, 0]
    # moving x+iy (same as in the paper)
    theta_array = [0.0 * np.pi, 0 * np.pi, 0]
    # braid scaling
    a_cos_array = [1, 1, 1]
    a_sin_array = [1, 1, 1]

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
    w = 1.5
    ans *= LG_simple(*mesh_3D[:2], 0, l=0, p=0, width=w, k0=1, x0=0, y0=0, z0=0)

    moments = {'p': (0, 7), 'l': (-7, 7)}

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


def whitehead(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    def lobe_rotate(mesh, angle1, angle2, rot_x=0, rot_y=0, rot_z=0, xshift=0, yshift=0, zshift=0):
        A, B = angle1, angle2
        phase = np.angle(mesh[0] + 1j * mesh[1])
        phase_mask = (phase > A) & (phase <= B)

        mesh2_flat = rotate_meshgrid(
            mesh[0][phase_mask] + xshift,
            mesh[1][phase_mask] + yshift,
            mesh[2][phase_mask] + xshift,
            np.radians(rot_x), np.radians(rot_y), np.radians(rot_z)
        )
        mesh2 = np.copy(mesh)
        mesh2[0][phase_mask] = mesh2_flat[0]
        mesh2[1][phase_mask] = mesh2_flat[1]
        mesh2[2][phase_mask] = mesh2_flat[2]
        return mesh2

    mesh_3D_1 = rotate_meshgrid(*mesh_3D, np.radians(0), np.radians(00), np.radians(0))
    mesh_3D_2 = rotate_meshgrid(*mesh_3D, np.radians(0), np.radians(00), np.radians(0))
    mesh_3D_3 = rotate_meshgrid(*mesh_3D, np.radians(0), np.radians(0), np.radians(0))
    ang = np.pi / 6
    angle1, angle2 = -np.pi / 2 - ang, -np.pi / 2 + ang
    angle1, angle2 = -np.pi / 2, np.pi / 2
    mesh_3D_new1 = lobe_rotate(mesh_3D_1, angle1, angle2)
    mesh_3D_new2 = lobe_rotate(mesh_3D_2, angle1, angle2)
    # angle1, angle2 = -np.pi, -np.pi / 2
    # mesh_3D_new1 = lobe_rotate(mesh_3D_new1, angle1, angle2, yshift=-0.7)
    # mesh_3D_new2 = lobe_rotate(mesh_3D_new2, angle1, angle2, yshift=-0.7)
    # angle1, angle2 = np.pi / 2, np.pi
    # mesh_3D_new1 = lobe_rotate(mesh_3D_new1, angle1, angle2, yshift=-0.7)
    # mesh_3D_new2 = lobe_rotate(mesh_3D_new2, angle1, angle2, yshift=-0.7)
    angle1, angle2 = -np.pi / 2 - ang, -np.pi / 2 + ang
    mesh_3D_new3 = lobe_rotate(mesh_3D_3, angle1, angle2, 0, 0, 0)
    alpha = 1.
    shift = 0.0
    xyz_array = [
        (mesh_3D_new1[0] + shift, mesh_3D_new1[1], mesh_3D_new1[2]),
        (mesh_3D_new2[0] * alpha - shift, mesh_3D_new2[1] * alpha, mesh_3D_new2[2] * alpha),
        #  (mesh_3D_new3[0], mesh_3D_new3[1], mesh_3D_new3[2]),

    ]
    shift1 = 0.0
    shift2 = 0.0
    shift3 = 0.0
    shift_scale = np.cos(np.pi / 6) * 1

    # starting angle for each braid
    angle_array = np.array([0, 2 * np.pi / 2, 1. * np.pi / 2])
    # angle_array = np.array([- np.pi / 6, np.pi + np.pi / 6 , 1. * np.pi / 2])
    # powers in cos in sin
    pow_cos_array = [1, 1, 1]
    pow_sin_array = [1, 1, 1]
    # conjugating the braid (in "Milnor" space)
    conj_array = [0, 0, 0]
    # moving x+iy (same as in the paper)
    theta_array = [0.0 * np.pi, 0 * np.pi, 0]
    # braid scaling
    a_cos_array = [1, 1, 1]
    a_sin_array = [1, 1, 1]

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
    w = 1.3
    ans *= LG_simple(*mesh_3D[:2], 0, l=0, p=0, width=w, k0=1, x0=0, y0=0, z0=0)

    moments = {'p': (0, 7), 'l': (-7, 7)}

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


def whitehead2(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    def lobe_rotate(mesh, angle1, angle2, rot_x, rot_y, rot_z):
        A, B = angle1, angle2
        phase = np.angle(mesh[0] + 1j * mesh[1])
        phase_mask = (phase > A) & (phase <= B)

        mesh2_flat = rotate_meshgrid(
            mesh[0][phase_mask],
            mesh[1][phase_mask],
            mesh[2][phase_mask],
            np.radians(rot_x), np.radians(rot_y), np.radians(rot_z)
        )
        mesh2 = np.copy(mesh)
        mesh2[0][phase_mask] = mesh2_flat[0]
        mesh2[1][phase_mask] = mesh2_flat[1]
        mesh2[2][phase_mask] = mesh2_flat[2]
        return mesh2

    mesh_3D_1 = rotate_meshgrid(*mesh_3D, np.radians(0), np.radians(00), np.radians(0))
    mesh_3D_2 = rotate_meshgrid(*mesh_3D, np.radians(0), np.radians(00), np.radians(0))
    mesh_3D_3 = rotate_meshgrid(*mesh_3D, np.radians(0), np.radians(-45), np.radians(0))
    mesh_3D_3 = rotate_meshgrid(*mesh_3D, np.radians(0), np.radians(0), np.radians(0))
    ang = np.pi / 6
    angle1, angle2 = -np.pi / 2 - ang, -np.pi / 2 + ang
    angle1, angle2 = -np.pi / 2 - ang, -np.pi / 2 + ang
    mesh_3D_new1 = lobe_rotate(mesh_3D_1, angle1, angle2, 0, 0, 0)
    mesh_3D_new2 = lobe_rotate(mesh_3D_2, angle1, angle2, 0, 0, 0)
    angle1, angle2 = -np.pi / 2 - ang, -np.pi / 2 + ang
    mesh_3D_new3 = lobe_rotate(mesh_3D_3, angle1, angle2, 0, 0, 0)
    alpha = 1.
    xyz_array = [
        (mesh_3D_new1[0], mesh_3D_new1[1], mesh_3D_new1[2]),
        (mesh_3D_new2[0] * alpha + 0, mesh_3D_new2[1] * alpha, mesh_3D_new2[2] * alpha),
        (mesh_3D_new3[0], mesh_3D_new3[1] - 0.5, mesh_3D_new3[2]),

    ]
    shift1 = 0.0
    shift2 = 0.0
    shift3 = 0.0
    shift_scale = np.cos(np.pi / 6) * 1
    xyz_array = [
        (mesh_3D_new1[0] - shift1 * shift_scale, mesh_3D_new1[1] - shift1 * shift_scale, mesh_3D_new1[2]),
        (mesh_3D_new2[0] * alpha + shift2 * shift_scale, mesh_3D_new2[1] * alpha - shift2 * shift_scale,
         mesh_3D_new2[2] * alpha),
        (mesh_3D_new3[0], mesh_3D_new3[1] + shift3, mesh_3D_new3[2]),

    ]
    # starting angle for each braid
    angle_array = np.array([0, np.pi, 1. * np.pi / 2])
    angle_array = np.array([- np.pi / 6, np.pi + np.pi / 6, 1. * np.pi / 2])
    # powers in cos in sin
    pow_cos_array = [1, 1, 1]
    pow_sin_array = [1, 1, 1]
    # conjugating the braid (in "Milnor" space)
    conj_array = [0, 0, 0]
    # moving x+iy (same as in the paper)
    theta_array = [0.0 * np.pi, 0 * np.pi, 0]
    # braid scaling
    a_cos_array = [1, 1, 1]
    a_sin_array = [1, 1, 1]

    ans = 2
    for i, xyz in enumerate(xyz_array):
        if conj_array[i]:
            ans *= np.conjugate(braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                                           a_cos_array[i], a_sin_array[i]))
        else:
            ans *= braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                              a_cos_array[i], a_sin_array[i])
    R = np.sqrt(mesh_3D[0] ** 2 + mesh_3D[1] ** 2)
    ans *= (1 + R ** 2) ** 3
    w = 1.6
    ans *= LG_simple(*mesh_3D[:2], 0, l=0, p=0, width=w, k0=1, x0=0, y0=0, z0=0)

    moments = {'p': (0, 7), 'l': (-7, 7)}

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


def hopf_new0(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
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


def trefoil_dennis(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    l_save = [0, 0, 0, 0, 3]
    p_save = [0, 1, 2, 3, 0]
    weight_save = [1.51, -5.06, 7.23, -2.03, -3.97]
    weight_save /= np.sqrt(np.sum(np.array(weight_save) ** 2)) * 100
    weights_important = {'l': l_save, 'p': p_save, 'weight': weight_save}
    return weights_important


def trefoil_optimized_math_many(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    l_save = [-3, -3, -3, 0, 0, 0, 0, 3, 3, 3, 3]
    p_save = [0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3]
    weight_save = [(0.00021092227580502785+7.920886118023887e-07j), (-0.0008067955939880228+3.607965773547045e-06j), (0.00012245355229622952+1.0489244513922499e-07j), (0.0011924249760862221+1.1372720865198612e-05j), (-0.002822503524696713+8.535015090975148e-06j), (0.0074027513552253985+5.475152609562587e-06j), (-0.0037869189890120283+8.990311302510444e-06j), (-0.0043335243263204586+8.720849197446247e-07j), (0.00013486021497321202+3.7649803780475746e-06j), (-0.0002255962470142481+3.4852313165379194e-07j), (-9.910118914013346e-05+1.5151969296215156e-08j)]
    weight_save /= np.sqrt(np.sum(np.array(weight_save) ** 2)) * 100
    weights_important = {'l': l_save, 'p': p_save, 'weight': weight_save}
    return weights_important

def trefoil_optimized_math_many_095(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    l_save = [-3, -3, -3, 0, 0, 0, 0, 3, 3, 3, 3]
    p_save = [0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3]
    weight_save = [(0.0003480344230302087-2.24808914259125e-08j), (-0.0008202669878716706+3.440862545658389e-06j), (0.00010265094423308904+2.31922738071223e-07j), (0.0011967299943849222+7.739735031831317e-06j), (-0.0038439481283424996+6.262024157567451e-06j), (0.007992793393037656+2.6957311985005077e-06j), (-0.0038705109086829936+6.2293722796143055e-06j), (-0.0038065491189479697+2.3047912587806214e-08j), (0.00020229977021637684+3.5303977611434e-06j), (-0.00019990398151559197+3.711187821973153e-07j), (-8.519625649010964e-05+1.3228967779423537e-08j)]
    weight_save /= np.sqrt(np.sum(np.array(weight_save) ** 2)) * 100
    weights_important = {'l': l_save, 'p': p_save, 'weight': weight_save}
    return weights_important

def trefoil_optimized_math_5(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    l_save = [0, 0, 0, 0, 3]
    p_save = [0, 1, 2, 3, 0]
    weight_save = [1.20, -2.85, 7.48, -3.83, -4.38]
    weight_save /= np.sqrt(np.sum(np.array(weight_save) ** 2)) * 100
    weights_important = {'l': l_save, 'p': p_save, 'weight': weight_save}
    return weights_important

def trefoil_standard_12_phase_only(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    mesh_3D_new1 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    mesh_3D_new2 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    xyz_array = [
        (mesh_3D_new1[0], mesh_3D_new1[1], mesh_3D_new1[2]),
        (mesh_3D_new2[0], mesh_3D_new2[1], mesh_3D_new2[2])
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

    ans = 1
    for i, xyz in enumerate(xyz_array):
        if conj_array[i]:
            ans *= np.conjugate(braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                                           a_cos_array[i], a_sin_array[i]))
        else:
            ans *= braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                              a_cos_array[i], a_sin_array[i])
    R = np.sqrt(mesh_3D[0] ** 2 + mesh_3D[1] ** 2)
    ans *= (1 + R ** 2) ** 3
    w = 1.2
    ans *= LG_simple(*mesh_3D[:2], 0, l=0, p=0, width=w, k0=1, x0=0, y0=0, z0=0)
    ans_phase = np.angle(ans)
    ans_amplitude = LG_simple(*mesh_3D[:2], 0, l=0, p=0, width=1.8, k0=1, x0=0, y0=0, z0=0)
    ans = ans_amplitude * np.exp(1j * ans_phase)
    return ans
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

def trefoil_standard_15(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    mesh_3D_new1 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    mesh_3D_new2 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    xyz_array = [
        (mesh_3D_new1[0], mesh_3D_new1[1], mesh_3D_new1[2]),
        (mesh_3D_new2[0], mesh_3D_new2[1], mesh_3D_new2[2])
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

    ans = 1
    for i, xyz in enumerate(xyz_array):
        if conj_array[i]:
            ans *= np.conjugate(braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                                           a_cos_array[i], a_sin_array[i]))
        else:
            ans *= braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                              a_cos_array[i], a_sin_array[i])
    R = np.sqrt(mesh_3D[0] ** 2 + mesh_3D[1] ** 2)
    ans *= (1 + R ** 2) ** 3
    w = 1.5
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
def trefoil_standard_13(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    mesh_3D_new1 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    mesh_3D_new2 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    xyz_array = [
        (mesh_3D_new1[0], mesh_3D_new1[1], mesh_3D_new1[2]),
        (mesh_3D_new2[0], mesh_3D_new2[1], mesh_3D_new2[2])
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

    ans = 1
    for i, xyz in enumerate(xyz_array):
        if conj_array[i]:
            ans *= np.conjugate(braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                                           a_cos_array[i], a_sin_array[i]))
        else:
            ans *= braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                              a_cos_array[i], a_sin_array[i])
    R = np.sqrt(mesh_3D[0] ** 2 + mesh_3D[1] ** 2)
    ans *= (1 + R ** 2) ** 3
    w = 1.3
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

def trefoil_standard_125(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    mesh_3D_new1 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    mesh_3D_new2 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    xyz_array = [
        (mesh_3D_new1[0], mesh_3D_new1[1], mesh_3D_new1[2]),
        (mesh_3D_new2[0], mesh_3D_new2[1], mesh_3D_new2[2])
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

    ans = 1
    for i, xyz in enumerate(xyz_array):
        if conj_array[i]:
            ans *= np.conjugate(braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                                           a_cos_array[i], a_sin_array[i]))
        else:
            ans *= braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                              a_cos_array[i], a_sin_array[i])
    R = np.sqrt(mesh_3D[0] ** 2 + mesh_3D[1] ** 2)
    ans *= (1 + R ** 2) ** 3
    w = 1.25
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

def trefoil_standard_12(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    mesh_3D_new1 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    mesh_3D_new2 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    xyz_array = [
        (mesh_3D_new1[0], mesh_3D_new1[1], mesh_3D_new1[2]),
        (mesh_3D_new2[0], mesh_3D_new2[1], mesh_3D_new2[2])
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

    ans = 1
    for i, xyz in enumerate(xyz_array):
        if conj_array[i]:
            ans *= np.conjugate(braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                                           a_cos_array[i], a_sin_array[i]))
        else:
            ans *= braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                              a_cos_array[i], a_sin_array[i])
    R = np.sqrt(mesh_3D[0] ** 2 + mesh_3D[1] ** 2)
    ans *= (1 + R ** 2) ** 3
    w = 1.2
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


def trefoil_standard_11(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    mesh_3D_new1 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    mesh_3D_new2 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    xyz_array = [
        (mesh_3D_new1[0], mesh_3D_new1[1], mesh_3D_new1[2]),
        (mesh_3D_new2[0], mesh_3D_new2[1], mesh_3D_new2[2])
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

    ans = 1
    for i, xyz in enumerate(xyz_array):
        if conj_array[i]:
            ans *= np.conjugate(braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                                           a_cos_array[i], a_sin_array[i]))
        else:
            ans *= braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                              a_cos_array[i], a_sin_array[i])
    R = np.sqrt(mesh_3D[0] ** 2 + mesh_3D[1] ** 2)
    ans *= (1 + R ** 2) ** 3
    w = 1.1
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


def trefoil_standard_115(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    mesh_3D_new1 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    mesh_3D_new2 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    xyz_array = [
        (mesh_3D_new1[0], mesh_3D_new1[1], mesh_3D_new1[2]),
        (mesh_3D_new2[0], mesh_3D_new2[1], mesh_3D_new2[2])
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

    ans = 1
    for i, xyz in enumerate(xyz_array):
        if conj_array[i]:
            ans *= np.conjugate(braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                                           a_cos_array[i], a_sin_array[i]))
        else:
            ans *= braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                              a_cos_array[i], a_sin_array[i])
    R = np.sqrt(mesh_3D[0] ** 2 + mesh_3D[1] ** 2)
    ans *= (1 + R ** 2) ** 3
    w = 1.15
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


def trefoil_standard_105(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    mesh_3D_new1 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    mesh_3D_new2 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    xyz_array = [
        (mesh_3D_new1[0], mesh_3D_new1[1], mesh_3D_new1[2]),
        (mesh_3D_new2[0], mesh_3D_new2[1], mesh_3D_new2[2])
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

    ans = 1
    for i, xyz in enumerate(xyz_array):
        if conj_array[i]:
            ans *= np.conjugate(braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                                           a_cos_array[i], a_sin_array[i]))
        else:
            ans *= braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                              a_cos_array[i], a_sin_array[i])
    R = np.sqrt(mesh_3D[0] ** 2 + mesh_3D[1] ** 2)
    ans *= (1 + R ** 2) ** 3
    w = 1.05
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

def fivefoil_standard_08(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    mesh_3D_new1 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    mesh_3D_new2 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    xyz_array = [
        (mesh_3D_new1[0], mesh_3D_new1[1], mesh_3D_new1[2]),
        (mesh_3D_new2[0], mesh_3D_new2[1], mesh_3D_new2[2])
    ]
    # starting angle for each braid
    angle_array = np.array([0, 1. * np.pi])
    # powers in cos in sin
    pow_cos_array = [2.5, 2.5]
    pow_sin_array = [2.5, 2.5]
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
    ans *= (1 + R ** 2) ** 5
    w = 0.8
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


def trefoil_optimized(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    l_save = [0, 0, 0, 0, 3]
    p_save = [0, 1, 2, 3, 0]
    weight_save = [1.29, -3.95, 7.49, -3.28, -3.98]
    weight_save /= np.sqrt(np.sum(np.array(weight_save) ** 2)) * 100
    weights_important = {'l': l_save, 'p': p_save, 'weight': weight_save}
    return weights_important


def trefoil_optimized_new(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False):
    l_save = [0, 0, 0, 0, 3, 0, 0]
    p_save = [0, 1, 2, 3, 0, 4, 5]
    weight_save = [1.3, -3.96, 7.53, -3.34, -4.05]
    weight_save = [1.29, -3.95, 7.49, -3.28, -3.98, -0.15, 0.04] #only 67
    weight_save = [1.29, -3.95, 7.49, -3.28, -3.98, -1.22, 0.58] #only 67 step 0.5
    weight_save = [1.35, -3.93, 7.48, -3.29, -4.02, -0.23, -0.47] #all step 0.5
    weight_save = [1.13, -4.02, 7.32, -3.48, -3.69, 0.0, -0.43] #all but 6 step 0.5
    weight_save = [1.05, -4.05, 7.48, -3.03, -4.03, -1.19, 0.0] #all but 7 step 0.5
    
    weight_save /= np.sqrt(np.sum(np.array(weight_save) ** 2)) * 100
    weights_important = {'l': l_save, 'p': p_save, 'weight': weight_save}
    return weights_important

def field_combination_LG(l, p, weights, values=0, mesh=0, w_real=0, k0=1, x0=0, y0=0, z0=0):
    l_save = l
    p_save = p
    weight_save = weights
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


def unknot_4_any(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False,
                 angle_size=(2, 2, 2, 2), cmap='jet'):
    # angle_size=((1, 0), (3, 1), (4, 0))):
    mesh_3D_new = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    for angle, size in enumerate(angle_size):
        angles_dict = {
            0: [(-np.pi / 4, np.pi / 4)],
            1: [(np.pi / 4, 3 * np.pi / 4)],
            2: [(-4 * np.pi / 4, -3 * np.pi / 4), (3 * np.pi / 4, 4 * np.pi / 4)],
            3: [(-3 * np.pi / 4, -np.pi / 4)]
        }
        for ang in angles_dict[angle]:
            if size == 2:
                continue
            if size == 1:
                mesh_3D_new = lobe_smaller(mesh_3D_new, ang[0], ang[1], rot_x=0, rot_y=0, rot_z=0)
            elif size == 0:
                mesh_3D_new = lobe_remove(mesh_3D_new, ang[0], ang[1], rot_x=0, rot_y=0, rot_z=0)
            else:
                print(f"Invalid size {size} for angle {angle}")

    # angles = [1, 2, 3, 4]
    # sizes = [1, 2]
    #
    # A, B = -np.pi / 4, np.pi / 4
    # mesh_3D_new = lobe_smaller(mesh_3D_new, A, B, rot_x=0, rot_y=0, rot_z=0)
    # A, B = -4 * np.pi / 4, -3 * np.pi / 4
    # mesh_3D_new = lobe_remove(mesh_3D_new, A, B, rot_x=0, rot_y=0, rot_z=0)
    # A, B = 3 * np.pi / 4, 4 * np.pi / 4
    # mesh_3D_new = lobe_remove(mesh_3D_new, A, B, rot_x=0, rot_y=0, rot_z=0)
    # A, B = -3 * np.pi / 6, -1 * np.pi / 6
    # mesh_3D_new2 = rotate_part(mesh_3D_new2, A, B, rot_x=0, rot_y=0, rot_z=0)

    xyz_array = [
        (mesh_3D_new[0], mesh_3D_new[1], mesh_3D_new[2]),
    ]
    # starting angle for each braid
    angle_array = np.array([0])
    # powers in cos in sin
    power = 4
    pow_cos_array = [power]
    pow_sin_array = [power]
    # conjugating the braid (in "Milnor" space)
    conj_array = [0]
    # moving x+iy (same as in the paper)
    theta_array = [0.0 * np.pi, 0 * np.pi]
    # braid scaling
    a_cos_array = [1]
    a_sin_array = [1]

    ans = 1
    for i, xyz in enumerate(xyz_array):
        if conj_array[i]:
            ans *= np.conjugate(braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                                           a_cos_array[i], a_sin_array[i]))
        else:
            ans *= braid_func(*xyz, angle_array[i], pow_cos_array[i], pow_sin_array[i], theta_array[i],
                              a_cos_array[i], a_sin_array[i])
    R = np.sqrt(mesh_3D[0] ** 2 + mesh_3D[1] ** 2)
    ans *= (1 + R ** 2) ** power
    ws = {
        0: 3,
        1: 2.6,
        # 2: 1.6,
        2: 2.6 ** (1 / 2),
        3: 1.2,
        4: 0.85,
        5: 0.75,
        6: 0.65,
    }
    w = ws[power]
    ans *= LG_simple(*mesh_3D[:2], 0, l=0, p=0, width=w, k0=1, x0=0, y0=0, z0=0)

    moments = {'p': (0, 10), 'l': (-10, 10)}

    _, _, res_z_3D = np.shape(mesh_3D_new[0])
    x_2D = mesh_3D[0][:, :, 0]
    y_2D = mesh_3D[1][:, :, 0]
    if plot:
        plot_field_both(ans[:, :, res_z_3D // 2])
    values = cbs.LG_spectrum(
        ans[:, :, res_z_3D // 2], **moments, mesh=(x_2D, y_2D), plot=plot, width=w, k0=1, cmap=cmap
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


def unknot_6_any(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False,
                 angle_size=(2, 2, 2, 2, 2, 2), cmap='jet'):
    """
    Create a 6-lobe structure from the input 3D mesh.

    Parameters:
      mesh_3D      : tuple of arrays representing the 3D meshgrid.
      braid_func   : function for the braid operation.
      modes_cutoff : threshold for selecting mode weights.
      plot         : whether to produce plots.
      angle_size   : a 6-element tuple controlling modifications in each lobe.
                     For each element:
                        2 = do nothing,
                        1 = lobe_smaller,
                        0 = lobe_remove.
      cmap         : colormap to use for plotting.

    Returns:
      weights_important : dictionary with keys 'l', 'p', and 'weight' describing the modes.
    """

    # Optionally rotate the original mesh (here with zero rotations)
    mesh_3D_new = rotate_meshgrid(*mesh_3D, np.radians(0), np.radians(0), np.radians(0))

    # Define the angular sectors for the 6 lobes.
    angles_dict = {
        0: [(-np.pi / 6, np.pi / 6)],  # Lobe centered at 0
        1: [(np.pi / 6, np.pi / 2)],  # Lobe centered at π/3
        2: [(np.pi / 2, 5 * np.pi / 6)],  # Lobe centered at 2π/3
        3: [(5 * np.pi / 6, np.pi), (-np.pi, -5 * np.pi / 6)],  # Lobe centered at π (split due to periodicity)
        4: [(-5 * np.pi / 6, -np.pi / 2)],  # Lobe centered at -2π/3
        5: [(-np.pi / 2, -np.pi / 6)]  # Lobe centered at -π/3
    }

    # Modify the mesh in each lobe according to the corresponding entry in angle_size.
    for angle, size in enumerate(angle_size):
        for ang in angles_dict[angle]:
            if size == 2:
                continue
            elif size == 1:
                mesh_3D_new = lobe_smaller(mesh_3D_new, ang[0], ang[1],
                                           rot_x=0, rot_y=0, rot_z=0)
            elif size == 0:
                mesh_3D_new = lobe_remove(mesh_3D_new, ang[0], ang[1],
                                          rot_x=0, rot_y=0, rot_z=0)
            else:
                print(f"Invalid size {size} for angle index {angle}")

    # Build the field using the braid function and additional factors.
    xyz_array = [
        (mesh_3D_new[0], mesh_3D_new[1], mesh_3D_new[2]),
    ]
    # Starting angle for each braid
    angle_array = np.array([0])
    # Powers in cosine and sine terms
    power = 6
    pow_cos_array = [power]
    pow_sin_array = [power]
    # Flag to indicate conjugation
    conj_array = [0]
    # Additional phase shifts (as in your paper)
    theta_array = [0.0 * np.pi, 0 * np.pi]
    # Braid scaling factors
    a_cos_array = [1]
    a_sin_array = [1]

    ans = 1
    for i, xyz in enumerate(xyz_array):
        if conj_array[i]:
            ans *= np.conjugate(braid_func(*xyz, angle_array[i],
                                           pow_cos_array[i], pow_sin_array[i],
                                           theta_array[i], a_cos_array[i], a_sin_array[i]))
        else:
            ans *= braid_func(*xyz, angle_array[i],
                              pow_cos_array[i], pow_sin_array[i],
                              theta_array[i], a_cos_array[i], a_sin_array[i])

    # Multiply by a radial scaling factor
    R = np.sqrt(mesh_3D[0] ** 2 + mesh_3D[1] ** 2)
    ans *= (1 + R ** 2) ** power

    # Use a Laguerre–Gaussian (LG) envelope (adjust parameters as needed)
    ws = {
        0: 3,
        1: 2.6,
        2: 2.6 ** (1 / 2),
        3: 1.2,
        4: 0.85,
        5: 0.75,
        6: 0.65,
    }
    w = ws[power]
    ans *= LG_simple(*mesh_3D[:2], 0, l=0, p=0, width=w, k0=1, x0=0, y0=0, z0=0)

    # Set the moments for the spectrum calculation.
    moments = {'p': (0, 10), 'l': (-10, 10)}
    _, _, res_z_3D = np.shape(mesh_3D_new[0])
    x_2D = mesh_3D[0][:, :, 0]
    y_2D = mesh_3D[1][:, :, 0]

    if plot:
        plot_field_both(ans[:, :, res_z_3D // 2])

    # Calculate the LG spectrum
    values = cbs.LG_spectrum(ans[:, :, res_z_3D // 2],
                             **moments, mesh=(x_2D, y_2D),
                             plot=plot, width=w, k0=1, cmap=cmap)

    # Extract significant mode components.
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
    weight_save = np.array(weight_save) / (np.sqrt(np.sum(np.array(weight_save) ** 2)) * 100)
    weights_important = {'l': l_save, 'p': p_save, 'weight': weight_save}

    return weights_important


def unknot_5_any(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False,
                 angle_size=(2, 2, 2, 2, 2), cmap='jet'):
    """
    Create a 5-lobe structure from the input 3D mesh.

    Parameters:
      mesh_3D      : tuple of arrays representing the 3D meshgrid.
      braid_func   : function for the braid operation.
      modes_cutoff : threshold for selecting mode weights.
      plot         : whether to produce plots.
      angle_size   : a 5-element tuple controlling modifications in each lobe.
                     For each element:
                        2 = do nothing,
                        1 = lobe_smaller,
                        0 = lobe_remove.
      cmap         : colormap to use for plotting.

    Returns:
      weights_important : dictionary with keys 'l', 'p', and 'weight' describing the modes.
    """
    # Optionally rotate the original mesh (here with zero rotations)
    mesh_3D_new = rotate_meshgrid(*mesh_3D, np.radians(0), np.radians(0), np.radians(0))

    # Partition the full circle into 5 equal sectors.
    # Here we divide the interval [-pi, pi] into 5 equal parts.
    # The boundaries are: -pi, -pi + 2π/5, -pi + 4π/5, -pi + 6π/5, -pi + 8π/5, π.
    angles_dict = {
        0: [(-np.pi, -np.pi + 2 * np.pi / 5)],               # Sector 0: [-π, -π+0.4π]
        1: [(-np.pi + 2 * np.pi / 5, -np.pi + 4 * np.pi / 5)],   # Sector 1: [-π+0.4π, -π+0.8π]
        2: [(-np.pi + 4 * np.pi / 5, -np.pi + 6 * np.pi / 5)],   # Sector 2: [-π+0.8π, -π+1.2π] => centered at 0
        3: [(-np.pi + 6 * np.pi / 5, -np.pi + 8 * np.pi / 5)],   # Sector 3: [-π+1.2π, -π+1.6π]
        4: [(-np.pi + 8 * np.pi / 5, np.pi)]                   # Sector 4: [-π+1.6π, π]
    }

    # Modify the mesh in each sector according to the corresponding entry in angle_size.
    for angle, size in enumerate(angle_size):
        for ang in angles_dict[angle]:
            if size == 2:
                continue
            elif size == 1:
                mesh_3D_new = lobe_smaller(mesh_3D_new, ang[0], ang[1],
                                           rot_x=0, rot_y=0, rot_z=0)
            elif size == 0:
                mesh_3D_new = lobe_remove(mesh_3D_new, ang[0], ang[1],
                                          rot_x=0, rot_y=0, rot_z=0)
            else:
                print(f"Invalid size {size} for angle index {angle}")

    # Build the field using the braid function and additional factors.
    xyz_array = [
        (mesh_3D_new[0], mesh_3D_new[1], mesh_3D_new[2]),
    ]
    # Starting angle for each braid
    angle_array = np.array([0])
    # Set the power (which now matches the 5-lobe structure)
    power = 5
    pow_cos_array = [power]
    pow_sin_array = [power]
    # Flag to indicate conjugation
    conj_array = [0]
    # Additional phase shifts (as in your paper)
    theta_array = [0.0 * np.pi, 0 * np.pi]
    # Braid scaling factors
    a_cos_array = [1]
    a_sin_array = [1]

    ans = 1
    for i, xyz in enumerate(xyz_array):
        if conj_array[i]:
            ans *= np.conjugate(braid_func(*xyz, angle_array[i],
                                           pow_cos_array[i], pow_sin_array[i],
                                           theta_array[i], a_cos_array[i], a_sin_array[i]))
        else:
            ans *= braid_func(*xyz, angle_array[i],
                              pow_cos_array[i], pow_sin_array[i],
                              theta_array[i], a_cos_array[i], a_sin_array[i])

    # Multiply by a radial scaling factor
    R = np.sqrt(mesh_3D[0] ** 2 + mesh_3D[1] ** 2)
    ans *= (1 + R ** 2) ** power

    # Use a Laguerre–Gaussian (LG) envelope (adjust parameters as needed)
    ws = {
        0: 3,
        1: 2.6,
        2: 2.6 ** (1 / 2),
        3: 1.2,
        4: 0.85,
        5: 0.75,
        6: 0.65,
    }
    w = ws[power]
    ans *= LG_simple(*mesh_3D[:2], 0, l=0, p=0, width=w, k0=1, x0=0, y0=0, z0=0)

    # Set the moments for the spectrum calculation.
    moments = {'p': (0, 10), 'l': (-10, 10)}
    _, _, res_z_3D = np.shape(mesh_3D_new[0])
    x_2D = mesh_3D[0][:, :, 0]
    y_2D = mesh_3D[1][:, :, 0]

    if plot:
        plot_field_both(ans[:, :, res_z_3D // 2])

    # Calculate the LG spectrum
    values = cbs.LG_spectrum(ans[:, :, res_z_3D // 2],
                             **moments, mesh=(x_2D, y_2D),
                             plot=plot, width=w, k0=1, cmap=cmap)

    # Extract significant mode components.
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
    weight_save = np.array(weight_save) / (np.sqrt(np.sum(np.array(weight_save) ** 2)) * 100)
    weights_important = {'l': l_save, 'p': p_save, 'weight': weight_save}

    return weights_important
def braids_testing(mesh_3D, braid_func=braid, modes_cutoff=0.01, plot=False, cmap='jet'):
    mesh_3D_new1 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    mesh_3D_new2 = rotate_meshgrid(*mesh_3D, np.radians(00), np.radians(00), np.radians(0))
    xyz_array = [
        (mesh_3D_new1[0], mesh_3D_new1[1], mesh_3D_new1[2]),
    ]
    # starting angle for each braid
    angle_array = np.array([0, 1. * np.pi])
    # powers in cos in sin
    pow_cos_array = [4, 1]
    pow_sin_array = [4, 1]
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
    ans *= (1 + R ** 2) ** 4
    w = 1.0
    plot_field_both(ans[:, :, np.shape(ans)[2] // 2])
    plt.show()
    exit()
    ans *= LG_simple(*mesh_3D[:2], 0, l=0, p=0, width=w, k0=1, x0=0, y0=0, z0=0)

    moments = {'p': (0, 6), 'l': (-6, 6)}

    _, _, res_z_3D = np.shape(mesh_3D_new1[0])
    x_2D = mesh_3D[0][:, :, 0]
    y_2D = mesh_3D[1][:, :, 0]
    if plot:
        plot_field_both(ans[:, :, res_z_3D // 2])
    values = cbs.LG_spectrum(
        ans[:, :, res_z_3D // 2], **moments, mesh=(x_2D, y_2D), plot=True, width=w, k0=1, cmap=cmap
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


if __name__ == "__main__":

    x_lim_3D_knot, y_lim_3D_knot, z_lim_3D_knot = (-7.0, 7.0), (-7.0, 7.0), (-2.0, 2.0)
    # x_lim_3D_knot, y_lim_3D_knot, z_lim_3D_knot = (-10.0, 10.0), (-10.0, 10.0), (-2.0, 2.0)
    # x_lim_3D_knot, y_lim_3D_knot, z_lim_3D_knot = (-5.0, 5.0), (-5.0, 5.0), (-2.0, 2.0)
    res_x_3D_knot, res_y_3D_knot, res_z_3D_knot = 90, 90, 40
    # res_x_3D_knot, res_y_3D_knot, res_z_3D_knot = 120, 120, 90
    # res_x_3D_knot, res_y_3D_knot, res_z_3D_knot = 100, 100, 100
    if res_z_3D_knot != 1:
        z_3D_knot = np.linspace(*z_lim_3D_knot, res_z_3D_knot)
    else:
        z_3D_knot = 0

    width0 = 1.6
    k0 = 1
    z0 = 0

    x_3D_knot, y_3D_knot = np.linspace(*x_lim_3D_knot, res_x_3D_knot), np.linspace(*y_lim_3D_knot, res_y_3D_knot)
    mesh_3D_knot = np.meshgrid(x_3D_knot, y_3D_knot, z_3D_knot, indexing='ij')
    # x_2D_origin = np.linspace(*x_lim_3D_knot, res_x_3D_knot)
    # y_2D_origin = np.linspace(*x_lim_3D_knot, res_x_3D_knot)
    # mesh_2D_original = np.meshgrid(x_2D_origin, y_2D_origin, indexing='ij')

    # values = unknot_4_any(mesh_3D_knot, braid_func=braid, plot=True,
    #                       angle_size=(2, 1, 2, 0))
    # values = trefoil_optimized_new(mesh_3D_knot, braid_func=braid, plot=True)
    values = braids_testing(mesh_3D_knot, braid_func=braid, plot=True)
    # field = trefoil_standard_12_phase_only(mesh_3D_knot, braid_func=braid, plot=True)
    # beam_par = (0, 0, width0, 1)
    # psh_par_0 = (1 * 1e100, res_x_3D_knot, (x_lim_3D_knot[1] - x_lim_3D_knot[0]) / res_x_3D_knot, 1, 1 * 1e100)
    # field = beam_expander(field[:, :, res_z_3D_knot // 2], beam_par, psh_par_0, distance_both=5, steps_one=20)
    # print(np.shape(field))
    # l = [0, 0]
    # p = [0, 2]
    # weights = [1, 1]
    # values = field_combination_LG(l, p, weights)
    field = field_knot_from_weights(
        values, mesh_3D_knot, width0, k0=k0, x0=0, y0=0, z0=z0
    )
    # grad_x, grad_y = np.gradient(field[:, :, res_z_3D_knot // 2])
    # magnitude = np.sqrt(np.abs(grad_x) ** 2 + np.abs(grad_y) ** 2)
    # plot_field_both(field[:, :, res_z_3D_knot // 2])
    # plot_field_both(field[:, :, res_z_3D_knot // 4])
    # plot_field_both(field[:, :, 0])
    # plot_field_both(magnitude)
    #
    dots_bound = [
        [0, 0, 0],
        [res_x_3D_knot, res_y_3D_knot, res_z_3D_knot],
    ]
    dots_init_dict, dots_init = sing.get_singularities(np.angle(field), axesAll=True, returnDict=True)
    pl.plotDots(dots_init, dots_bound, color='black', show=True, size=10)
