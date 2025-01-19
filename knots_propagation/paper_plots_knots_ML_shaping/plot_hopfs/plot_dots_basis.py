import matplotlib.pyplot as plt

from functions.all_knots_functions import *
from knots_propagation.paper_plots_knots_ML_shaping.plots_functions_general import *
import os
import pickle
import csv
import json
from tqdm import trange
import itertools
from matplotlib.colors import LinearSegmentedColormap

foils_paths = [
	'data_all_hopfs_basis\\hopf_dots_1e-40_standard_14.npy',
	'data_all_hopfs_basis\\hopf_dots_1e-40_standard_16.npy',
	'data_all_hopfs_basis\\hopf_dots_1e-40_standard_18.npy',
	'data_all_hopfs_basis\\hopf_dots_1e-40_30both.npy',
	'data_all_hopfs_basis\\hopf_dots_1e-40_30oneZ.npy',
	'data_all_hopfs_basis\\hopf_dots_1e-40_optimized.npy',
	'data_all_hopfs_basis\\hopf_dots_1e-40_pm_03_z.npy',
	'data_all_hopfs_basis\\hopf_dots_1e-40_30oneX.npy',
	'data_all_hopfs_basis\\hopf_dots_1e-40_15oneZ.npy',
	'data_all_hopfs_basis\\hopf_dots_1e-40_trefoil_standard_12.npy',
	'data_all_hopfs_basis\\hopf_dots_1e-40_trefoil_optimized.npy',

]
# foils_paths = [
# 	'data_all_hopfs_basis\\hopf_dots_1e-40_standard_14.npy',
#
#
# ]


for path in foils_paths:
	# foil4_dots_sorted = np.roll(np.load(path), 100, axis=0)
	foil4_dots = np.load(path) - np.array([100 / 2 * 1.5, 100 / 2 * 1.5, 0])
	dots_bound = [
		[-100 / 2 * 1.5, -100 / 2 * 1.5, 0],
		[100 / 2 * 1.5, 100 / 2 * 1.5, 129 / 2 * 1.5],
	]

	plot_black_dots_paper(foil4_dots, dots_bound=dots_bound)
