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
	'data_foils_turb/foil4_field_XY_0.05_[2, 2, 2, 2].npy',
	# 'data_foils_turb/foil4_field_XY_0.05_[2, 2, 2, 0].npy',
	'data_foils_turb/foil4_field_XY_0.05_[2, 1, 2, 0].npy',
'data_foils_turb/foil4_field_XY_0.15_[2, 2, 2, 2].npy',
	# 'data_foils_turb/foil4_field_XY_0.15_[2, 2, 2, 0].npy',
	'data_foils_turb/foil4_field_XY_0.15_[2, 1, 2, 0].npy',
'data_foils_turb/foil4_field_XY_0.25_[2, 2, 2, 2].npy',
	# 'data_foils_turb/foil4_field_XY_0.25_[2, 2, 2, 0].npy',
	'data_foils_turb/foil4_field_XY_0.25_[2, 1, 2, 0].npy',
]

# foils_paths = [
# 	'data_foils_turb/foil4_field_XY_0.05_[2, 2, 2, 2].npy',
# # 'foil4_field_XY_0.15_[2, 2, 2, 2].npy',
# # 'foil4_field_XY_0.25_[2, 2, 2, 2].npy',
# ]

for path in foils_paths:
	foil4_field = np.load(path)
	foil4_field = foil4_field / np.max(np.abs(foil4_field))
	XY_max = 30e-3 * 185 / 300 * 1e3 / 6 * np.sqrt(2)
	X = [-XY_max, XY_max]
	Y = [-XY_max, XY_max]
	plot_field_both_paper(foil4_field, extend=[*X, *Y], colorbars='both')
