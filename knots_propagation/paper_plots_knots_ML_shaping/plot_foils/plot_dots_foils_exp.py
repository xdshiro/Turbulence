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
	'data_foils_exp/foil4_2222_exp_turb_dots.npy',
]
foils_paths = [
	'data_foils_exp/foil4_2222_exp_noturb_dots.npy',
]
# foils_paths = [
# 	'data_foils_exp/foil4_dots_XY_noturb_[2, 2, 2, 2]_sorted.npy',
# ]





for path in foils_paths:
	# foil4_dots_sorted = np.roll(np.load(path), 100, axis=0)
	foil4_dots = np.load(path) - np.array([50, 50, 0])
	# foil4_dots = np.load(path) #- np.array([100, 100, 0])
	# print(foil4_dots)
	# exit()
	dots_bound = [
		[-50, -50, 0],
		[50, 50, 65],
	]
	# foil4_dts_sorted = sort_dots_to_create_line_with_threshold(foil4_dots, TH=30)
	# np.save('foil4_dots_XY_noturb_[2, 2, 2, 2]_sorted.npy', foil4_dots_sorted)
	# plot_3d_line(foil4_dots_sorted)
	plot_black_dots_paper(foil4_dots, dots_bound=dots_bound)
	plot_black_dots_paper(foil4_dots, dots_bound=dots_bound, general_view=True)
# plotDots_foils_paper_by_phi(foil4
# _dots, dots_bound, show=True, size=10)
# print(foil4_dots)
# foil4_field = foil4_field / np.max(np.abs(foil4_field))
# XY_max = 30e-3 * 185 / 300 * 1e3
# X = [-XY_max, XY_max]
# Y = [-XY_max, XY_max]
# plot_field_both_paper(foil4_field, extend=[*X, *Y])
