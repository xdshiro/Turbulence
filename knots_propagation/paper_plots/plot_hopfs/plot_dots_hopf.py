import matplotlib.pyplot as plt

from functions.all_knots_functions import *
from knots_propagation.paper_plots.plots_functions_general import *
import os
import pickle
import csv
import json
from tqdm import trange
import itertools
from matplotlib.colors import LinearSegmentedColormap

foils_paths = [
	'hopf_dots_1e-40_trefoil_standard_12.npy',

]




for path in foils_paths:
	# foil4_dots_sorted = np.roll(np.load(path), 100, axis=0)
	foil4_dots = np.load(path) - np.array([100, 100, 0])
	dots_bound = [
		[-100, -100, 0],
		[100, 100, 129],
	]

	plot_black_dots_paper(foil4_dots, dots_bound=dots_bound)
