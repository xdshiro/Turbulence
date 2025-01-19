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
	# 'hopf_dots_0.05_trefoil_standard_12.npy',
	'hopf_dots_0.15_trefoil_standard_12.npy',
	# 'hopf_dots_0.25_trefoil_standard_12.npy',

]

for path in foils_paths:
	foil4_dots = np.load(path) - np.array([100, 100, 0])
	# element = [-10, -56, 122]
	# positions = np.where((foil4_dots == element).all(axis=1))[0]
	# foil4_dots = np.delete(foil4_dots, positions, axis=0)
	# np.save('hopf_dots_0.15_trefoil_standard_12.npy', foil4_dots + 1 * np.array([100, 100, 0]))
	# exit()
	dots_bound = [
		[-100, -100, 0],
		[100, 100, 129],
	]
	
	plot_black_dots_paper(foil4_dots, dots_bound=dots_bound)
