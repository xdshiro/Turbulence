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
    'data_trefoils_turb\\hopf_dots_0.15_trefoil_standard_12.npy',
    # 'hopf_dots_0.25_trefoil_standard_12.npy',

]

# Coordinates you want to delete
coordinates_to_delete = [
    [16, 9, 43],
    [14, 0, 43],  # add as many as you need
[10, 15, 39],
[11, 12, 40],
[27, 15, 45],
[42, 23, 71],
[38, 26, 71],
[38, 23, 69],
[38, 32, 68],
[38, 7, 63],
[14, 3, 40],
[24, 13, 40],
[12, 6, 41],
[12, 10, 40],


[15, -23, 28],
[11, 3, 44],
[5, 12, 38],
[24, 14, 37],
[20, -24, 26],
[19, -25, 25],
[22, -26, 25],
[38, 25, 70],
[38, 21, 68],
[-13, 54, 10],
[2, 2, 25],
[10, 10, 44],
[43, 20, 67],
]

for path in foils_paths:
    foil4_dots = np.load(path) - np.array([100, 100, 0])

    # Delete specified dots
    for coord in coordinates_to_delete:
        positions = np.where((foil4_dots == coord).all(axis=1))[0]
        if len(positions) > 0:
            foil4_dots = np.delete(foil4_dots, positions, axis=0)

    # (Optional) Save modified dots
    # np.save('new_filename.npy', foil4_dots + np.array([100, 100, 0]))

    dots_bound = [
        [-100, -100, 0],
        [100, 100, 129],
    ]

    plot_black_dots_paper(foil4_dots, dots_bound=dots_bound, general_view=True, font_size=48)