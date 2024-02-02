import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.signal import convolve2d
from scipy.optimize import curve_fit
from scipy.optimize import brute
# import cv2
import torch
import json
import csv
from functions.all_knots_functions import *
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import collections
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

knot_types = {
    'standard_14': 0,  # 1
    'standard_16': 1,  # 2
    'standard_18': 2,  # 3
    '30both': 3,  # 4
    '30oneZ': 4,  # 5
    'optimized': 5,  # 6
    'pm_03_z': 6,  # 7
    '4foil': 7,  # 8
    '6foil': 8,  # 9
    'stand4foil': 9,  # 10
    '30oneX': 10,  # 11

}
knots = [
    'standard_14', 'standard_16', 'standard_18', '30both', '30oneZ',
    'optimized', 'pm_03_z', '4foil', '6foil', 'stand4foil',
    '30oneX'
]

desired_res = (16, 16, 16)

num_classes = len(knots)
X_list = []
Y_list = []
for knot in knots:
    filename = f'..\data\data_{knot}.csv'
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Deserialize the JSON string back to a list
            data_list = json.loads(row[0])
            # Convert the list back to a NumPy array if needed
            data_array = np.array(data_list)
            points_list = data_array[2:]
            Nx, Ny, Nz = data_array[1]
            if desired_res != (Nx, Ny, Nz):
                scale_x = desired_res[0] / Nx
                scale_y = desired_res[1] / Ny
                scale_z = desired_res[2] / Nz
                points_list = np.rint(points_list * np.array([scale_x, scale_y, scale_z])).astype(int)

            X_list.append(points_list)
            # X_list.append(data_array)
            Y_list.append(knot_types[knot])

dots_bound = [
    [0, 0, 0],
    [*desired_res],
]
pl.plotDots(X_list[0], dots_bound, color='black', show=True, size=10)
