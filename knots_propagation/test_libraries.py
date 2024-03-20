# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from scipy.signal import convolve2d
# from scipy.optimize import curve_fit
# from scipy.optimize import brute
# # import cv2
# import torch
import json
import csv
# from functions.all_knots_functions import *
from torch.utils.data import TensorDataset, DataLoader
# from torch import nn
from sklearn.model_selection import train_test_split
# from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
# import collections
# import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import pickle
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)