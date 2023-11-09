import sys
sys.path.insert(1, './analysis/analysis_functions')
from neighbours_density import *

import numpy as np
import matplotlib.pyplot as plt



mode = "G"
nPart = 10000
phi = 1.0
noise = "0.20"
K_avg = 0.0
K_std_range = [1.0, 2.0, 8.0]
# K = "0.0_8.0"
Rp = 1.0
xTy = 1.0
seed = 1
r_max = 1
n_max = 35
c_max = 1800

stats_list = []

# K = "0.0_1.0"
# neighbour_hist(mode, nPart, phi, noise, K, xTy, seed)
# snapshot(mode, nPart, phi, noise, K, xTy, seed, pos_ex=True)

# for K_std in K_std_range:
K_std = 1.0
K = str(K_avg) + "_" + str(K_std)
neighbour_hist(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max=r_max, n_max=n_max, c_max=c_max)
# snapshot(mode, nPart, phi, noise, K, Rp, xTy, seed, pos_ex=True, neigh_col=True, r_max=r_max)
# print(neighbour_stats(mode, nPart, phi, noise, K, xTy, seed, pos_ex=True))

