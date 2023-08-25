import numpy as np
from analysis.analysis_functions import *
import os
import matplotlib.pyplot as plt
import sys
import time
import csv

# mode = "C"
# nPart = int(sys.argv[1])
# phi = float(sys.argv[2])
# noise = float(sys.argv[3])
# K = str(sys.argv[4])
# xTy = float(sys.argv[5])
# seed = int(sys.argv[6])

# mode = "G"
# nPart = 50000
# phi = 1.0
# noise = "0.70"
# K = "1.0_0.0"
# K_avg_range = [1.0]
# K_std_range = np.arange(0.0, 2.1, 1.0)
# Rp = 1.0
# xTy=5.0
# seed_range=np.arange(1,5,1)
# seed=1
# timestep_range=np.arange(0,11,1)

mode = "G"
nPart = 100
phi = 1.0
noise = "0.20"
K_avg = 0.0
# K_std_range = np.arange(1.0, 8.1, 1.0)
K = "0.0_8.0"
Rp = 2.0
xTy=5.0
seed=1

# plot_corr_vel(mode, nPart, phi, noise, K, xTy, seed, type='v', r_max=10, r_bin_num=100)

plot_dist_coupling_hist(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed, bin_size=100, bin_ratio=1, diff=True)
# plot_dist_coupling_hist_diff(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed, bin_size=50)
# del_files(mode, nPart, phi, noise, K, xTy, seed, files=["coupling"])

# snapshot(mode, nPart, phi, noise, K, xTy, seed, pos_ex=True)
# for t in timestep_range:
#     snapshot(mode, nPart, phi, noise, K, Rp, xTy, seed, pos_ex=False, show_color=True, save_in_folder=False, timestep=t)
# plot_average_band_profile(mode=mode, nPart=nPart, phi=phi, noise=noise, K_avg_range=K_avg_range, K_std_range=K_std_range, Rp=Rp, xTy=xTy, seed_range=seed_range, pos_ex=False, timestep_range=timestep_range, min_grid_size=3)

# print(neighbour_stats(mode, nPart, phi, noise, K, Rp, xTy, seed, pos_ex=True))
