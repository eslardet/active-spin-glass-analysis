import numpy as np
import sys
sys.path.insert(1, '/Users/el2021/Code/2D_ActiveSpinGlass_EL/Active_Spin_Glass/analysis')
from analysis_functions import *
import os
import matplotlib.pyplot as plt
import time
import csv


mode = "G"
nPart = 22500
phi = 1.0
noise = "0.20"
K_avg = 1.0
# K_std_range = np.arange(1.0, 8.1, 1.0)
K = "1.0_1.0"
Rp = 1.0
xTy=1.0
seed=113

# plot_corr_vel(mode, nPart, phi, noise, K, xTy, seed, type='v', r_max=10, r_bin_num=100)

# plot_dist_coupling_hist(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed, bin_size=100, bin_ratio=1, diff=True)
# plot_dist_coupling_hist_diff(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed, bin_size=50)
# del_files(mode, nPart, phi, noise, K, xTy, seed, files=["coupling"])

# snapshot(mode, nPart, phi, noise, K, xTy, seed, pos_ex=True)
# for t in timestep_range:
#     snapshot(mode, nPart, phi, noise, K, Rp, xTy, seed, pos_ex=False, show_color=True, save_in_folder=False, timestep=t)
# plot_average_band_profile(mode=mode, nPart=nPart, phi=phi, noise=noise, K_avg_range=K_avg_range, K_std_range=K_std_range, Rp=Rp, xTy=xTy, seed_range=seed_range, pos_ex=False, timestep_range=timestep_range, min_grid_size=3)

# print(neighbour_stats(mode, nPart, phi, noise, K, Rp, xTy, seed, pos_ex=True))

posFileExact = get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, file_name="pos_exact")
x, y, theta, view_time = get_pos_ex_snapshot(posFileExact)
print(np.max(x))
print(np.max(y))
print(np.max(theta))