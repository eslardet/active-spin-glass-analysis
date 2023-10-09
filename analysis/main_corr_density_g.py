import numpy as np
from analysis_functions import *
import os
import matplotlib.pyplot as plt
import sys
import time

mode = 'G'
nPart = 10000
phi = 1.0
noise = "0.20"
# K_avg = 0.0
# K_std = 8.0
K_avg_range = [0.0]
K_std_range = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]
# K = str(K_avg) + "_" + str(K_std)
Rp = 1.0
xTy = 1.0
seed_range = np.arange(1,21,1)

pos_ex = False
timestep_range = np.arange(0,6,1)
log_x = False
log_y = True
min_grid_size = 0.5
min_r = 0
max_r = 10

for log_y in [False, True]:
    t0 = time.time()
    plot_corr_density_superimpose(mode, nPart, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed_range, 
                                    pos_ex, timestep_range, log_x, log_y, min_grid_size, min_r, max_r)
    print("Time taken: " + str(time.time() - t0))

# xscale='lin'
# yscale='lin'
# d_type='dv_perp'
# r_max=10
# r_bin_num=20

# t0 = time.time()
# plot_corr_vel(mode, nPart, phi, noise, K, Rp, xTy, seed_range, xscale=xscale, yscale=yscale, d_type=d_type, r_max=r_max, r_bin_num=r_bin_num)

# print("Time taken: " + str(time.time() - t0))

# xscale='lin'
# yscale='lin'
# d_type='dv_par'
# r_max=10
# r_bin_num=20

# t0 = time.time()
# plot_corr_vel(mode, nPart, phi, noise, K, Rp, xTy, seed_range, xscale=xscale, yscale=yscale, d_type=d_type, r_max=r_max, r_bin_num=r_bin_num)

# print("Time taken: " + str(time.time() - t0))
