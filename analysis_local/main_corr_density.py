import numpy as np
import numpy as np
from analysis.analysis_functions import *
import os
import matplotlib.pyplot as plt
import sys
import time

mode = 'G'
nPart = 10000
nPart_range = [10000]
phi = 1.0
noise = "0.20"
K_avg = 0.0
K_std = 8.0
K = str(K_avg) + "_" + str(K_std)
K_arr = [1.0,1.5,2.0]
K_avg_range = np.concatenate((np.round(np.arange(-1.0,0.0,0.1),1),np.round(np.arange(0.0,0.6,0.1),1),K_arr))
# K_avg_range = [-1.0, -0.5, 0.0, 0.5]
K_std_range = [8.0]
Rp = 1.0
xTy = 1.0
seed_range = np.arange(1,21,1)
r_scale = "log"
log_y = True
timestep_range = [0,1,2,3,4,5]

# corr_r_min=0.1
# corr_r_max=10
# r_bin_num=120

t0 = time.time()
# plot_corr_vel(mode, nPart, phi, noise, K, Rp, xTy, seed_range, d_type=d_type, r_max=r_max, r_bin_num=r_bin_num, linlin=linlin, loglin=loglin, loglog=loglog)

# write_corr_vel(mode, nPart, phi, noise, K, Rp, xTy, seed_range, r_scale, timestep_range, d_type, corr_r_min, corr_r_max, r_bin_num)
# # read_corr_vel(mode, nPart, phi, noise, K, Rp, xTy, seed_range, r_scale, d_type)
# plot_corr_vel_file(mode, nPart, phi, noise, K, Rp, xTy, seed_range, d_type, x_scale=r_scale, y_scale=y_scale, bin_ratio=1)

# plot_corr_density_file_superimpose(mode, nPart, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed_range, r_scale, log_y, bin_ratio=1)

# plot_corr_density_file_superimpose(mode, nPart, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed_range, log_y=True, bin_ratio=1)

plot_exponents_Kavg_corr_density(mode, nPart_range, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed_range, min_r=2, max_r=10)

print("Time taken: " + str(time.time() - t0))
