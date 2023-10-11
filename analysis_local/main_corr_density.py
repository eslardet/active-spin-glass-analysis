import numpy as np
import sys
sys.path.insert(1, '/Users/el2021/Code/2D_ActiveSpinGlass_EL/Active_Spin_Glass/analysis')
from analysis_functions import *
import os
import matplotlib.pyplot as plt
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
# K_avg_range = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
K_std_range = np.arange(1.0, 8.1, 1.0)
Rp = 1.0
xTy = 1.0
seed_range = np.arange(1,2,1)
r_scale = "log"
log_y = True
pos_ex = True
min_grid_size = 1
min_r = 0
max_r = 2
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

# plot_exponents_Kavg_corr_density(mode, nPart_range, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed_range, min_r=2, max_r=10)
# plot_corr_density(mode, nPart, phi, noise, K, Rp, xTy, seed_range, pos_ex=True, timestep_range=[0], log_y=True, min_grid_size=1, min_r=0, max_r=2)

# print(get_exponent_corr_density_grid(mode, nPart, phi, noise, K, Rp, xTy, seed_range, pos_ex, timestep_range, min_grid_size, min_r, max_r))

# plot_exponents_Kstd_corr_density_grid(mode, nPart, phi, noise, K_avg, K_std_range, Rp, xTy, seed_range, pos_ex, timestep_range, min_grid_size, min_r, max_r)

expo = []
for K_std in K_std_range:
    K = str(K_avg) + "_" + str(K_std)
    dist, corr = get_r_corr_all(mode, nPart, phi, noise, K, Rp, xTy, seed_range, pos_ex=True, timestep_range=[0], min_grid_size=1, take_abs=False)
    r, corr = get_corr_binned(dist, corr, min_r=0, max_r=50)
    corr = np.array(corr)
    idx = np.where(corr<0)[0]
    r = np.array(r)
    expo.append(r[idx[0]])

plt.plot(K_std_range, expo, '-o')
plt.xlabel(r"$\sigma_K$")
plt.ylabel(r"$\xi$")
plt.show()

print("Time taken: " + str(time.time() - t0))
