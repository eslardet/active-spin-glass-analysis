import numpy as np
import numpy as np
import analysis_functions_vicsek_new as fun
import os
import matplotlib.pyplot as plt
import sys
import time

mode = 'G'
nPart = 1000
phi = 1.0
noise = "0.20"
K_avg = 0.0
K_std = 8.0
K = str(K_avg) + "_" + str(K_std)
Rp = 1.0
xTy = 1.0
seed_range = np.arange(1,2,1)
r_scale = "log"
y_scale = "log"
timestep_range = [0,1,2,3,4,5]

# linlin=False
# loglin=False
# loglog=True
d_type='dv'
corr_r_max=10
r_bin_num=120

t0 = time.time()
# fun.plot_corr_vel(mode, nPart, phi, noise, K, Rp, xTy, seed_range, d_type=d_type, r_max=r_max, r_bin_num=r_bin_num, linlin=linlin, loglin=loglin, loglog=loglog)

fun.write_corr_vel(mode, nPart, phi, noise, K, Rp, xTy, seed_range, r_scale, timestep_range, d_type, corr_r_max, r_bin_num)
# fun.read_corr_vel(mode, nPart, phi, noise, K, Rp, xTy, seed_range, r_scale, d_type)
fun.plot_corr_vel_file(mode, nPart, phi, noise, K, Rp, xTy, seed_range, d_type, x_scale=r_scale, y_scale=y_scale, bin_ratio=1)

print("Time taken: " + str(time.time() - t0))
