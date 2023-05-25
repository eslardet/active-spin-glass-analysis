import numpy as np
import numpy as np
import analysis_functions_vicsek_new as fun
import os
import matplotlib.pyplot as plt
import sys
import time

mode = 'G'
nPart = 10000
phi = 1.0
noise = "0.20"
K_avg = 0.0
K_std = 8.0
K = str(K_avg) + "_" + str(K_std)
Rp = 1.0
xTy = 1.0
seed_range = np.arange(1,2,1)

linlin=True
loglin=True
loglog=False
d_type='dv'
r_max=10
r_bin_num=20

t0 = time.time()
fun.plot_corr_vel(mode, nPart, phi, noise, K, Rp, xTy, seed_range, d_type=d_type, r_max=r_max, r_bin_num=r_bin_num, linlin=linlin, loglin=loglin, loglog=loglog)

print("Time taken: " + str(time.time() - t0))

# xscale='lin'
# yscale='lin'
# d_type='dv_perp'
# r_max=10
# r_bin_num=20

# t0 = time.time()
# fun.plot_corr_vel(mode, nPart, phi, noise, K, Rp, xTy, seed_range, xscale=xscale, yscale=yscale, d_type=d_type, r_max=r_max, r_bin_num=r_bin_num)

# print("Time taken: " + str(time.time() - t0))

# xscale='lin'
# yscale='lin'
# d_type='dv_par'
# r_max=10
# r_bin_num=20

# t0 = time.time()
# fun.plot_corr_vel(mode, nPart, phi, noise, K, Rp, xTy, seed_range, xscale=xscale, yscale=yscale, d_type=d_type, r_max=r_max, r_bin_num=r_bin_num)

# print("Time taken: " + str(time.time() - t0))
