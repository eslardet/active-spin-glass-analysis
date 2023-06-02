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
timestep_range = np.arange(0,1,1)

linlin=True
loglin=False
loglog=False

t0 = time.time()
# fun.plot_corr_density_pos_ex(mode, nPart, phi, noise, K, Rp, xTy, seed_range, linlin=linlin, loglin=loglin, loglog=loglog)
fun.plot_corr_density(mode, nPart, phi, noise, K, Rp, xTy, seed_range, timestep_range=timestep_range, linlin=linlin, loglin=loglin, loglog=loglog)

# inparFile, posFile = fun.get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=1)
# x, y, theta = fun.get_pos_snapshot(posFile, nPart, timestep=5)
# print(x[0])
# posexFile = fun.get_file_path(mode, nPart, phi, noise, K, Rp, xTy, seed=1, file_name="pos_exact")
# x,y,theta,view_time = fun.get_pos_ex_snapshot(posexFile)
# print(x[0])

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
