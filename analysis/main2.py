import numpy as np
from analysis_functions import *
import os
import matplotlib.pyplot as plt
import sys
import time


mode = "G"
nPart = 10000
phi = 1.0
noise = "0.60"
K_avg = 2.0
#K_avg_range = [1.0]
#K_avg_range = np.round(np.arange(0.0,0.1,0.5),1)
#K_avg_range = np.concatenate((np.round(np.arange(-0.5,0.0,0.1),1), np.round(np.arange(0.0,1.1,0.1),1)))
K_std = 1.0
#K_std_range = np.round(np.arange(1,8.1,1),1)
#K_std_range = [0.0, 1.0, 5.0, 10.0]
K = str(K_avg) + "_" + str(K_std)
Rp = 1.0
xTy = 5.0
seed_range = np.arange(2,3,1)
seed = 1
timestep_range = range(1)
r_max = 20
r_bin_num = 100

#snapshot(mode, nPart, phi, noise, K, xTy, seed, pos_ex=True)

plot_corr_vel(mode, nPart, phi, noise, K, xTy, seed, type='v', r_max=r_max, r_bin_num=r_bin_num)

##plot_band_profiles(mode, nPart, phi, noise, K, Rp, xTy, seed)
##plot_average_band_profile(mode, nPart, phi, noise, K, Rp, xTy, seed_range)

#plot_dist_coupling(mode, nPart, phi, noise, K, xTy, seed)

#plot_corr_vel_fluc(mode, nPart, phi, noise, K, xTy, seed)
# plot_correlation(mode=mode, nPart=nPart, phi=phi, noise=noise, K_avg_range=K_avg_range, K_std_range=K_std_range, xTy=xTy, seed_range=seed_range, timestep_range=timestep_range, pos_ex=True)

# snapshot(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, xTy=xTy, seed=seed, pos_ex=True)
# animate(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, xTy=xTy, seed=seed)

##plot_porder_time(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed)
#write_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed, remove_pos=True)

#for K_std in K_std_range:
#    for K_avg in K_avg_range:
#        K = str(K_avg) + "_" + str(K_std)
        #read_stats(mode, nPart, phi, noise, K, xTy, seed)
        #snapshot(mode, nPart, phi, noise, K, xTy, seed, pos_ex=True)
#        plot_density_profile(mode, nPart, phi, noise, K, Rp, xTy, seed)
