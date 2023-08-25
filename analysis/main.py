import numpy as np
from analysis.analysis_functions import *
import os
import matplotlib.pyplot as plt
import sys
import time


mode = "A"
nPart = 10000
phi = 1.0
noise = "0.20"
#noise_range = ["0.20", "0.40", "0.60", "0.80"]
K_avg = 1.0
#K_avg_range = [4.0]
#K_avg_range = np.round(np.arange(0.0,0.1,0.5),1)
#K_avg_range = np.concatenate((np.round(np.arange(-1.0,0.0,0.5),1), np.round(np.arange(0.0,2.1,0.5),1)))
K_std = 1.0
#K_std_range = np.round(np.arange(1.0,7.1,1.0),1)
#K_std_range = [0.0]
#K = str(K_avg) + "_" + str(K_std)
Rp = 1.0
xTy = 1.0
seed_range = np.arange(1,21,1)
seed = 1
#timestep_range = range(1)

#for seed in seed_range:
#    for K_avg in K_avg_range:
#        K = str(K_avg) + "_" + str(K_std)
#    plot_dist_coupling_hist(mode, nPart, phi, noise, K, xTy, seed, bin_size=160, bin_ratio=16 , r_max=5)
#        del_files(mode, nPart, phi, noise, K, Rp, xTy, seed, files=["pos"])

animate(mode, nPart, phi, noise, K, Rp, xTy, seed)

#plot_dist_coupling(mode, nPart, phi, noise, K, xTy, seed)
#dist_coupling(mode, nPart, phi, noise, K, xTy, seed)

#plot_corr_vel_fluc(mode, nPart, phi, noise, K, xTy, seed, scatter=False)
# plot_correlation(mode=mode, nPart=nPart, phi=phi, noise=noise, K_avg_range=K_avg_range, K_std_range=K_std_range, xTy=xTy, seed_range=seed_range, timestep_range=timestep_range, pos_ex=True)

# snapshot(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, xTy=xTy, seed=seed, pos_ex=True)
# animate(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, xTy=xTy, seed=seed)

#plot_porder_time(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
#write_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, moments=True, remove_pos=False)

#for K_std in K_std_range:
#    for K_avg in K_avg_range:
#        K = str(K_avg) + "_" + str(K_std)
        #read_stats(mode, nPart, phi, noise, K, xTy, seed)
        #snapshot(mode, nPart, phi, noise, K, xTy, seed, pos_ex=True)
#        plot_density_profile(mode, nPart, phi, noise, K, Rp, xTy, seed)

#for noise in noise_range:
#    snapshot(mode, nPart, phi, noise, K, xTy, seed, pos_ex=True)
