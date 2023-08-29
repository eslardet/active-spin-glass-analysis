import numpy as np
sys.path.insert(1, '/Users/el2021/Code/2D_ActiveSpinGlass_EL/Active_Spin_Glass/analysis')
from analysis_functions import *
import os
import matplotlib.pyplot as plt
import sys
import time


mode = "G"
nPart = 10000
phi = 1.0
noise = "0.20"
#K_avg = 0.0
#K_avg_range = [1.0]
#K_avg_range = np.round(np.arange(0.0,0.1,0.5),1)
#K_avg_range = np.concatenate((np.round(np.arange(-0.5,0.0,0.1),1), np.round(np.arange(0.0,1.1,0.1),1)))
#K_std = 0.0
#K_std_range = np.round(np.arange(1,8.1,1),1)
#K_std_range = [0.0, 1.0, 2.0]
#K = str(K_avg) + "_" + str(K_std)
K = "0.0_8.0"
Rp = 1.0
xTy = 1.0
# seed_range = np.arange(1,6,1)
seed = 1
# timestep_range = np.arange(150,201,10)
timestep_range = [0]
# time_av = np.arange(-10,1,1) 
time_av = np.arange(0,6,1)
r_max = 3
#r_bin_num = 100
# random_sample = True
# samples = 50000
bins = 30

#plot_band_profiles(mode, nPart, phi, noise, K, Rp, xTy, seed)
local_density_distribution_freud(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max=r_max, timestep_range=timestep_range, time_av=time_av, bins=bins)
# local_density_distribution_diff_freud(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max=r_max, timestep_range=timestep_range, time_av=time_av, density_cap=8)
# local_density_distribution_voronoi(mode, nPart, phi, noise, K, Rp, xTy, seed, timestep_range=timestep_range, time_av=time_av, bins=bins)

# snapshot(mode, nPart, phi, noise, K, Rp, xTy, seed, pos_ex=True)

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
