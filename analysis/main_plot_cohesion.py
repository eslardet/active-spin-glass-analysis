import numpy as np
from analysis_functions import *
import os
import matplotlib.pyplot as plt
import sys
import time

mode = 'G'
nPart = 10000
nPart_range = [1000]
# phi_range = np.round(np.arange(0.1,2.1,0.1),1)
phi = 1.0
phi_range = [1.0]
noise = '0.20'
# noise_range = [format(i, '.2f') for i in np.arange(0.04,0.81,0.02)]
noise_range = ["0.20"]
#K_avg_range = [format(i, '.2f') for i in np.arange(0.4,0.61,0.01)]
# K_avg_range = np.round(np.concatenate((np.arange(-1.0,0.0,0.1), np.arange(0.0,1.1,0.1))),1)
K_avg_range = [1.0]
# K_std_range = np.round(np.arange(0.0,8.1,1.0),1)
# K_std_range = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,10.0,20.0]
K_std_range = [0.0]
#K_std = 1.0
#K = 1.0
xTy = 1.0
Rp_range = np.concatenate((np.round(np.arange(0.2,2.1,0.2),1), np.arange(3.0,4.1,1.0)))
# Rp_range = np.concatenate((np.round(np.arange(0.1,1.1,0.1),1), np.arange(2.0, 5.1, 1.0)))
# Rp_range = [1.0]
seed_range = np.arange(1,21,1)
from_stats = True
save_data = True

t0 = time.time()

# plot_nn_vs_noise(mode=mode, nPart_range=nPart_range, phi_range=phi_range, noise_range=noise_range, K_avg_range=K_avg_range, K_std_range=K_std_range, Rp_range=Rp_range, xTy=xTy, seed_range=seed_range, from_stats=from_stats, save_data=save_data)
# plot_com_vs_noise(mode, nPart_range, phi_range, noise_range, K_avg_range, K_std_range, Rp_range, xTy, seed_range, from_stats, save_data)

# plot_nn_vs_Kavg(mode, nPart_range, phi_range, noise_range, K_avg_range, K_std_range, Rp_range, xTy, seed_range, from_stats, save_data)

plot_nn_vs_RI(mode, nPart, phi, noise, K_avg_range, K_std_range, Rp_range, xTy, seed_range, from_stats, save_data)
# plot_com_vs_RI(mode, nPart, phi, noise, K_avg_range, K_std_range, Rp_range, xTy, seed_range, from_stats, save_data)
# plot_av_n_vs_RI(mode, nPart, phi, noise, K_avg_range, K_std_range, Rp_range, xTy, seed_range, from_stats, save_data)

print(time.time()-t0)
