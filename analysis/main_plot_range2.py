import numpy as np
from analysis_functions import *
import os
import matplotlib.pyplot as plt
import sys
import time

mode = 'G'
nPart = 1000
nPart_range = [1000]
#phi_range = np.round(np.arange(0.1,2.1,0.1),1)
phi = 1.0
phi_range = [0.4, 1.0]
noise = '0.20'
noise_range = [format(i, '.2f') for i in np.arange(0.02,0.81,0.02)]
#noise_range = ["0.60"]
#K_avg_range = [format(i, '.2f') for i in np.arange(0.4,0.61,0.01)]
#K_avg_range = np.round(np.concatenate((np.arange(-2.0,0.0,0.1), np.arange(0.0,0.1,0.1))),1)
K_avg_range = [1.0]
#K_std_range = np.round(np.arange(1.0,8.1,1.0),1)
#K_std_range = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,10.0,20.0]
K_std_range = [0.0]
#K_std = 1.0
#K = 1.0
xTy = 1.0
Rp_range = [1.0,2.0,3.0,4.0,5.0,10.0,15.0,20.0,30.0,40.0]
Rp_range = [1.0]
seed_range = np.arange(1,21,1)


#plot_porder_noise(mode=mode, nPart=nPart, phi=phi, noise_range=noise_range, K=K, xTy=xTy, seed_range=seed_range)
#plot_porder_phi(mode=mode, nPart=nPart, phi_range=phi_range, noise=noise, K=K, xTy=xTy, seed_range=seed_range)
#plot_porder_Kavg(mode=mode, nPart=nPart, phi=phi, noise_range=noise_range, K_avg_range=K_avg_range, K_std_range=K_std_range, xTy=xTy, seed_range=seed_range)
#plot_kcrit_kstd(mode=mode, nPart=nPart, phi=phi, noise_range=noise_range, K_avg_range=K_avg_range, K_std_range=K_std_range, xTy=xTy, seed_range=seed_range)

#plot_com_vs_RI(mode=mode, nPart=nPart, phi=phi, noise=noise, K_avg_range=K_avg_range, K_std_range=K_std_range, Rp_range=Rp_range, xTy=xTy, seed_range=seed_range)
#plot_nn_vs_RI(mode=mode, nPart=nPart, phi=phi, noise=noise, K_avg_range=K_avg_range, K_std_range=K_std_range, Rp_range=Rp_range, xTy=xTy, seed_range=seed_range)

plot_nn_vs_noise(mode=mode, nPart_range=nPart_range, phi_range=phi_range, K_avg_range=K_avg_range, K_std_range=K_std_range, Rp_range=Rp_range, xTy=xTy, seed_range=seed_range)
plot_com_vs_noise(mode, nPart_range, phi_range, K_avg_range, K_std_range, Rp_range, xTy, seed_range)

#t0 = time.time()
#for noise in noise_range:
#    plot_var_density_Kavg(mode=mode, nPart=nPart, phi=phi, noise=noise, K_avg_range=K_avg_range, K_std_range=K_std_range, xTy=xTy, seed_range=seed_range)

#print(time.time()-t0)
