import numpy as np
import analysis_functions_vicsek_rep as fun
import os
import matplotlib.pyplot as plt
import sys
import time

mode = 'G'
nPart = 1000
phi_range = np.round(np.arange(0.1,2.1,0.1),1)
phi = 0.3
noise = '0.60'
#noise_range = [format(i, '.2f') for i in np.arange(0.20,0.65,0.20)]
noise_range = ["0.20", "0.60"]
#K_avg_range = np.round(np.arange(0.0,1.1,0.1),1)
K_avg_range = np.round(np.concatenate((np.arange(-1.0,0.0,0.5), np.arange(0.0,5.1,0.5))),1)
K_std_range = np.round(np.arange(1.0,8.1,1.0),1)
K_std = 1.0
K = 1.0
xTy = 5.0
seed_range = np.arange(1,21,1)


#plot_porder_noise(mode=mode, nPart=nPart, phi=phi, noise_range=noise_range, K=K, xTy=xTy, seed_range=seed_range)
#plot_porder_phi(mode=mode, nPart=nPart, phi_range=phi_range, noise=noise, K=K, xTy=xTy, seed_range=seed_range)
#plot_porder_Kavg(mode=mode, nPart=nPart, phi=phi, noise_range=noise_range, K_avg_range=K_avg_range, K_std_range=K_std_range, xTy=xTy, seed_range=seed_range)

plot_kcrit_kstd(mode=mode, nPart=nPart, phi=phi, noise_range=noise_range, K_avg_range=K_avg_range, K_std_range=K_std_range, xTy=xTy, seed_range=seed_range)

#plot_porder_Kratio(mode=mode, nPart=nPart, phi=phi, noise=noise, K_avg_range=K_avg_range, K_std_range=K_std_range, xTy=xTy, seed_range=seed_range)

#t0 = time.time()
#for noise in noise_range:
#    plot_var_density_Kavg(mode=mode, nPart=nPart, phi=phi, noise=noise, K_avg_range=K_avg_range, K_std_range=K_std_range, xTy=xTy, seed_range=seed_range)

#print(time.time()-t0)
