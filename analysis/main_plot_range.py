
import numpy as np
from analysis_functions import *
import os
import matplotlib.pyplot as plt
import sys
import time

mode = 'F'
#nPart=10000
nPart_range = [10000]
#phi_range = np.round(np.arange(0.1,2.1,0.1),1)
#phi_range = [0.1,0.2,0.5,1.0,2.0,3.0,4.0,10.0]
phi_range = [1.0]
phi = 1.0
noise = '0.20'
#noise_range = [format(i, '.3f') for i in np.arange(0.800,0.8401,0.001)]
noise_range = ["0.20"]
#K_avg_range = [format(i, '.3f') for i in np.arange(0.47,0.49,0.001)]
#K_avg_range = np.round(np.arange(0.0,1.1,0.1),2)
K_avg_range = np.round(np.concatenate((np.arange(-1.0,0.0,0.1), np.arange(0.0,1.1,0.1))),1)
#K_std_range = [8.0]
K_std_range = np.round(np.arange(1.0,8.1,1.0),1)
#K_std_range = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,10.0,20.0]
#K_std_range = [1.0, 4.0, 8.0]
K_avg = 0.2
K_std = 8.0
#K = 1.0
#K0_range = [1.0]
#alpha_range = np.round(np.arange(0.0, 1.01, 0.1))
xTy = 1.0
Rp = 1.0
Rp_range = [1.0]
#Rp_range = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 30.0]
#Rp_range = np.arange(1.0,5.1,1.0)
seed_range = np.arange(1,21,1)
save_data = True


#K = "0.0430_0.0"
##print(get_binder(mode, nPart_range, phi, noise_range, K_avg_range, K_std_range Rp_range, xTy, seed_range))

#plot_porder_noise(mode=mode, nPart=nPart, phi=phi, noise_range=noise_range, K=K, xTy=xTy, seed_range=seed_range)
#plot_porder_phi(mode=mode, nPart=nPart, phi_range=phi_range, noise=noise, K=K, xTy=xTy, seed_range=seed_range)
plot_porder_Kavg(mode=mode, nPart_range=nPart_range, phi_range=phi_range, noise_range=noise_range, K_avg_range=K_avg_range, K_std_range=K_std_range, Rp_range=Rp_range, xTy=xTy, seed_range=seed_range, save_data=save_data)
#plot_porder_noise(mode, nPart_range, phi, noise_range, K_avg_range, K_std_range, Rp_range, xTy, seed_range, save_data)
#plot_porder_Kavg(mode, nPart, phi, noise_range, K_avg_range, K_std_range, xTy, seed_range)
#plot_kcrit_kstd(mode=mode, nPart=nPart, phi=phi, noise_range=noise_range, K_avg_range=K_avg_range, K_std_range=K_std_range, xTy=xTy, seed_range=seed_range)
#plot_porder_alpha(mode=mode, nPart_range=nPart_range, phi=phi, noise_range=noise_range, K0_range=K0_range, alpha_range=alpha_range, Rp_range=Rp_range, xTy=xTy, seed_range=seed_range)

#plot_porder_logL(mode=mode, nPart_range=nPart_range, phi=phi, noise=noise, K_avg=K_avg, K_std=K_std, Rp=Rp, xTy=xTy, seed_range=seed_range, save_data=save_data)

#plot_norder_Kavg(mode, nPart_range, phi, noise_range, K_avg_range, K_std_range, Rp_range, xTy, seed_range, save_data)

#plot_binder_Kavg(mode=mode, nPart_range=nPart_range, phi=phi, noise_range=noise_range, K_avg_range=K_avg_range, K_std_range=K_std_range, Rp_range=Rp_range, xTy=xTy, seed_range=seed_range)

#t0 = time.time()
#for noise in noise_range:
#    plot_var_density_Kavg(mode=mode, nPart=nPart, phi=phi, noise=noise, K_avg_range=K_avg_range, K_std_range=K_std_range, xTy=xTy, seed_range=seed_range)

#print(time.time()-t0)
