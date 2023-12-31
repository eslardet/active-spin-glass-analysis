import sys
sys.path.insert(1, './analysis/analysis_functions')
from pt import *
from bands_analysis import *

import os
import matplotlib.pyplot as plt
import numpy as np

mode = 'G'
nPart = 10000
phi_range = np.round(np.arange(0.1,2.1,0.1),1)
phi = 1.0
noise = '0.60'
noise_range = [format(i, '.2f') for i in np.arange(0.1,1.05,0.05)]
K_avg_range = np.round(np.arange(0.0,1.1,0.1),1)
K_std_range = [1.0,2.0,3.0,4.0,5.0]
K_std = 1.0
K = 1.0
xTy = 5.0
seed_range = np.arange(1,21,1)


#plot_porder_noise(mode=mode, nPart=nPart, phi=phi, noise_range=noise_range, K=K, xTy=xTy, seed_range=seed_range)
#plot_porder_phi(mode=mode, nPart=nPart, phi_range=phi_range, noise=noise, K=K, xTy=xTy, seed_range=seed_range)
plot_porder_Kavg(mode=mode, nPart=nPart, phi=phi, noise=noise, K_avg_range=K_avg_range, K_std_range=K_std_range, xTy=xTy, seed_range=seed_range)

for K_std in K_std_range:
    plot_var_density_Kavg(mode=mode, nPart=nPart, phi=phi, noise=noise, K_avg_range=K_avg_range, K_std=K_std, xTy=xTy, seed_range=seed_range)

