import sys
sys.path.insert(1, './analysis_functions')
from pt import *

import time

mode = 'G'
#nPart=10000
nPart_range = [10000]
#phi_range = np.round(np.arange(0.1,2.1,0.1),1)
#phi_range = [0.1,0.2,0.5,1.0,2.0,3.0,4.0,10.0]
phi_range = [1.0]
phi = 1.0
noise = '0.20'
# noise_range = [format(i, '.2f') for i in np.arange(0.1,0.81,0.02)]
noise_range = ["0.05"]
# K_avg_range = [1.0]
# K_avg_range = [format(i, '.3f') for i in np.arange(0.47,0.49,0.001)]
# K_avg_range = np.round(np.arange(-1.0,1.0,0.1),1)
K_avg_range = np.round(np.concatenate((np.arange(-2.0,0.0,0.1), np.arange(0.0,1.1,0.1))),1)
K_std_range = [1.0, 4.0, 8.0]
# K_std_range = np.round(np.arange(0.0,8.1,1.0),1)
#K_std_range = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,10.0,20.0]
# K_std_range = [0.0, 1.0, 4.0, 8.0]
K_avg = 0.2
K_std = 8.0
xTy = 1.0
Rp = 1.0
Rp_range = [1.0]
seed_range = np.arange(1,21,1)
save_data = True

# plot_porder_Kavg(mode=mode, nPart_range=nPart_range, phi_range=phi_range, noise_range=noise_range, K_avg_range=K_avg_range, K_std_range=K_std_range, Rp_range=Rp_range, xTy=xTy, seed_range=seed_range, save_data=save_data)
# plot_porder_noise(mode, nPart_range, phi, noise_range, K_avg_range, K_std_range, Rp_range, xTy, seed_range)

plot_psus_Kavg(mode=mode, nPart_range=nPart_range, phi_range=phi_range, noise_range=noise_range, K_avg_range=K_avg_range, K_std_range=K_std_range, Rp_range=Rp_range, xTy=xTy, seed_range=seed_range, save_data=save_data)