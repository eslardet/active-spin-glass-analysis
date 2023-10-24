import sys
sys.path.insert(1, './analysis_functions')
from pt import *

import time


mode = 'G'
#nPart=10000
nPart_range = [1000,1600,2500,3600,6400,10000,22500,40000,90000,160000]
phi = 1.0
noise = '0.20'
noise_range = ["0.20"]
K_avg = 0.1
K_std = 1.0
xTy = 1.0
Rp = 1.0
Rp_range = [1.0]
seed_range = np.arange(1,101,1)
save_data = True
y_log = False

for K_avg in [1.0]:
    K = str(K_avg) + "_" + str(K_std)
    plot_porder_logL(mode=mode, nPart_range=nPart_range, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed_range=seed_range, save_data=save_data, y_log=y_log)

