import sys
sys.path.insert(1, './analysis_functions')
from stats import *
from visuals import *

import time


mode = 'G'
nPart = 10000
phi = 1.0
noise = "0.20"
K_avg_range = np.round(np.concatenate((np.arange(-1.0,0.0,0.1), np.arange(0.0,1.1,0.1))),1)
K_std_range = np.arange(0.0,8.1,1.0)
Rp = 1.0
xTy = 1.0
seed = 1

for K_avg in K_avg_range:
    for K_std in K_std_range:
        K = str(K_avg) + "_" + str(K_std)
        snapshot(mode, nPart, phi, noise, K, Rp, xTy, seed, pos_ex=True)
        plot_polar_hist(mode, nPart, phi, noise, K, Rp, xTy, seed)
