import sys
sys.path.insert(1, './analysis/analysis_functions')
from visuals import *

import numpy as np
import os
import matplotlib.pyplot as plt
import time


# f.plot_vorder_ksd(mode="G", nPart=500, phi=0.2, KAVG=1.0, KSTD_range=np.arange(0, 41, 1.0), seed=2, save=True, view=False)

# f.snapshot(mode="C", nPart=1000, phi=0.4, K=1.0, seed=2, view_time=10)


mode = "G"
nPart = 50000
phi = 1.0
noise = "0.20"
# K_std = 8.0
# K_avg_range = [0.1]
# K_avg_range = np.round(np.arange(0.0, 0.8, 0.1),1)
# K = "5.0_5.0_1.0"
K = "-0.1_8.0"
Rp = 1.0
xTy = 5.0
seed = 1

animate(mode, nPart, phi, noise, K, Rp, xTy, seed)
# animate_multi(mode, nPart, phi, noise, K, Rp, xTy, seed)
# animate_multi_blue(mode, nPart, phi, noise, K, Rp, xTy, seed)

# t0 = time.time()
# for K_avg in K_avg_range:
#     K = str(K_avg) + "_" + str(K_std)
#     animate(mode, nPart, phi, noise, K, Rp, xTy, seed)

# print(time.time()-t0)

del_pos(mode, nPart, phi, noise, K, Rp, xTy, seed)