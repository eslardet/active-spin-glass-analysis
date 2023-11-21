import sys
sys.path.insert(1, './analysis/analysis_functions')
from visuals import *

import numpy as np
import os
import matplotlib.pyplot as plt
import time


# f.plot_vorder_ksd(mode="G", nPart=500, phi=0.2, KAVG=1.0, KSTD_range=np.arange(0, 41, 1.0), seed=2, save=True, view=False)

# f.snapshot(mode="C", nPart=1000, phi=0.4, K=1.0, seed=2, view_time=10)


mode = "T"
nPart = 1000
phi = 1.0
noise = "0.20"
KAA = 0.0
KBB = 0.0
KCC = 0.0
KAB = 1.0
KBA = -1.0
KBC = 1.0
KCB = -1.0
KCA = 1.0
KAC = -1.0
K = str(KAA) + "_" + str(KBB) + "_" + str(KCC) + "_" + str(KAB) + "_" + str(KBA) + "_" + str(KBC) + "_" + str(KCB) + "_" + str(KCA) + "_" + str(KAC)
Rp = 1.0
xTy = 1.0
seed = 1

animate_multi(mode, nPart, phi, noise, K, Rp, xTy, seed)
# animate_multi_blue(mode, nPart, phi, noise, K, Rp, xTy, seed)

# t0 = time.time()
# for K_avg in K_avg_range:
#     K = str(K_avg) + "_" + str(K_std)
#     animate(mode, nPart, phi, noise, K, Rp, xTy, seed)

# print(time.time()-t0)

# del_pos(mode, nPart, phi, noise, K, Rp, xTy, seed)