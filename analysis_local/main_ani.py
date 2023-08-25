import numpy as np
from analysis.analysis_functions import *
import os
import matplotlib.pyplot as plt
import sys

# f.plot_vorder_ksd(mode="G", nPart=500, phi=0.2, KAVG=1.0, KSTD_range=np.arange(0, 41, 1.0), seed=2, save=True, view=False)

# f.snapshot(mode="C", nPart=1000, phi=0.4, K=1.0, seed=2, view_time=10)


mode = "G"
nPart = 10000
phi = 1.0
noise = "0.20"
K_std = 8.0
# K_avg_range = np.round(np.arange(-0.6,1.1,0.2),1)
K_avg_range = [-0.5,-0.4,0.5]
# K = "0.0_8.0"
Rp = 1.0
xTy = 1.0
seed = 1


# write_stats(mode, nPart, phi, noise, K, Rp, xTy, seed)
# animate(mode, nPart, phi, noise, K, Rp, xTy, seed, max_T=3300)
# plot_porder_time(mode, nPart, phi, noise, K, Rp, xTy, seed)
# snapshot(mode, nPart, phi, noise, K, Rp, xTy, seed, show_color=True)

# for K_avg in K_avg_range:
#     K = str(K_avg) + "_" + str(K_std)
#     animate(mode, nPart, phi, noise, K, Rp, xTy, seed, min_T=3000, max_T=3500)
