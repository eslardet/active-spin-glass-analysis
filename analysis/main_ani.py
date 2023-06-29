import numpy as np
import analysis_functions_vicsek_new as fun
import os
import matplotlib.pyplot as plt
import sys

# f.plot_vorder_ksd(mode="G", nPart=500, phi=0.2, KAVG=1.0, KSTD_range=np.arange(0, 41, 1.0), seed=2, save=True, view=False)

# f.snapshot(mode="C", nPart=1000, phi=0.4, K=1.0, seed=2, view_time=10)


mode = "A"
nPart = 10000
phi = 1.0
noise = "0.20"
K_std = 8.0
K_avg_range = np.round(np.arange(0.6,0.9,0.2),1)
# K_avg_range = [-0.6]
# K = "0.0_8.0"
Rp = 1.0
xTy = 1.0
seed = 1


# fun.write_stats(mode, nPart, phi, noise, K, Rp, xTy, seed)
# fun.animate(mode, nPart, phi, noise, K, Rp, xTy, seed, max_T=3300)
# fun.plot_porder_time(mode, nPart, phi, noise, K, Rp, xTy, seed)
# fun.snapshot(mode, nPart, phi, noise, K, Rp, xTy, seed, show_color=True)

for K_avg in K_avg_range:
    K = str(K_avg) + "_" + str(K_std)
    fun.animate(mode, nPart, phi, noise, K, Rp, xTy, seed, min_T=3000, max_T=3300)
