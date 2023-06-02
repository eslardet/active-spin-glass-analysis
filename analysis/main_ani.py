import numpy as np
import analysis_functions_vicsek_new as fun
import os
import matplotlib.pyplot as plt
import sys

# f.plot_vorder_ksd(mode="G", nPart=500, phi=0.2, KAVG=1.0, KSTD_range=np.arange(0, 41, 1.0), seed=2, save=True, view=False)

# f.snapshot(mode="C", nPart=1000, phi=0.4, K=1.0, seed=2, view_time=10)


mode = "G"
nPart = 10000
phi = 1.0
noise = "0.20"
K = "0.0_8.0"
Rp = 1.0
xTy = 1.0
seed = 1


# fun.write_stats(mode, nPart, phi, noise, K, Rp, xTy, seed)
fun.animate(mode, nPart, phi, noise, K, Rp, xTy, seed, max_T=3300)
# fun.plot_porder_time(mode, nPart, phi, noise, K, Rp, xTy, seed)
# fun.snapshot(mode, nPart, phi, noise, K, Rp, xTy, seed, show_color=True)

# for K in [1.1]:
#     fun.write_stats(mode, nPart, phi, noise, K, Rp, xTy, seed, remove_pos=True, moments=True)
#     print(fun.get_binder(mode, nPart, phi, noise, K, Rp, xTy, seed_range=[seed]))
