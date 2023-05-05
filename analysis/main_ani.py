import numpy as np
import analysis_functions_vicsek_new as fun
import os
import matplotlib.pyplot as plt
import sys

# f.plot_vorder_ksd(mode="G", nPart=500, phi=0.2, KAVG=1.0, KSTD_range=np.arange(0, 41, 1.0), seed=2, save=True, view=False)

# f.snapshot(mode="C", nPart=1000, phi=0.4, K=1.0, seed=2, view_time=10)


mode = "G"
nPart = 1000
phi = 1.0
noise = "0.80"
K = "1.0_0.0"
Rp = "I"
xTy = 1.0
seed = 1


# fun.write_stats(mode, nPart, phi, noise, K, Rp, xTy, seed)
# fun.animate(mode, nPart, phi, noise, K, Rp, xTy, seed)
fun.plot_porder_time(mode, nPart, phi, noise, K, Rp, xTy, seed)
# fun.snapshot_pos_ex(mode, nPart, phi, noise, K, Rp, seed, xTy, show_color=True)

# for K in [1.1]:
#     fun.write_stats(mode, nPart, phi, noise, K, Rp, xTy, seed, remove_pos=True, moments=True)
#     print(fun.get_binder(mode, nPart, phi, noise, K, Rp, xTy, seed_range=[seed]))
