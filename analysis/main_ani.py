import numpy as np
import analysis_functions_vicsek_new as fun
import os
import matplotlib.pyplot as plt
import sys

# f.plot_vorder_ksd(mode="G", nPart=500, phi=0.2, KAVG=1.0, KSTD_range=np.arange(0, 41, 1.0), seed=2, save=True, view=False)

# f.snapshot(mode="C", nPart=1000, phi=0.4, K=1.0, seed=2, view_time=10)


mode = "G"
nPart = 100
phi = 1.0
noise = "0.20"
K = "1.0_1.0"
Rp = 1.0
xTy = 1.0
seed = 1

fun.animate(mode, nPart, phi, noise, K, Rp, xTy, seed)
# fun.snapshot_pos_ex(mode, nPart, phi, Pe, K, seed, show_color=False)
