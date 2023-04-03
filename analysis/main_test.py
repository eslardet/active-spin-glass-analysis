import numpy as np
import analysis_functions_vicsek_new as fun
import os
import matplotlib.pyplot as plt
import sys
import time
import csv

# mode = "C"
# nPart = int(sys.argv[1])
# phi = float(sys.argv[2])
# noise = float(sys.argv[3])
# K = str(sys.argv[4])
# xTy = float(sys.argv[5])
# seed = int(sys.argv[6])

mode = "G"
nPart = 50000
phi = 1.0
noise = "0.70"
# K = "1.0_2.0"
K_avg_range = [1.0]
K_std_range = np.arange(0.0, 2.1, 1.0)
Rp = 1.0
xTy=5.0
seed_range=np.arange(1,5,1)
seed=1
timestep_range=np.arange(0,11,1)

# for t in timestep_range:
#     fun.snapshot(mode, nPart, phi, noise, K, Rp, xTy, seed, pos_ex=False, show_color=True, save_in_folder=False, timestep=t)
fun.plot_average_band_profile(mode=mode, nPart=nPart, phi=phi, noise=noise, K_avg_range=K_avg_range, K_std_range=K_std_range, Rp=Rp, xTy=xTy, seed_range=seed_range, pos_ex=False, timestep_range=timestep_range, min_grid_size=3)

