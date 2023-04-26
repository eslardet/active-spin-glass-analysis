import numpy as np
import analysis_functions_vicsek_new as fun
import os
import matplotlib.pyplot as plt
import sys
import time


# mode = str(sys.argv[1])
# nPart = int(sys.argv[2])
# phi = float(sys.argv[3])
# noise = sys.argv[4]
# K = str(sys.argv[5]) + "_" + str(sys.argv[6])
# xTy = float(sys.argv[7])
# seed = int(sys.argv[8])
# simulT = float(sys.argv[9])

mode = "G"
nPart = 1000
phi = 1.0
noise = "0.20"
K = "0.0_8.0"
Rp = 2.0
xTy = 1.0
seed = 1
bin_size = 160
bin_ratio = 16
r_max = 4

K_avg = 0.0
K_avg_compare = -1.0
K_std = 8.0

fun.plot_dist_coupling_hist(mode, nPart, phi, noise, K, Rp, xTy, seed, bin_size=bin_size, bin_ratio=bin_ratio, r_max=r_max, pos_ex=False, timestep_range=np.arange(5,9,1))
# fun.plot_dist_coupling_hist_diff(mode, nPart, phi, noise, K_avg, K_avg_compare, K_std, Rp, xTy, seed, bin_size=bin_size, bin_ratio=bin_ratio, r_max=r_max)
# fun.del_files(mode, nPart, phi, noise, K, xTy, seed, files=["coupling", "initpos", "pos"])

