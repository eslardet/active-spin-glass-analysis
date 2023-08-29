import numpy as np
sys.path.insert(1, '/Users/el2021/Code/2D_ActiveSpinGlass_EL/Active_Spin_Glass/analysis')
from analysis_functions import *
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
nPart = 100
phi = 1.0
noise = "0.20"
# K = "-1.0_8.0"
Rp = 1.0
xTy = 1.0
seed_range = [1]
bin_size = 100
bin_ratio = 2
r_max = 2
K_max = 30

K_avg = 0.0
# K_avg_compare = -1.0
K_std = 8.0


# snapshot(mode, nPart, phi, noise, K, Rp, xTy, seed, pos_ex=True)
plot_dist_coupling_hist(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed_range, bin_size=bin_size, bin_ratio=bin_ratio, r_max=r_max, K_max=K_max, pos_ex=True, init_pos=False)
# plot_dist_coupling_hist_diff(mode, nPart, phi, noise, K_avg, K_avg_compare, K_std, Rp, xTy, seed, bin_size=bin_size, bin_ratio=bin_ratio, r_max=r_max, K_max=K_max)
# plot_dist_coupling_hist_diff_init(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed, bin_size=bin_size, bin_ratio=bin_ratio, r_max=r_max, K_max=K_max)
# del_files(mode, nPart, phi, noise, K, xTy, seed, files=["coupling", "initpos", "pos"])

