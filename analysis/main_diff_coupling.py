import numpy as np
from analysis_functions import *
import os
import matplotlib.pyplot as plt
import sys
import time


mode = str(sys.argv[1])
nPart = int(sys.argv[2])
phi = float(sys.argv[3])
noise = sys.argv[4]
K_avg = float(sys.argv[5])
K_std = float(sys.argv[6])
#K = str(sys.argv[5]) + "_" + str(sys.argv[6])
Rp = float(sys.argv[7])
xTy = float(sys.argv[8])
seed = int(sys.argv[9])
seed_range = np.arange(1, seed+1, 1)
bin_size = int(sys.argv[10])
bin_ratio = int(sys.argv[11])
r_max = float(sys.argv[12])
#r_max = None
K_max = float(sys.argv[13])
#K_max = None
#init_pos=bool(int(sys.argv[14]))
#pos_ex=bool(int(sys.argv[15]))

#print(init_pos)
#plot_dist_coupling_hist(mode=mode, nPart=nPart, phi=phi, noise=noise, K_avg=K_avg, K_std=K_std, Rp=Rp, xTy=xTy, seed_range=seed_range, bin_size=bin_size, bin_ratio=bin_ratio, r_max=r_max, K_max=K_max, init_pos=init_pos, pos_ex=pos_ex)

#plot_dist_coupling_hist_diff(mode=mode, nPart=nPart, phi=phi, noise=noise, K_avg=K_avg, K_avg_compare=K_avg_compare, K_std=K_std, Rp=Rp, xTy=xTy, seed=seed, bin_size=bin_size, bin_ratio=bin_ratio, r_max=r_max, K_max=K_max)

plot_dist_coupling_hist_diff_init(mode=mode, nPart=nPart, phi=phi, noise=noise, K_avg=K_avg, K_std=K_std, Rp=Rp, xTy=xTy, seed_range=seed_range, bin_size=bin_size, bin_ratio=bin_ratio, r_max=r_max, K_max=K_max)

#K = str(K_avg) + "_" + str(K_std)

#for seed in seed_range:
#    del_files(mode, nPart, phi, noise, K, Rp, xTy, seed, files=["coupling"])

