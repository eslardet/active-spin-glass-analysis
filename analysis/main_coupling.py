import numpy as np
import analysis_functions_vicsek as fun
import os
import matplotlib.pyplot as plt
import sys
import time


mode = str(sys.argv[1])
nPart = int(sys.argv[2])
phi = float(sys.argv[3])
noise = sys.argv[4]
K = str(sys.argv[5]) + "_" + str(sys.argv[6])
xTy = float(sys.argv[7])
seed = int(sys.argv[8])
simulT = float(sys.argv[9])

# fun.plot_dist_coupling_hist(mode, nPart, phi, noise, K, xTy, seed, bin_size=160, bin_ratio=16, r_max=5)
fun.plot_dist_coupling_hist_diff(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed, bin_size=50)
fun.del_files(mode, nPart, phi, noise, K, xTy, seed, files=["coupling", "initpos", "pos"])

