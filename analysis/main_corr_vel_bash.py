import numpy as np
import analysis_functions_vicsek_new as fun
import os
import matplotlib.pyplot as plt
import sys
import time


mode = str(sys.argv[1])
nPart = int(sys.argv[2])
phi = float(sys.argv[3])
noise = sys.argv[4]
K = str(sys.argv[5]) + "_" + str(sys.argv[6])
Rp = sys.argv[7]
xTy = float(sys.argv[8])
seed = int(sys.argv[9])
seed_range = np.arange(1,seed+1,1)

linlin = bool(sys.argv[10])
loglin = bool(sys.argv[11])
loglog = bool(sys.argv[12])

d_type = str(sys.argv[13])
r_max = float(sys.argv[14])
r_bin_num = int(sys.argv[15])


t0 = time.time()
fun.plot_corr_vel(mode, nPart, phi, noise, K, Rp, xTy, seed_range, d_type=d_type, r_max=r_max, r_bin_num=r_bin_num, linlin=linlin, loglin=loglin, loglog=loglog)

print("Time taken: " + str(time.time() - t0))
