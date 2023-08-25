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
K = str(sys.argv[5]) + "_" + str(sys.argv[6])
Rp = sys.argv[7]
xTy = float(sys.argv[8])
seed = int(sys.argv[9])
seed_range = np.arange(1,seed+1,1)

r_scale = str(sys.argv[10])

d_type = str(sys.argv[11])
max_time = float(sys.argv[12])
timestep_range = np.arange(0, max_time+1, 1)
corr_r_min = float(sys.argv[13])
corr_r_max = float(sys.argv[14])
r_bin_num = int(sys.argv[15])

t0 = time.time()

write_corr_vel(mode, nPart, phi, noise, K, Rp, xTy, seed_range, r_scale, timestep_range, d_type, corr_r_min=corr_r_min, corr_r_max=corr_r_max, r_bin_num=r_bin_num)
plot_corr_vel_file(mode, nPart, phi, noise, K, Rp, xTy, seed_range, d_type, x_scale=r_scale, y_scale="log")

print("Time taken: " + str(time.time() - t0))
