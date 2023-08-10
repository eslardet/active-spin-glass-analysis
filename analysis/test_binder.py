import numpy as np
import analysis_functions_vicsek_new as fun
import matplotlib.pyplot as plt
import time
import scipy.stats as sps
import os
import freud
import analysis_functions_vicsek_new as fun
from matplotlib import cm, colors



mode = "G"
nPart = 10000
phi = 1.0
# noise_range = [format(i, '.3f') for i in np.arange(0.80,0.811,0.01)]
noise = "0.20"
# noise_range = np.arange(0.80,0.01,0.82)
Rp = 1.0
K = "2.0_8.0"
xTy = 1.0
seed = 1



posFileExact = fun.get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, file_name="pos_exact")
L = np.sqrt(nPart/phi)
x, y, theta, view_time = fun.get_pos_ex_snapshot(posFileExact)
print(fun.centre_of_mass(x,L))
print(fun.centre_of_mass(y,L))
print(fun.mean_dist_com(posFileExact, L))
fun.snapshot(mode, nPart, phi, noise, K, Rp, xTy, seed, pos_ex=True)

# print(fun.neighbour_stats(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max=1))