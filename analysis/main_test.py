import numpy as np
import analysis_functions_vicsek as fun
import os
import matplotlib.pyplot as plt
import sys
import time

mode = "C"
nPart = int(sys.argv[1])
phi = float(sys.argv[2])
noise = float(sys.argv[3])
K = str(sys.argv[4])
xTy = float(sys.argv[5])
seed = int(sys.argv[6])


# fun.snapshot(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, xTy=xTy, seed=seed, pos_ex=True)
# fun.animate(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, xTy=xTy, seed=seed)

# fun.write_stats(mode, nPart, phi, Pe, K, xTy, seed, remove_pos=True)
# fun.snapshot(mode, nPart, phi, Pe, K, xTy, seed, pos_ex=True, save_in_folder=True)

fun.plot_porder_time(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed)
fun.write_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed, remove_pos=True)