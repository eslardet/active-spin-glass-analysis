import numpy as np
import analysis_functions_vicsek_rep as fun
import os
import matplotlib.pyplot as plt
import sys
import time


mode = str(sys.argv[1])
nPart = int(sys.argv[2])
phi = float(sys.argv[3])
noise = sys.argv[4]
K = str(sys.argv[5])
#K = str(sys.argv[5]) + "_" + str(sys.argv[6])
xTy = float(sys.argv[7])
seed = int(sys.argv[8])
simulT = float(sys.argv[9])


fun.snapshot(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed, pos_ex=True)
fun.animate(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed)

# fun.write_stats(mode, nPart, phi, Pe, K, xTy, seed, remove_pos=True)
# fun.snapshot(mode, nPart, phi, noise, K, xTy, seed, pos_ex=True, save_in_folder=True)

#fun.plot_porder_time(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed)
fun.write_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed, remove_pos=True)
#sim_dir = fun.get_sim_dir(mode, nPart, phi, noise, K, xTy, seed)
#os.remove(os.path.join(sim_dir, "initpos"))
