import numpy as np
import analysis_functions_vicsek_nd as fun
import os
import matplotlib.pyplot as plt
import sys
import time

mode = sys.argv[1]
nPart = int(sys.argv[2])
phi = float(sys.argv[3])
Pe = float(sys.argv[4])
K = str(sys.argv[5])
xTy = float(sys.argv[6])
seed = int(sys.argv[7])


# fun.snapshot(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, xTy=xTy, seed=seed, pos_ex=True)
# fun.animate(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, xTy=xTy, seed=seed)

fun.write_stats(mode, nPart, phi, Pe, K, xTy, seed, remove_pos=True)
fun.snapshot(mode, nPart, phi, Pe, K, xTy, seed, pos_ex=True, save_in_folder=True)
