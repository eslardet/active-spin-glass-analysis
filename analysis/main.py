import numpy as np
import analysis_functions_lattice as fun
import os
import matplotlib.pyplot as plt
import sys

# mode = sys.argv[1]
# nPart = int(sys.argv[2])
# K = str(sys.argv[3])
# Rp = float(sys.argv[4])
# rotD = float(sys.argv[5])
# seed = int(sys.argv[6])
# view_time = float(sys.argv[7])

mode = "C"
nPart = 1024
Rp = 1.0
rotD = 1.0
K = 0.0
seed = 1

# fun.snapshot(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, seed=seed, view_time=view_time)
fun.animate(mode=mode, nPart=nPart, K=K, Rp=Rp, rotD=rotD, seed=seed)

fun.plot_porder_time(mode=mode, nPart=nPart, K=K, Rp=Rp, rotD=rotD, seed=seed)
fun.plot_norder_time(mode=mode, nPart=nPart, K=K, Rp=Rp, rotD=rotD, seed=seed)

# fun.write_stats(mode=mode, nPart=nPart, K=K, Rp=Rp, rotD=rotD, seed=seed, min_T=10.0, remove_pos=True)
# Pe_range = np.concatenate((np.arange(2.0, 22.0, 2.0), np.arange(25.0, 55.0, 5.0)))

# for Pe in Pe_range:
#     fun.pos_lowres(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, seed=seed, DT_new=1.0, delete=True)
