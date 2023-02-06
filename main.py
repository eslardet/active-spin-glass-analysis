import numpy as np
import analysis_functions as fun
import os
import matplotlib.pyplot as plt
import sys

mode = sys.argv[1]
nPart = int(sys.argv[2])
phi = float(sys.argv[3])
Pe = float(sys.argv[4])
K = str(sys.argv[5])
seed = int(sys.argv[6])
##view_time = float(sys.argv[7])

#mode = "C"
#nPart = 6000
#phi = 0.1
#Pe = 10.0
#K = 10.0
#seed = 1

# fun.snapshot(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, seed=seed, view_time=view_time)
fun.animate(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, seed=seed)
fun.snapshot_pos_ex(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, seed=seed)
# fun.pos_lowres(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, seed=seed, DT_new=0.1, delete=False)

# Pe_range = np.concatenate((np.arange(2.0, 22.0, 2.0), np.arange(25.0, 55.0, 5.0)))

# for Pe in Pe_range:
#     fun.pos_lowres(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, seed=seed, DT_new=1.0, delete=True)
