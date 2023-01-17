import numpy as np
import analysis_functions as fun
import os
import matplotlib.pyplot as plt
import sys

# mode = "C"
# nPart = int(sys.argv[1])
# phi = 0.2
# Pe = float(sys.argv[2])
# K = str(sys.argv[3])
# seed = int(sys.argv[4])
# view_time = float(sys.argv[5])

mode = "C"
nPart = 5000
phi = 0.2
K = 1.0
seed = 1

# fun.snapshot(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, seed=seed, view_time=view_time)
# fun.animate(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, seed=seed)
Pe_range = np.concatenate((np.arange(2.0, 22.0, 2.0), np.arange(25.0, 55.0, 5.0)))

for Pe in Pe_range:
    fun.pos_lowres(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, seed=seed, DT_new=1.0, delete=True)