import numpy as np
import analysis_functions as fun
import os
import matplotlib.pyplot as plt
import sys

mode = "C"
nPart = int(sys.argv[1])
phi = 0.2
Pe = float(sys.argv[2])
K = str(sys.argv[3])
seed = int(sys.argv[4])
view_time = float(sys.argv[5])

# fun.snapshot(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, seed=seed, view_time=view_time)
# fun.animate(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, seed=seed)

fun.pos_lowres(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, seed=seed, DT_new=1.0, delete=False)