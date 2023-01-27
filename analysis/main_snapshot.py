import numpy as np
import analysis_functions_lattice as fun
import os
import matplotlib.pyplot as plt
import sys
import csv

# f.plot_vorder_ksd(mode="G", nPart=500, phi=0.2, KAVG=1.0, KSTD_range=np.arange(0, 41, 1.0), seed=2, save=True, view=False)

# f.snapshot(mode="C", nPart=1000, phi=0.4, K=1.0, seed=2, view_time=10)

# fun.snapshot(mode="C", nPart=5000, phi=0.2, Pe=2.0, K="1.0", seed=1, view_time=20)

# mode = "C"
# nPart = sys.argv[1]
# phi = 0.2
# Pe = sys.argv[2]
# K = str(sys.argv[3])
# seed = sys.argv[4]
view_time=0

mode = "C"
nPart = 100
K = 1.0
Rp = 2.0
seed = 1

fun.snapshot(mode=mode, nPart=nPart, K=K, Rp=Rp, seed=seed, view_time=view_time)

