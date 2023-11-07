import sys
sys.path.insert(1, './analysis/analysis_functions')
from stats import *

import numpy as np
import os
import matplotlib.pyplot as plt
import time


mode = 'T'
nPart = 1000
phi = 1.0
noise = "0.20"
K = "1.0_-1.0_-1.0"
Rp=1.0
xTy = 1.0
seed = 1
# simulT = float(sys.argv[9])


# snapshot(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, xTy=xTy, seed=seed, pos_ex=True)
# animate(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, xTy=xTy, seed=seed)

# write_stats(mode, nPart, phi, Pe, K, xTy, seed, remove_pos=True)
# snapshot(mode, nPart, phi, noise, K, xTy, seed, pos_ex=True, save_in_folder=True)

##plot_porder_time(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed)
write_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, remove_pos=False)
# sim_dir = get_sim_dir(mode, nPart, phi, noise, K, xTy, seed)
# os.remove(os.path.join(sim_dir, "initpos"))

