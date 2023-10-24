import sys
sys.path.insert(1, './analysis_functions')
from stats import *
from visuals import *

import time



mode = str(sys.argv[1])
nPart = int(sys.argv[2])
phi = float(sys.argv[3])
noise = sys.argv[4]
K = str(sys.argv[5]) + "_" + str(sys.argv[6])
Rp = sys.argv[7]
xTy = float(sys.argv[8])
seed = int(sys.argv[9])
simulT = float(sys.argv[10])


##plot_porder_time(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed)
write_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, remove_pos=False, moments=False)

snapshot(mode, nPart, phi, noise, K, Rp, xTy, seed, pos_ex=True)