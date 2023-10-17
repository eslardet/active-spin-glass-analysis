import sys
sys.path.insert(1, './analysis/analysis_functions')
from stats import *

import time



mode = str(sys.argv[1])
nPart = int(sys.argv[2])
phi = float(sys.argv[3])
noise = sys.argv[4]
K = str(sys.argv[5]) + "_" + str(sys.argv[6]) + "_Kn" + str(sys.argv[7])
Rp = sys.argv[8]
xTy = float(sys.argv[9])
seed = int(sys.argv[10])
simulT = float(sys.argv[11])


write_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, remove_pos=True, moments=False)
