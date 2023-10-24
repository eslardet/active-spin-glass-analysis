import sys
sys.path.insert(1, './analysis_functions')
from cohesion import *

import time


mode = str(sys.argv[1])
nPart = int(sys.argv[2])
phi = float(sys.argv[3])
noise = sys.argv[4]
K = str(sys.argv[5]) + "_" + str(sys.argv[6])
Rp = float(sys.argv[7])
xTy = float(sys.argv[8])
seed = int(sys.argv[9])


t0 = time.time()
write_cohesion_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)

print("Time elapsed in seconds: ", time.time() - t0)



