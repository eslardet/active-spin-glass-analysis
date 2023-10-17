import sys
sys.path.insert(1, './analysis/analysis_functions')
from import_files import *

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

del_pos(mode, nPart, phi, noise, K, Rp, xTy, seed)
