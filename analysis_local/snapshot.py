import sys
sys.path.insert(1, './analysis/analysis_functions')
from visuals import *

import numpy as np
import os
import matplotlib.pyplot as plt
import csv


mode = "G"
nPart = 100000
phi = 1.0
noise = "0.20"
K = "0.0_4.0"
Rp = 1.0
xTy = 5.0
seed = 1

for K_avg in [1.0]:
    K = str(K_avg) + "_8.0"
    snapshot(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, pos_ex=True, neigh_col=True)