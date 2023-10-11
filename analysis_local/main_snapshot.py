import numpy as np
import sys
sys.path.insert(1, '/Users/el2021/Code/2D_ActiveSpinGlass_EL/Active_Spin_Glass/analysis')
from analysis_functions import *
import os
import matplotlib.pyplot as plt
import csv


mode = "G"
nPart = 10000
phi = 1.0
noise = "0.20"
K = "-1.0_8.0"
Rp = 1.0
xTy = 1.0
seed = 1

for K_std in np.arange(0.0, 8.1, 1.0):
    K = "0.0_" + str(K_std)
    snapshot(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, pos_ex=True)