import numpy as np
sys.path.insert(1, '/Users/el2021/Code/2D_ActiveSpinGlass_EL/Active_Spin_Glass/analysis')
from analysis_functions import *
import os
import matplotlib.pyplot as plt
import sys
import csv


mode = "F"
nPart = 1000
phi = 1.0
noise = "0.20"
K = "0.0_8.0_Kn-8.0"
Rp = 1.0
xTy = 1.0
seed = 1

snapshot(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, pos_ex=True)