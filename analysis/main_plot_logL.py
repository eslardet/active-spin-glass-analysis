
import numpy as np
from analysis_functions import *
import os
import matplotlib.pyplot as plt
import sys
import time

mode = 'G'
#nPart=10000
nPart_range = [1000,2500,6400,10000,22500,40000]
phi = 1.0
noise = '0.20'
noise_range = ["0.20"]
K_avg = 0.1
K_std = 8.0
xTy = 1.0
Rp = 1.0
Rp_range = [1.0]
seed_range = np.arange(1,51,1)
save_data = True
y_log = False

for K_avg in [0.1, 1.0]:
    K = str(K_avg) + "_" + str(K_std)
    plot_porder_logL(mode=mode, nPart_range=nPart_range, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed_range=seed_range, save_data=save_data, y_log=y_log)

