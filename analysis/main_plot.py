import numpy as np
import analysis_functions_vicsek as fun
import os
import matplotlib.pyplot as plt
import sys

mode = 'C'
nPart = 5000
phi = 0.2
noise_range = np.arange(0.1,1.1,1.0)
K = 1.0
xTy = 5.0
seed_range = [1]


noise_range = np.arange(0.1,1.1,0.1)

fun.plot_porder_noise(mode=mode, nPart=nPart, phi=phi, noise_range=noise_range, K=K, xTy=xTy, seed_range=[2])

