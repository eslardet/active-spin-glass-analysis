import numpy as np
from analysis.analysis_functions import *
import os
import matplotlib.pyplot as plt
import sys


mode = 'G'
nPart = 10000
phi = 1.0
noise_range_1 = ["0.20", "0.40", "0.60"]
noise_range_2 = ["0.80"]
K_avg_range_1 = np.concatenate((np.arange(-1.0,0.0,0.1), np.arange(0.0,2.1,1.0)))
K_avg_range_2 = np.arange(0.0,2.1,0.1)
K_std_range = [8.0]
xTy = 5.0
seed_range = np.arange(1,21,1)


fig, ax = plt.subplots()
fig, ax = plot_porder_Kavg_ax(mode, nPart, phi, noise_range_1, K_avg_range_1, K_std_range, xTy, seed_range, fig, ax)
fig, ax = plot_porder_Kavg_ax(mode, nPart, phi, noise_range_2, K_avg_range_2, K_std_range, xTy, seed_range, fig, ax)

ax.set_xlim([-1,2])

folder = os.path.abspath('../plots/p_order_vs_Kavg/')
filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_Kstd' + str(8.0) + '_xTy' + str(xTy) + '.png'
if not os.path.exists(folder):
    os.makedirs(folder)
plt.savefig(os.path.join(folder, filename))
