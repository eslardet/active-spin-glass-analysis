import sys
sys.path.insert(1, './analysis/analysis_functions')
from pt import *

import numpy as np
import os
import matplotlib.pyplot as plt


mode = 'G'
nPart = 10000
phi = 1.0
noise = "0.60"
K_avg_range = np.concatenate((np.round(np.arange(-1.0,0.0,0.1),1), np.round(np.arange(0.0,2.1,0.1),1)))
# K_avg_range_2 = np.arange(0.0,2.1,0.1)
K_std_range = np.round(np.arange(1.0,8.0,1.0),1)
xTy = 5.0
seed_range = np.arange(1,21,1)

fig, ax = plt.subplots()
for K_std in K_std_range:
    p_ss = []
    for K_avg in K_avg_range:
        K = str(K_avg) + "_" + str(K_std)
        p_ss_sum = 0
        for seed in seed_range:
            sim_dir = get_sim_dir(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed)
            p_ss_sum += read_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed)["p_mean"]
        p_ss.append(p_ss_sum/len(seed_range))

    ax.plot([(i-0.6)/K_std for i in K_avg_range], p_ss, '-o', label=r"$K_{STD}=$" + str(K_std))
ax.set_xlabel(r"$(K_{AVG}-0.6)/K_{STD}$")
ax.set_ylabel(r"Polar order parameter, $\Psi$")
ax.set_ylim([0,1])
ax.legend()

folder = os.path.abspath('../plots/p_order_vs_Kratio/')
filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_xTy' + str(xTy) + '.png'
if not os.path.exists(folder):
    os.makedirs(folder)
plt.savefig(os.path.join(folder, filename))