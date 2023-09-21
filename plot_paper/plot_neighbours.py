import numpy as np
import sys
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import csv
# sys.path.insert(0, os.path.abspath('../analysis/'))
sys.path.insert(1, '/Users/el2021/Code/2D_ActiveSpinGlass_EL/Active_Spin_Glass/analysis')
from analysis_functions import *    

mode = "G"
nPart = 10000
phi = 1.0
noise = "0.20"
K = "0.0_8.0"
Rp = 1.0
xTy = 1.0
seed = 1

r_max = 1
pos_ex = False
timestep_range = np.arange(0,6,1)

small = 18
medium = 22
big = 28

plt.rc('font', size=medium)          # controls default text sizes
plt.rc('axes', labelsize=medium)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=medium)    # fontsize of the tick labels
plt.rc('ytick', labelsize=medium)    # fontsize of the tick labels
plt.rc('legend', fontsize=medium)    # legend fontsize

matplotlib.rcParams["font.family"] = "serif"
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.labelpad']=10

fig, ax = plt.subplots(figsize=(8,4))

# K="0.0_0.0"
# n_nei = neighbour_counts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, pos_ex, timestep_range)
# print(neighbour_stats(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, pos_ex, timestep_range))
# # print(len(av_nei_i))
# unique, counts = np.unique(n_nei, return_counts=True)
# ax.bar(unique, counts/(nPart*len(timestep_range)), alpha=0.5, label=r"$\sigma_K=0.0$")

# K="0.0_1.0"
# n_nei = neighbour_counts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, pos_ex, timestep_range)
# print(neighbour_stats(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, pos_ex, timestep_range))
# # print(len(av_nei_i))
# unique, counts = np.unique(n_nei, return_counts=True)
# # ax.bar(unique, counts/(nPart*len(timestep_range)), alpha=0.5, label=r"$\sigma_K=1$")
# ax.plot(unique, counts/(nPart*len(timestep_range)), "-", label=r"$\sigma_K=1$")


# K="0.0_8.0"
# n_nei = neighbour_counts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, pos_ex, timestep_range)
# print(neighbour_stats(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, pos_ex, timestep_range))
# unique, counts = np.unique(n_nei, return_counts=True)
# # ax.bar(unique, counts/(nPart*len(timestep_range)), alpha=0.5, label=r"$\sigma_K=8$")
# ax.plot(unique, counts/(nPart*len(timestep_range)), "-", label=r"$\sigma_K=8$")

K_std_range = [1.0, 8.0]
# K_std_range = np.arange(1.0, 8.1,1.0)

colors = plt.cm.BuPu(np.linspace(0.2, 1, len(K_std_range)))

for K_std in K_std_range:
    K = "0.0_" + str(K_std)
    n_nei = neighbour_counts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, pos_ex, timestep_range)
    unique, counts = np.unique(n_nei, return_counts=True)
    # ax.plot(unique, counts/(nPart*len(timestep_range)), "-", label=r"$\sigma_K=" + str(round(K_std)) + r"$", color=colors[bisect.bisect(K_std_range, K_std)-1])
    ax.bar(unique, counts/(nPart*len(timestep_range)), alpha=0.5, label=r"$\sigma_K=" + str(round(K_std)) + r"$")
ax.set_xlabel(r"$n_i$", labelpad=0)
# ax.set_yticklabels([0.0,0.1,0.2])
ax.set_ylabel(r"$P(n_i)$")
ax.legend(frameon=False)
# ax.set_title(r"$N=$" + str(nPart) + r"; $\rho=$" + str(phi) + r"; $\eta=$" + str(noise) + r"; $K=$" + str(K) + r"; $r_{max}=$" + str(r_max))

ax.set_xlim([0,25])
ax.set_ylim([0,0.20])

# yticks = ax.yaxis.get_major_ticks()
# for i in range(1,8,2):
#     yticks[i].set_visible(False)

folder = os.path.abspath('../plots/for_figures/neighbour_hist')
filename =  'superimposed.svg'
if not os.path.exists(folder):
    os.makedirs(folder)
plt.savefig(os.path.join(folder, filename), bbox_inches="tight")

plt.show()