import numpy as np
import sys
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import csv
sys.path.insert(1, './analysis/analysis_functions')
from neighbours_density import *

mode = "G"
nPart = 1000
phi = 0.1
noise = "0.20"
K = "0.0_8.0"
Rp = 10.0
xTy = 1.0
seed = 1

r_max = 10
pos_ex = True
timestep_range = np.arange(0,1,1)

small = 18
big = 28

plt.rc('font', size=big)          # controls default text sizes
plt.rc('axes', labelsize=big)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
plt.rc('legend', fontsize=big)    # legend fontsize

matplotlib.rcParams["font.family"] = "serif"
plt.rcParams['text.usetex'] = True

fig, ax = plt.subplots(figsize=(10,5))

# K="0.0_0.0"
# n_nei = neighbour_counts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, pos_ex, timestep_range)
# print(neighbour_stats(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, pos_ex, timestep_range))
# # print(len(av_nei_i))
# unique, counts = np.unique(n_nei, return_counts=True)
# ax.bar(unique, counts/(nPart*len(timestep_range)), alpha=0.5, label=r"$K_{STD}=0.0$")

# K="0.0_1.0"
# n_nei = neighbour_counts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, pos_ex, timestep_range)
# print(neighbour_stats(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, pos_ex, timestep_range))
# # print(len(av_nei_i))
# unique, counts = np.unique(n_nei, return_counts=True)
# ax.bar(unique, counts/(nPart*len(timestep_range)), alpha=0.5, label=r"$K_{STD}=1.0$")
# # ax.plot(unique, counts/(nPart*len(timestep_range)), "-", label=r"$K_{STD}=1.0$")
K="0.0_8.0"
Rp=2.0
r_max=2
n_nei = neighbour_counts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, pos_ex, timestep_range)
print(neighbour_stats(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, pos_ex, timestep_range))
unique, counts = np.unique(n_nei, return_counts=True)
ax.bar(unique, counts/(nPart*len(timestep_range)), alpha=0.5, label=r"$R_p=2$")

Rp=10.0
r_max=10
n_nei = neighbour_counts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, pos_ex, timestep_range)
print(neighbour_stats(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, pos_ex, timestep_range))
unique, counts = np.unique(n_nei, return_counts=True)
ax.bar(unique, counts/(nPart*len(timestep_range)), alpha=0.5, label=r"$R_p=10$")
# ax.plot(unique, counts/(nPart*len(timestep_range)), "--", label=r"$K_{STD}=8.0$")
ax.set_xlabel(r"$n_i$")
# ax.set_yticklabels([0.0,0.1,0.2])
ax.set_ylabel("Density")
ax.legend()
# ax.set_title(r"$N=$" + str(nPart) + r"; $\rho=$" + str(phi) + r"; $\eta=$" + str(noise) + r"; $K=$" + str(K) + r"; $r_{max}=$" + str(r_max))

ax.set_xlim([0,25])
# ax.set_ylim([0,0.20])

# yticks = ax.yaxis.get_major_ticks()
# for i in range(1,8,2):
#     yticks[i].set_visible(False)

folder = os.path.abspath('../plots/for_figures/neighbour_hist')
filename =  'superimposed.svg'
if not os.path.exists(folder):
    os.makedirs(folder)
# plt.savefig(os.path.join(folder, filename), bbox_inches="tight")

plt.show()