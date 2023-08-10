import numpy as np
import sys
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import csv
# sys.path.insert(0, os.path.abspath('../analysis/'))
from analysis_functions_vicsek_new import *    

mode = "G"
nPart = 10000
phi = 1.0
noise = "0.20"
K = "0.0_8.0"
Rp = 1.0
xTy = 1.0
seed = 1

r_max = 1


small = 18
big = 32

plt.rc('font', size=big)          # controls default text sizes
plt.rc('axes', labelsize=big)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
plt.rc('legend', fontsize=small)    # legend fontsize

matplotlib.rcParams["font.family"] = "serif"
plt.rcParams['text.usetex'] = True

av_nei_i = neighbour_counts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max)

fig, ax = plt.subplots(figsize=(8,8), dpi=500)
plt.tight_layout()
# ax.hist(av_nei_i, bins=np.arange(0, np.max(av_nei_i)+1))
unique, counts = np.unique(av_nei_i, return_counts=True)
ax.bar(unique, counts/10000)
ax.set_xlabel(r"$\langle N_i\rangle$")
# ax.set_yticklabels([0.0,0.1,0.2])
ax.set_ylabel("Density")
# ax.set_title(r"$N=$" + str(nPart) + r"; $\rho=$" + str(phi) + r"; $\eta=$" + str(noise) + r"; $K=$" + str(K) + r"; $r_{max}=$" + str(r_max))

ax.set_xlim([0,25])
ax.set_ylim([0,0.20])

yticks = ax.yaxis.get_major_ticks()
for i in range(1,8,2):
    yticks[i].set_visible(False)

folder = os.path.abspath('../plots/for_figures/neighbour_hist')
filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed) + '.png'
if not os.path.exists(folder):
    os.makedirs(folder)
plt.savefig(os.path.join(folder, filename), bbox_inches='tight')