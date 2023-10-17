import numpy as np
import sys
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import csv
sys.path.insert(1, './analysis/analysis_functions')
from coupling import *
from correlation import *


small = 22
big = 28

plt.rc('font', size=small)          # controls default text sizes
plt.rc('axes', labelsize=small)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
plt.rc('legend', fontsize=small)    # legend fontsize

matplotlib.rcParams["font.family"] = "serif"
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.labelpad']=10

# file = 'plot_paper/data/coupling_rij.txt'

bin_size=100
bin_ratio=2
K_max=4
r_max=2

fig, axs = plt.subplots(3,1, figsize=(8,12))
counter = 0
labels = ["(a)", "(b)", "(c)"]
for K_std in [1.0, 8.0]:
    ax = axs[counter]
    filename = 'G_N10000_phi1.0_n0.20_K0.0_' + str(K_std) + '_Rp1.0_xTy1.0_hist'
    file = os.path.abspath('../plot_data/dist_coupling/' + filename + '.txt')


    K_list = []
    rij_list = []
    save_file = open(file, "r")
    for line in save_file:
        line = line.split("\t")
        K_list.append(float(line[0])/K_std)
        rij_list.append(float(line[1].strip()))

    ## If wanting to shift K_avg to origin
    # K_list = [k - K_avg for k in K_list]

    cmap = cm.plasma
    norm = colors.Normalize(vmin=0, vmax=0.35)
    h = ax.hist2d(K_list, rij_list, bins=(bin_size, int(bin_size/bin_ratio)), range=[[-K_max,K_max], [0,r_max]], cmap=cmap, norm=norm, density=True)
    # cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))
    # cbar = fig.colorbar(h[3], ax=ax, label="Density")
    # cbar = plt.colorbar(h[3])
    # cbar = fig.colorbar(h[3])
    # cbar.ax.get_yaxis().labelpad = 40
    # cbar.ax.set_ylabel('Density', rotation=270)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(r"$r_{ij}$")
    ax.text(-0.12, 1.0, labels[counter], transform=ax.transAxes, va='top', ha='right')

    counter += 1

ax.set_xlabel(r"$K_{ij}/\sigma_K$")
# ax.set_xlim(-5,5)
fig.subplots_adjust(right=0.8)
cbar = fig.add_axes([0.85, 0.4, 0.03, 0.45])
fig.colorbar(h[3], cax=cbar, ticks=[0.0, 0.1, 0.2, 0.3])
cbar.get_yaxis().labelpad = 40
cbar.set_ylabel('Density', rotation=270)

plt.subplots_adjust(hspace = 0.4)
ax = axs[counter]

mode = 'G'
nPart_range = [10000]
phi = 1.0
noise = "0.20"
K_avg_range = [0.0]
K_std_range = [1.0, 8.0]
Rp = 1.0
xTy = 1.0
seed_range = np.arange(1,21,1)
r_scale = "log"

colors = plt.cm.GnBu(np.linspace(0.2, 1, len(K_avg_range)))

i = 0
for nPart in nPart_range:
    for K_std in K_std_range:
        # exponents = []
        for K_avg in K_avg_range:
            K = str(K_avg) + "_" + str(K_std)
            r_plot, corr_bin_av = read_corr_density(mode, nPart, phi, noise, K, Rp, xTy, seed_range, r_scale, bin_ratio)
            # ax.plot(r_plot, np.abs(corr_bin_av), '-', label=r"$\overline{K}=" + str(K_avg) + r"$", color=colors[i])
            ax.plot(r_plot, np.abs(corr_bin_av), '-', label=r"$\sigma_K=" + str(K_std) + r"$")
            ax.text(-0.12, 1.0, labels[counter], transform=ax.transAxes, va='top', ha='right')
            i += 1

ax.set_xscale("log")
ax.set_xlim(left=10**-1, right=30)
ax.set_yscale("log")
ax.set_ylim(bottom=10**-4, top=1.01)
ax.set_xlabel(r"$r$", labelpad=0)
ax.set_ylabel(r"$C_\rho(r)$", labelpad=10)
ax.legend(frameon=False)

folder = os.path.abspath('../plots/for_figures/coupling_rij')
filename =  'coupling_rij_hist_subplots_with_corr'
filename += '.pdf'
if not os.path.exists(folder):
    os.makedirs(folder)
plt.savefig(os.path.join(folder, filename), bbox_inches="tight")

# plt.show()