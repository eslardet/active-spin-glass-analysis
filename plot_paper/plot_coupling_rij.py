import numpy as np
import sys
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import csv
sys.path.insert(1, './analysis/analysis_functions')
from coupling import *


small = 22
big = 28

plt.rc('font', size=big)          # controls default text sizes
plt.rc('axes', labelsize=big)    # fontsize of the x and y labels
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

K_std = 8.0

for K_std in [3.0, 8.0]:
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

    fig, ax = plt.subplots(figsize=(10,10/bin_ratio))

    cmap = cm.plasma
    norm = colors.Normalize(vmin=0, vmax=0.35)
    h = ax.hist2d(K_list, rij_list, bins=(bin_size, int(bin_size/bin_ratio)), range=[[-K_max,K_max], [0,r_max]], cmap=cmap, norm=norm, density=True)
    # cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))
    # cbar = fig.colorbar(h[3], ax=ax, label="Density")
    # cbar = plt.colorbar(h[3])
    cbar = fig.colorbar(h[3])
    cbar.ax.get_yaxis().labelpad = 40
    cbar.ax.set_ylabel('Density', rotation=270)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$K_{ij}/\sigma_K$")
    ax.set_ylabel(r"$r_{ij}$")
    # ax.set_xlim(-5,5)

    folder = os.path.abspath('../plots/for_figures/coupling_rij')
    # filename =  'coupling_rij_hist.pdf'
    filename += '.pdf'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename), bbox_inches="tight")

# plt.show()