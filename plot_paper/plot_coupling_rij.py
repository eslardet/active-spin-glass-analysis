import numpy as np
import sys
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import csv
# sys.path.insert(0, os.path.abspath('../analysis/'))
from analysis_functions_vicsek_new import *    

bin_size=100
bin_ratio=2
K_max=30
r_max=2

small = 18
big = 28

plt.rc('font', size=big)          # controls default text sizes
plt.rc('axes', labelsize=big)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
plt.rc('legend', fontsize=big)    # legend fontsize

matplotlib.rcParams["font.family"] = "serif"
plt.rcParams['text.usetex'] = True

filename = 'plot_paper/coupling_rij.txt'

K_list = []
rij_list = []
save_file = open(filename, "r")
for line in save_file:
    line = line.split("\t")
    K_list.append(float(line[0]))
    rij_list.append(float(line[1].strip()))

## If wanting to shift K_avg to origin
# K_list = [k - K_avg for k in K_list]

fig, ax = plt.subplots(figsize=(10,10/bin_ratio))

cmap = cm.plasma
norm = colors.Normalize(vmin=0, vmax=1)
h = ax.hist2d(K_list, rij_list, bins=(bin_size, int(bin_size/bin_ratio)), range=[[-K_max,K_max], [0,r_max]], cmap=cmap, density=True)
# cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))
# cbar = fig.colorbar(h[3], ax=ax, label="Density")
cbar = plt.colorbar(h[3])
cbar.ax.get_yaxis().labelpad = 40
cbar.ax.set_ylabel('Density', rotation=270)
ax.set_ylim(bottom=0)
ax.set_xlabel(r"$K_{ij}$")
ax.set_ylabel(r"$r_{ij}$")

folder = os.path.abspath('../plots/for_figures/coupling_rij')
filename =  'coupling_rij_hist.pdf'
if not os.path.exists(folder):
    os.makedirs(folder)
plt.savefig(os.path.join(folder, filename), bbox_inches="tight")

# plt.show()