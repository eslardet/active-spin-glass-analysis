import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm, colors
from decimal import Decimal
import freud
import scipy.stats as sps
import bisect
from matplotlib.ticker import AutoMinorLocator

import csv, os


small = 22
big = 28

plt.rc('font', size=big)          # controls default text sizes
plt.rc('axes', labelsize=big)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
plt.rc('legend', fontsize=small)    # legend fontsize

matplotlib.rcParams["font.family"] = "serif"
plt.rcParams['text.usetex'] = True
# plt.rcParams['axes.labelpad']=10

num_Kstd = 1

filename = "binder_pt"
file = os.path.abspath("plot_paper/data/" + filename + ".txt")
with open(file) as f:
    reader = csv.reader(f, delimiter="\n")
    r_pt = list(reader)

filename = "binder_Kstd1_all"
file = os.path.abspath("plot_paper/data/" + filename + ".txt")
with open(file) as f:
    reader = csv.reader(f, delimiter="\n")
    r_binder = list(reader)


fig, ax = plt.subplots(figsize=(10,7))

ax_in = ax.inset_axes([0.15, 0.5, 0.45, 0.45])

for k in range(num_Kstd):

    K_avg = r_pt[3*k+1][0].split('\t')[:-1]
    K_avg_plot = [float(i) for i in K_avg]
    p_ss = r_pt[3*k+2][0].split('\t')[:-1]
    p_ss_plot = [float(i) for i in p_ss]
    ax.plot(K_avg_plot, p_ss_plot, "-o")

    K_avg = r_binder[3*k+1][0].split('\t')[:-1]
    K_avg_plot = [float(i) for i in K_avg]
    p_ss = r_binder[3*k+2][0].split('\t')[:-1]
    p_ss_plot = [float(i) for i in p_ss]
    ax_in.plot(K_avg_plot, p_ss_plot, "-o", markersize=4)

ax.set_xlabel(r"$\overline{K}$", labelpad=10)
ax.set_ylabel(r"$\Psi$", labelpad=10)
ax.set_ylim([0,1])
ax.set_xlim(0.3, 0.7)


ax_in.set_xlabel(r"$\overline{K}$", labelpad=8, fontsize=22)
ax_in.set_ylabel(r"$G$", labelpad=0, fontsize=22)
ax_in.set_xlim(0.3, 0.7)
ax_in.set_ylim(-1.0, 0.8)
ax_in.set_yticks([-1.0, -0.5, 0.0, 0.5], fontsize=18)
ax_in.set_xticks(np.arange(0.3, 0.71, 0.1), fontsize=18)


folder = os.path.abspath('../plots/for_figures/binder')
if not os.path.exists(folder):
    os.makedirs(folder)
plt.savefig(os.path.join(folder, "binder_Kstd1.pdf"), bbox_inches="tight")

# plt.show()


