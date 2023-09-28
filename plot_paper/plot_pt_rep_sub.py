import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm, colors
from decimal import Decimal
import freud
import scipy.stats as sps
import bisect

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
plt.rcParams['axes.labelpad']=10


filenames = ["phase_transition_rep", "phase_transition_rep_R10"]
labels = ["(a)", "(b)"]

fig, axs = plt.subplots(1,2,figsize=(22,7))

for j in range(2):
    filename = filenames[j]
    ax = axs[j]

    file = os.path.abspath("plot_paper/data/" + filename + ".txt")
    with open(file) as f:
        reader = csv.reader(f, delimiter="\n")
        r = list(reader)

    num_Kstd = int(len(r)/3)

    colors = plt.cm.BuPu(np.linspace(0.2, 1, num_Kstd))

    ax.vlines(0, 0, 1, linestyle="dashed", color="black")

    for k in range(num_Kstd):
        params = r[3*k][0].split('\t')
        K_std = float(params[3])

        K_avg = r[3*k+1][0].split('\t')[:-1]
        K_avg_plot = [float(i) for i in K_avg]
        
        p_ss = r[3*k+2][0].split('\t')[:-1]
        p_ss_plot = [float(i) for i in p_ss]
        ax.plot(K_avg_plot, p_ss_plot, "-o", color=colors[k], label=r"$\sigma_K=" + str(round(K_std)) + r"$")
        # ax.plot(K_avg_plot, p_ss_plot, "-o")

    ax.set_xlabel(r"$\overline{K}$")
    ax.set_ylabel(r"$\Psi$")
    # ax.set_xlabel(r"$K_{AVG}$", fontsize=16)
    # ax.set_ylabel(r"Polar order parameter, $\Psi$", fontsize=16)
    ax.set_ylim([0,1])
    if j == 0:
        ax.set_xlim([-1.0,5.0])
    else:
        ax.set_xlim([-1.0,1.0])
    ax.legend(loc="lower right", frameon=False)
    ax.text(-0.08, 1.0, labels[j], transform=ax.transAxes, va='top', ha='right')


folder = os.path.abspath('../plots/for_figures/pt')
if not os.path.exists(folder):
    os.makedirs(folder)
filename = "pt_rep"
plt.savefig(os.path.join(folder, filename + ".pdf"), bbox_inches="tight")

# plt.show()


