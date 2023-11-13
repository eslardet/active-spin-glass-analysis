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

small = 18
big = 28

plt.rc('font', size=big)          # controls default text sizes
plt.rc('axes', labelsize=big)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
plt.rc('legend', fontsize=small)    # legend fontsize

matplotlib.rcParams["font.family"] = "serif"
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.labelpad']=10

fig, axs = plt.subplots(1,2, figsize=(22,7))

#### (a) ####
filenames = ["phase_transition_F", "phase_transition"]
labels = ["(a)", "(b)"]

for j in range(2):
    filename = filenames[j]
    ax = axs[j]

    file = os.path.abspath("plot_paper/data/" + filename + ".txt")
    with open(file) as f:
            reader = csv.reader(f, delimiter="\n")
            r = list(reader)

    num_Kstd = int(len(r)/3)

    if j == 0:
        cols = plt.cm.PuRd(np.linspace(0.2, 1, num_Kstd))
    elif j == 1:
        cols = plt.cm.BuPu(np.linspace(0.2, 1, num_Kstd))

    K_std_range = []
    K_avg_crit = []

    for k in range(1, num_Kstd):
        params = r[3*k][0].split('\t')
        # print(params)
        K_std = float(params[4])
        K_std_range.append(K_std)
        nPart = params[0]
        Rp = params[1]

        K_avg_range = r[3*k+1][0].split('\t')[:-1]
        K_avg_range = [float(i) for i in K_avg_range]
        
        p_ss = r[3*k+2][0].split('\t')[:-1]
        p_ss = [float(i) for i in p_ss]
        
        # ax.plot(K_avg_plot, p_ss_plot, "-o", color=colors[k], label=r"$F, \sigma_K=\ $" + str(K_std))

        cutoff = 0.2
        for i in range(len(p_ss)):
            if p_ss[i] > cutoff: # For a strictly increasing function
                break

        # Equation of line method (more accurate)
        grad = (p_ss[i]-p_ss[i-1])/(K_avg_range[i]-K_avg_range[i-1])
        intercept = p_ss[i] - grad*K_avg_range[i]

        KAVG_crit = (cutoff-intercept)/grad
        K_avg_crit.append(KAVG_crit)

    slope, coeff = np.polyfit(K_std_range, K_avg_crit, 1)
    print(slope, coeff)
    if j == 0:
        ax_in = fig.add_axes([0.3, 0.2, 0.17, 0.38])
    else:
        ax_in = fig.add_axes([0.72, 0.2, 0.17, 0.38])
    ax_in.plot(K_std_range, K_avg_crit, 'o')
    K_std_plot = np.linspace(K_std_range[0], K_std_range[-1], 100)
    ax_in.plot(K_std_plot, K_std_plot*slope + coeff, '--', label="Linear fit")
    ax_in.set_xlabel(r"$\sigma_K$", fontsize=small, labelpad=0)
    ax_in.set_ylabel(r"$\overline{K}_c$", fontsize=small, labelpad=0)
    ax_in.legend(frameon=False)

    for k in range(1, num_Kstd):
        params = r[3*k][0].split('\t')
        # print(params)
        K_std = float(params[4])
        # K_std_range.append(K_std)
        nPart = params[0]
        Rp = params[1]

        K_avg_range = r[3*k+1][0].split('\t')[:-1]
        K_avg_range = [float(i) for i in K_avg_range]
        
        p_ss = r[3*k+2][0].split('\t')[:-1]
        p_ss = [float(i) for i in p_ss]

        K_plot = [(k-coeff)/(K_std) for k in K_avg_range]

        ax.plot(K_plot, p_ss, "o", color=cols[k], label=r"$\sigma_K=\ $" + str(round(K_std)))

    ax.set_xlim([-0.2,0.2])
    ax.set_xlabel(r"$(\overline{K}-$" + str(round(coeff,3)) + r"$)/\sigma_K$")
    ax.set_ylabel(r"$\Psi$")
    ax.legend(frameon=False)
    ax.text(-0.08, 1.0, labels[j], transform=ax.transAxes, va='top', ha='right')


folder = os.path.abspath('../plots/for_figures/crit')
if not os.path.exists(folder):
    os.makedirs(folder)
filename = "collapse_both"
plt.savefig(os.path.join(folder, filename + ".pdf"), bbox_inches='tight')

# plt.show()


