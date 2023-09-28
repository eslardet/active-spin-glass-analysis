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

plt.rc('font', size=small)          # controls default text sizes
plt.rc('axes', labelsize=big)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
plt.rc('legend', fontsize=small)    # legend fontsize

matplotlib.rcParams["font.family"] = "serif"
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.labelpad']=10

fig, axs = plt.subplots(1,2, figsize=(24,7))
plt.subplots_adjust(wspace=0.2)

ax = axs[0]

filename = "phase_transition_RI_rho0.13"
file = os.path.abspath("plot_paper/data/" + filename + ".txt")
with open(file) as f:
    reader = csv.reader(f, delimiter="\n")
    r = list(reader)

num_RI = int((len(r))/3)

colors = plt.cm.BuPu(np.linspace(0.2, 1, num_RI))

ax.vlines(0, 0, 1, linestyle="dashed", color="black")

for k in range(num_RI):
    params = r[3*k][0].split('\t')
    # print(params)
    K_std = float(params[4])
    nPart = params[0]
    Rp = params[1]

    K_avg = r[3*k+1][0].split('\t')[:-1]
    K_avg_plot = [float(i) for i in K_avg]
    
    p_ss = r[3*k+2][0].split('\t')[:-1]
    p_ss_plot = [float(i) for i in p_ss]
    # ax.plot(K_avg_plot, p_ss_plot, "-o", color=colors[k], label=r"$\sigma_I=\ $" + str(Rp))
    if str(Rp) == "I":
        ax.plot(K_avg_plot, p_ss_plot, linestyle="dashdot", marker='v', label=r"$\sigma_I=\infty$", color="tab:red")
    else:
        ax.plot(K_avg_plot, p_ss_plot, "-o", color=colors[k], label=r"$\sigma_I=" + str(round(float(Rp))) + r"$")


ax.set_xlabel(r"$\overline{K}$")
ax.set_ylabel(r"$\Psi$")

ax.set_ylim([0,1])
ax.set_xlim([-1.0,1.0])

ax.legend(ncol=1, frameon=False, fontsize=18)
ax.text(-0.06, 1.0, "(a)", transform=ax.transAxes, va='top', ha='right')

ax = axs[1]

filename = "phase_transition_RI_rho0.13"
file = os.path.abspath("plot_paper/data/" + filename + ".txt")
with open(file) as f:
    reader = csv.reader(f, delimiter="\n")
    r = list(reader)

num_RI = int((len(r))/3)

file = os.path.abspath("plot_paper/data/" + filename + ".txt")

with open(file) as f:
        reader = csv.reader(f, delimiter="\n")
        r = list(reader)

RI_range = []
K_avg_crit = []

for k in range(num_RI):
    params = r[3*k][0].split('\t')

    if k < num_RI-1:
        RI = float(params[1])
        RI_range.append(RI)
    K_avg_range = r[3*k+1][0].split('\t')[:-1]
    K_avg_range = [float(i) for i in K_avg_range]
        
    p_ss = r[3*k+2][0].split('\t')[:-1]
    p_ss = [float(i) for i in p_ss]

    # ax.plot(K_avg_plot, p_ss_plot, "-o", color=colors[k], label=r"$F, K_{STD}=\ $" + str(K_std))

    cutoff = 0.2
    for i in range(len(p_ss)):
        if p_ss[i] > cutoff: # For a strictly increasing function
            break

    # Equation of line method (more accurate)
    grad = (p_ss[i]-p_ss[i-1])/(K_avg_range[i]-K_avg_range[i-1])
    intercept = p_ss[i] - grad*K_avg_range[i]

    KAVG_crit = (cutoff-intercept)/grad
    K_avg_crit.append(KAVG_crit)

ax.plot(RI_range, K_avg_crit[:-1], '-o')
ax.hlines(K_avg_crit[-1], 0, 50, linestyle="dashdot", color="tab:red")
ax.text(5, 0.18, r"$\sigma_K=\infty$", color="tab:red")
ax.text(-0.06, 1.0, "(b)", transform=ax.transAxes, va='top', ha='right')

ax.set_xlim([0,45])
ax.set_ylabel(r"$\overline{K}_c$")
ax.set_xlabel(r"$\sigma_I$", labelpad=5)


folder = os.path.abspath('../plots/for_figures/pt')
if not os.path.exists(folder):
    os.makedirs(folder)
filename = "phase_transition_RI_subplot"
plt.savefig(os.path.join(folder, filename + ".pdf"), bbox_inches='tight')

# plt.show()


