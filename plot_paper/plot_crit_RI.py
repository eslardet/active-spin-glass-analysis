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

fig, ax = plt.subplots(figsize=(10,7))


filename = "phase_transition_RI"
for rho in ['0.13']:
    file = os.path.abspath("plot_paper/data/" + filename + "_rho" + rho + ".txt")
    with open(file) as f:
        reader = csv.reader(f, delimiter="\n")
        r = list(reader)

    num_RI = int((len(r))/3)

    K_avg_crit = []
    Rp_plot = []
    for k in range(num_RI):
        params = r[3*k][0].split('\t')
        # print(params)
        K_std = float(params[4])
        nPart = params[0]
        Rp = params[1]
        if Rp != "I":
            Rp_plot.append(float(Rp))

        K_avg = r[3*k+1][0].split('\t')[:-1]
        K_avg_plot = [float(i) for i in K_avg]
        
        p_ss = r[3*k+2][0].split('\t')[:-1]
        p_ss_plot = [float(i) for i in p_ss]

        cutoff = 0.2
        for i in range(len(p_ss_plot)):
            if p_ss_plot[i] > cutoff: # For a strictly increasing function
                break

        # Equation of line method (more accurate)
        grad = (p_ss_plot[i]-p_ss_plot[i-1])/(K_avg_plot[i]-K_avg_plot[i-1])
        intercept = p_ss_plot[i] - grad*K_avg_plot[i]

        KAVG_crit = (cutoff-intercept)/grad
        K_avg_crit.append(KAVG_crit)

    ax.plot(Rp_plot, K_avg_crit[:-1], '-o')


ax.axhline(K_avg_crit[-1], color='k', linestyle='--', label=r"$R_I=\infty$")

params = r[3*k][0].split('\t')
nPart = params[0]
Rp = params[1]
noise = params[3]
# noise = "0.20"
Kstd = params[3]
# Kstd = "8.0"
rho = 1.0
phi = 0.1

ax.set_xlabel(r"$R_I$")
ax.set_ylabel(r"$K_{AVG}^c$")
# ax.set_xlabel(r"$K_{AVG}$", fontsize=16)
# ax.set_ylabel(r"Polar order parameter, $\Psi$", fontsize=16)
# ax.set_ylim([0,1])
# ax.set_xlim([-1.0,1.0])
ax.legend()


folder = os.path.abspath('../plots/for_figures/crit')
if not os.path.exists(folder):
    os.makedirs(folder)
plt.savefig(os.path.join(folder, filename + ".pdf"), bbox_inches="tight")

plt.show()


