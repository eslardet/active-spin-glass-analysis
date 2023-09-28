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

filename = "phase_transition_RI_rho0.13"
file = os.path.abspath("plot_paper/data/" + filename + ".txt")
with open(file) as f:
    reader = csv.reader(f, delimiter="\n")
    r = list(reader)

num_RI = int((len(r))/3)

# colors = plt.cm.OrRd(np.linspace(0.2, 1, num_Kstd))
# colors = plt.cm.PuRd(np.linspace(0.2, 1, num_Kstd))
# colors = plt.cm.BuPu(np.linspace(0.2, 1, num_RI))

file = os.path.abspath("plot_paper/data/" + filename + ".txt")

with open(file) as f:
        reader = csv.reader(f, delimiter="\n")
        r = list(reader)

RI_range = []
K_avg_crit = []
fig, ax = plt.subplots(figsize=(10,7))
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

L = np.sqrt(1000/0.13)

ax.plot(RI_range/L, K_avg_crit[:-1], '-o')
ax.hlines(K_avg_crit[-1], 0, 50, linestyle="dashed", color="tab:red")
ax.text(5, 0.18, r"$\sigma_K=\infty$", color="tab:red")

ax.set_xlim([0,0.6])
ax.set_ylabel(r"$\overline{K}_c$")
ax.set_xlabel(r"$\sigma_I/L$")


folder = os.path.abspath('../plots/for_figures/crit')
if not os.path.exists(folder):
    os.makedirs(folder)
# filename = "collapse"
# plt.savefig(os.path.join(folder, filename + ".svg"), bbox_inches='tight')

plt.show()


