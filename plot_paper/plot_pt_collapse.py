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

num_Kstd = 9
filename = "phase_transition"

# colors = plt.cm.OrRd(np.linspace(0.2, 1, num_Kstd))
# colors = plt.cm.PuRd(np.linspace(0.2, 1, num_Kstd))
colors = plt.cm.BuPu(np.linspace(0.2, 1, num_Kstd))

file = os.path.abspath("plot_paper/data/" + filename + ".txt")

with open(file) as f:
        reader = csv.reader(f, delimiter="\n")
        r = list(reader)

K_std_range = []
K_avg_crit = []
fig, ax = plt.subplots(figsize=(10,7))
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

slope, coeff = np.polyfit(K_std_range, K_avg_crit, 1)
print(slope, coeff)
print(np.arange(1,9,1)*slope+coeff)

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

    ax.plot(K_plot, p_ss, "o", color=colors[k], label=r"$\sigma_K=\ $" + str(round(K_std)))

ax.set_xlim([-0.2,0.2])
ax.set_xlabel(r"$(\overline{K}-$" + str(round(coeff,3)) + r"$)/\sigma_K$")
ax.set_ylabel(r"$\Psi$")
ax.legend()

# ax.plot(K_std_range, K_avg_crit, 'o')
# ax.plot(K_std_range, [slope*k+coeff for k in K_std_range], '--', label='Linear fit')
# ax.set_xlabel(r"$K_{STD}$")
# ax.set_ylabel(r"$K_{AVG}^c$")
# ax.legend(fontsize=big)

folder = os.path.abspath('../plots/for_figures/crit')
if not os.path.exists(folder):
    os.makedirs(folder)
filename = "collapse"
# plt.savefig(os.path.join(folder, filename + ".svg"), bbox_inches='tight')

plt.show()


