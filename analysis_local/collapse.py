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

K_std_range = [1.0, 4.0, 8.0]
K_avg_crit = [0.032, -0.18, -0.51]
fig, ax = plt.subplots(figsize=(10,7))

slope, coeff = np.polyfit(K_std_range, K_avg_crit, 1)
print(slope, coeff)

for k in [1,4,8]:

    K_avg_range = r[3*k+1][0].split('\t')[:-1]
    K_avg_range = [float(i) for i in K_avg_range]
    
    K_std = float(r[3*k][0].split('\t')[4])

    p_ss = r[3*k+2][0].split('\t')[:-1]
    p_ss = [float(i) for i in p_ss]

    K_plot = [(k-coeff)/(K_std) for k in K_avg_range]

    ax.plot(K_plot, p_ss, "o", color=colors[k], label=r"$K_{STD}=\ $" + str(round(K_std)))

ax.set_xlim([-0.2,0.2])
ax.set_xlabel(r"$(K_{AVG}-$" + str(round(coeff,3)) + r"$)/K_{STD}$")
ax.set_ylabel(r"$\Psi$")
ax.legend()

ax_in = fig.add_axes([0.5, 0.2, 0.38, 0.38])
ax_in.plot(K_std_range, K_avg_crit, 'o')
K_std_plot = np.linspace(K_std_range[0], K_std_range[-1], 100)
ax_in.plot(K_std_plot, K_std_plot*slope + coeff, '--', label="Linear fit")
ax_in.set_xlabel(r"$\sigma_K$", fontsize=small, labelpad=0)
ax_in.set_ylabel(r"$\overline{K}_c$", fontsize=small, labelpad=0)

folder = os.path.abspath('../plots/collapse')
if not os.path.exists(folder):
    os.makedirs(folder)
filename = "collapse_from_binder"
plt.savefig(os.path.join(folder, filename + ".png"), bbox_inches='tight')

plt.show()


