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



filename = "phase_transition_RI"
file = os.path.abspath("plot_paper/data/" + filename + ".txt")
with open(file) as f:
    reader = csv.reader(f, delimiter="\n")
    r = list(reader)

num_RI = int((len(r))/3)

small = 18
big = 28

plt.rc('font', size=big)          # controls default text sizes
plt.rc('axes', labelsize=big)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
plt.rc('legend', fontsize=small)    # legend fontsize

matplotlib.rcParams["font.family"] = "serif"
plt.rcParams['text.usetex'] = True

# font = {'family' : 'normal',
#         'weight' : 'normal',
#         'size'   : 14}

# matplotlib.rc('font', **font)

colors = plt.cm.BuPu(np.linspace(0.2, 1, num_RI))
# colors = plt.cm.OrRd(np.linspace(0.2, 1, num_Kstd))
# colors = plt.cm.binary(np.linspace(0.2, 1, num_Kstd))

fig, ax = plt.subplots(figsize=(10,7))

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
    # ax.plot(K_avg_plot, p_ss_plot, "-o", color=colors[k], label=r"$R_I=\ $" + str(Rp))
    if str(Rp) == "I":
        ax.plot(K_avg_plot, p_ss_plot, "--", label=r"$R_I=\infty$", color="black")
    else:
        ax.plot(K_avg_plot, p_ss_plot, "-o", color=colors[k], label=r"$R_I=\ $" + str(Rp))
    # ax.plot(K_avg_plot, p_ss_plot, "-o", label=str(Rp))
    
    # ax.plot(K_avg_plot, p_ss_plot, "-o")

params = r[3*k][0].split('\t')
nPart = params[0]
Rp = params[1]
noise = params[3]
# noise = "0.20"
Kstd = params[3]
# Kstd = "8.0"
rho = 1.0
phi = 0.1

ax.set_xlabel(r"$K_{AVG}$")
ax.set_ylabel(r"$\Psi$")
# ax.set_xlabel(r"$K_{AVG}$", fontsize=16)
# ax.set_ylabel(r"Polar order parameter, $\Psi$", fontsize=16)
ax.set_ylim([0,1])
ax.set_xlim([-1.0,1.0])
# ax.set_title("With repulsion", fontsize=14)
# plt.suptitle("With repulsion", fontsize=14)
# ax.set_title("With repulsion ($N=$" + str(nPart) + r"; $\phi=$" + str(phi) + r"; $\eta=$" + str(noise) + r"; $R_I=$" + str(Rp) + ")", fontsize=14)
# ax.set_title(r"$N=$" + str(nPart) + r"; $\rho=$" + str(rho) + r"; $\eta=$" + str(noise) + r"; $R_I=$" + str(Rp), fontsize=10)
# ax.set_title(r"$N=$" + str(nPart) + r"; $\phi=$" + str(phi) + r"; $\eta=$" + str(noise) + r"; $K_{STD}=$" + str(Kstd), fontsize=10)
# ax.set_title("Rep vs no rep MF; " + r"$N=$" + str(nPart) + r"; $\eta=$" + str(noise) + r"; $K_{STD}=$" + str(Kstd))
# ax.legend(loc="lower right", fontsize=14)
ax.legend()


folder = os.path.abspath('../plots/for_figures/pt')
if not os.path.exists(folder):
    os.makedirs(folder)
plt.savefig(os.path.join(folder, filename + ".pdf"), bbox_inches="tight")

plt.show()


