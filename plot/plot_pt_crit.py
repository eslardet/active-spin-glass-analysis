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

small = 12
big = 18

plt.rc('font', size=big)          # controls default text sizes
plt.rc('axes', labelsize=big)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
plt.rc('legend', fontsize=small)    # legend fontsize

matplotlib.rcParams["font.family"] = "serif"
plt.rcParams['text.usetex'] = True

num_Kstd = 9
filename = "phase_transition_F"

colors = plt.cm.OrRd(np.linspace(0.2, 1, num_Kstd))

file = os.path.abspath("plot/" + filename + ".txt")

with open(file) as f:
        reader = csv.reader(f, delimiter="\n")
        r = list(reader)

K_std_range = []
K_avg_crit = []
fig, ax = plt.subplots()
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

    # ax.plot(K_plot, p_ss, "-o", color=colors[k], label=r"$F, K_{STD}=\ $" + str(K_std))

# ax.set_xlim([-0.2,0.2])
# ax.set_xlabel(r"$K_{STD}$")
# ax.set_ylabel(r"$K_{AVG}^c$")
# plt.show()

ax.plot(K_std_range, K_avg_crit, 'o', label='Data')
ax.plot(K_std_range, [slope*k+coeff for k in K_std_range], '--', label='Linear fit')
ax.set_xlabel(r"$K_{STD}$")
ax.set_ylabel(r"$K_{AVG}^c$")
ax.legend()
plt.show()

# params = r[3*k][0].split('\t')
# nPart = params[0]
# Rp = params[1]
# noise = params[3]
# # noise = "0.20"
# Kstd = params[3]
# # Kstd = "8.0"
# rho = 1.0
# phi = 0.1

# ax.set_xlabel(r"$K_{AVG}$")
# ax.set_ylabel(r"$\Psi$")
# # ax.set_xlabel(r"$K_{AVG}$", fontsize=16)
# # ax.set_ylabel(r"Polar order parameter, $\Psi$", fontsize=16)
# ax.set_ylim([0,1])
# ax.set_xlim([-0.1,0])
# # ax.set_title("With repulsion", fontsize=14)
# # plt.suptitle("With repulsion", fontsize=14)
# # ax.set_title("With repulsion ($N=$" + str(nPart) + r"; $\phi=$" + str(phi) + r"; $\eta=$" + str(noise) + r"; $R_I=$" + str(Rp) + ")", fontsize=14)
# # ax.set_title(r"$N=$" + str(nPart) + r"; $\rho=$" + str(rho) + r"; $\eta=$" + str(noise) + r"; $R_I=$" + str(Rp), fontsize=10)
# # ax.set_title(r"$N=$" + str(nPart) + r"; $\phi=$" + str(phi) + r"; $\eta=$" + str(noise) + r"; $K_{STD}=$" + str(Kstd), fontsize=10)
# # ax.set_title("Rep vs no rep MF; " + r"$N=$" + str(nPart) + r"; $\eta=$" + str(noise) + r"; $K_{STD}=$" + str(Kstd))
# # ax.legend(loc="lower right", fontsize=14)
# ax.legend(loc="lower right")


folder = os.path.abspath('../plots/for_figures')
if not os.path.exists(folder):
    os.makedirs(folder)
# plt.savefig(os.path.join(folder, filename + ".png"))

# plt.show()


