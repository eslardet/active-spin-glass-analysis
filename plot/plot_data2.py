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

save_plot = True
num_Kstd = 6
filename = "G_N90000_phi4.0_n0.20_Kstd8.0_Rp1.0_xTy1.0"

file = os.path.abspath("plot/" + filename + ".txt")

with open(file) as f:
        reader = csv.reader(f, delimiter="\n")
        r = list(reader)

fig, ax = plt.subplots()
for k in range(num_Kstd):
    params = r[3*k][0].split('\t')
    # print(params)
    K_std = params[3]
    nPart = params[0]
    Rp = params[1]
    rho = params[2]

    K_avg = r[3*k+1][0].split('\t')[:-1]
    # K_avg_plot = [float(i)/float(K_std)*float(nPart)**(1/2) for i in K_avg]
    K_avg_plot = [float(i) for i in K_avg]
    
    p_ss = r[3*k+2][0].split('\t')[:-1]
    p_ss_plot = [float(i) for i in p_ss]

    # ax.plot(K_avg_plot, p_ss_plot, "-o", label=r"$R_I=$" + str(Rp))
    # if str(Rp) == "I":
    #     ax.plot(K_avg_plot, p_ss_plot, "--", label=r"$R_I=\infty$", color="black")
    # else:
    #     ax.plot(K_avg_plot, p_ss_plot, "-o", label=r"$R_I=$" + str(Rp), color=cm.tab20(k))
    # ax.plot(K_avg_plot, p_ss_plot, "-o", label=str(Rp))
    ax.plot(K_avg_plot, p_ss_plot, "-o", label=r"$\rho=$" + str(rho))
    # ax.plot(K_avg_plot, p_ss_plot, "-o")

params = r[3*k][0].split('\t')
nPart = params[0]
Rp = params[1]
noise = params[3]
# noise = "0.20"
Kstd = params[4]
# Kstd = "8.0"
rho = 1.0
phi = 0.1

# ax.set_xlim([-20,10])

# ax.set_xlabel(r"$N^{1/2} \times K_{AVG}/K_{STD}$")
ax.set_ylabel(r"Polar order parameter, $\Psi$")
ax.set_xlabel(r"$K_{AVG}$")
# ax.set_ylabel(r"Polar order parameter, $\Psi$", fontsize=16)
ax.set_ylim([0,1])
# ax.set_title("With repulsion", fontsize=14)
# plt.suptitle("With repulsion", fontsize=14)
# ax.set_title("With repulsion ($N=$" + str(nPart) + r"; $\phi=$" + str(phi) + r"; $\eta=$" + str(noise) + r"; $R_I=$" + str(Rp) + ")", fontsize=14)
# ax.set_title(r"$N=$" + str(nPart) + r"; $\rho=$" + str(rho) + r"; $\eta=$" + str(noise) + r"; $R_I=$" + str(Rp), fontsize=10)
ax.set_title(r"$N=$" + str(nPart) + r"; $\eta=$" + str(noise) + r"; $K_{STD}=$" + str(Kstd), fontsize=10)
# ax.set_title("Rep vs no rep MF; " + r"$N=$" + str(nPart) + r"; $\eta=$" + str(noise) + r"; $K_{STD}=$" + str(Kstd))
# ax.legend(loc="lower right", fontsize=14)
ax.legend(loc="upper left")

if save_plot:
    folder = os.path.abspath('../plots/local')
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename + ".png"))

plt.show()


