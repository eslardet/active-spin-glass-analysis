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



# filename = "G_N10000_phi1.0_n0.20_Kstd8.0_Rp1.0_xTy1.0_amode"
# filename = "G_N10000_phi1.0_n0.20_Kstd8.0_Rp1.0_xTy1.0_hr_a"
# filename = "G_N10000_phi1.0_n0.20_Kstd8.0_Rp1.0_xTy1.0_amode_highres"
filename = "G_N60000_phi1.0_n0.20_Kstd8.0_Rp1.0_xTy1.0_amode"
file = os.path.abspath('../plots/p_order_vs_Kavg/' + filename + '.txt')
with open(file) as f:
    reader = csv.reader(f, delimiter="\n")
    r = list(reader)

# num_Kstd = int(len(r)/3)


small = 18
medium = 22
big = 28

plt.rc('font', size=big)          # controls default text sizes
plt.rc('axes', labelsize=big)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=medium)    # fontsize of the tick labels
plt.rc('ytick', labelsize=medium)    # fontsize of the tick labels
plt.rc('legend', fontsize=medium)    # legend fontsize

matplotlib.rcParams["font.family"] = "serif"
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.labelpad']=10

fig, ax = plt.subplots(figsize=(10,7))
# ax.vlines(0, 0, 1, linestyle="dashed", color="black")

labels = ["Ordered initial state", "Disordered initial state"]

for k in range(0,2):
    # params = r[2*k][0].split('\t')

    K_avg = r[3*k+1][0].split('\t')[:-1]
    K_avg_plot = [float(i) for i in K_avg]
    
    p_ss = r[3*k+2][0].split('\t')[:-1]
    p_ss_plot = [float(i) for i in p_ss]
    ax.plot(K_avg_plot, p_ss_plot, "-o", label = labels[k])
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

ax.set_xlabel(r"$\overline{K}$")
ax.set_ylabel(r"$\Psi$")
# ax.set_xlabel(r"$K_{AVG}$", fontsize=16)
# ax.set_ylabel(r"Polar order parameter, $\Psi$", fontsize=16)
ax.set_ylim([0,1])
# ax.set_xlim([-1.0,1.0])
ax.legend(frameon=False)


# folder = os.path.abspath('../plots/for_figures/pt')
folder = os.path.abspath('../plots/p_order_vs_Kavg')
if not os.path.exists(folder):
    os.makedirs(folder)
plt.savefig(os.path.join(folder, filename + ".png"), bbox_inches="tight")

# plt.show()


