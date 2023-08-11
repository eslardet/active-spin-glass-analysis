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

num_Kstd = 11
cutoff = 0.4
KAVG_crit = []
Rp_plot = []
save_plot = True
filename = "G_N1000_phi0.13_n0.20_Kstd8.0_RpI_xTy1.0"

file = os.path.abspath("plot/" + filename + ".txt")

with open(file) as f:
        reader = csv.reader(f, delimiter="\n")
        r = list(reader)

fig, ax = plt.subplots()
for k in range(num_Kstd-1):
    params = r[3*k][0].split('\t')
    # print(params)
    K_std = params[4]
    nPart = params[0]
    Rp = params[1]
    Rp_plot.append(float(Rp))

    K_avg = r[3*k+1][0].split('\t')[:-1]
    K_avg_plot = [float(i) for i in K_avg]
    
    p_ss = r[3*k+2][0].split('\t')[:-1]
    p_ss_plot = [float(i) for i in p_ss]

    for i in range(len(p_ss)):
        if p_ss_plot[i] > cutoff: # For a strictly increasing function
            break

    # Midpoint method
    # KAVG_crit = (KAVG_range[i] + KAVG_range[i-1])/2

    # Equation of line method (more accurate)
    grad = (p_ss_plot[i]-p_ss_plot[i-1])/(K_avg_plot[i]-K_avg_plot[i-1])
    intercept = p_ss_plot[i] - grad*K_avg_plot[i]

    KAVG_crit.append((cutoff-intercept)/grad)

ax.plot(Rp_plot, KAVG_crit, '-o', label=r"No rep, $\rho=0.13$")

num_Kstd = 11
cutoff = 0.4
KAVG_crit = []
Rp_plot = []
# save_plot = False
filename = "G_N1000_phi0.1_n0.20_Kstd8.0_RpI_xTy1.0"

file = os.path.abspath("plot/" + filename + ".txt")

with open(file) as f:
        reader = csv.reader(f, delimiter="\n")
        r = list(reader)

# fig, ax = plt.subplots()
for k in range(1,num_Kstd-1):
    params = r[3*k][0].split('\t')
    # print(params)
    # K_std = params[4]
    nPart = params[0]
    Rp = params[1]
    Rp_plot.append(float(Rp))

    K_avg = r[3*k+1][0].split('\t')[:-1]
    K_avg_plot = [float(i) for i in K_avg]
    
    p_ss = r[3*k+2][0].split('\t')[:-1]
    p_ss_plot = [float(i) for i in p_ss]

    for i in range(len(p_ss)):
        if p_ss_plot[i] > cutoff: # For a strictly increasing function
            break

    # Midpoint method
    # KAVG_crit = (KAVG_range[i] + KAVG_range[i-1])/2

    # Equation of line method (more accurate)
    grad = (p_ss_plot[i]-p_ss_plot[i-1])/(K_avg_plot[i]-K_avg_plot[i-1])
    intercept = p_ss_plot[i] - grad*K_avg_plot[i]

    KAVG_crit.append((cutoff-intercept)/grad)
ax.plot(Rp_plot, KAVG_crit, '-o', label=r"Rep, $\phi=0.1$")


num_Kstd = 9
cutoff = 0.4
KAVG_crit = []
Rp_plot = []
# save_plot = False
filename = "G_N1000_phi1.0_n0.20_Kstd8.0_Rp30.0_xTy1.0"

file = os.path.abspath("plot/" + filename + ".txt")

with open(file) as f:
        reader = csv.reader(f, delimiter="\n")
        r = list(reader)

# fig, ax = plt.subplots()
for k in range(num_Kstd):
    params = r[3*k][0].split('\t')
    # print(params)
    # K_std = params[4]
    nPart = params[0]
    Rp = params[1]
    Rp_plot.append(float(Rp))

    K_avg = r[3*k+1][0].split('\t')[:-1]
    K_avg_plot = [float(i) for i in K_avg]
    
    p_ss = r[3*k+2][0].split('\t')[:-1]
    p_ss_plot = [float(i) for i in p_ss]

    for i in range(len(p_ss)):
        if p_ss_plot[i] > cutoff: # For a strictly increasing function
            break

    # Midpoint method
    # KAVG_crit = (KAVG_range[i] + KAVG_range[i-1])/2

    # Equation of line method (more accurate)
    grad = (p_ss_plot[i]-p_ss_plot[i-1])/(K_avg_plot[i]-K_avg_plot[i-1])
    intercept = p_ss_plot[i] - grad*K_avg_plot[i]

    KAVG_crit.append((cutoff-intercept)/grad)
ax.plot(Rp_plot, KAVG_crit, '-o', label=r"No Rep, $\rho=1.0$")



params = r[3*k][0].split('\t')
nPart = params[0]
Rp = params[1]
noise = params[3]
# noise = "0.20"
Kstd = params[3]
# Kstd = "8.0"
rho = 0.13
phi = 0.1

ax.set_xlabel(r"$R_I$")
ax.set_ylabel(r"$K_{AVG}^C$")
# ax.set_xlabel(r"$K_{AVG}$", fontsize=16)
# ax.set_ylabel(r"Polar order parameter, $\Psi$", fontsize=16)
# ax.set_ylim([0,1])
# ax.set_title("With repulsion", fontsize=14)
# plt.suptitle("With repulsion", fontsize=14)
# ax.set_title("With repulsion ($N=$" + str(nPart) + r"; $\phi=$" + str(phi) + r"; $\eta=$" + str(noise) + r"; $R_I=$" + str(Rp) + ")", fontsize=14)
# ax.set_title(r"$N=$" + str(nPart) + r"; $\rho=$" + str(rho) + r"; $\eta=$" + str(noise) + r"; $R_I=$" + str(Rp), fontsize=10)
# ax.set_title(r"$N=$" + str(nPart) + r"; $\phi=$" + str(phi) + r"; $\eta=$" + str(noise) + r"; $K_{STD}=$" + str(Kstd), fontsize=10)
# ax.set_title("Rep vs no rep MF; " + r"$N=$" + str(nPart) + r"; $\eta=$" + str(noise) + r"; $K_{STD}=$" + str(Kstd))
# ax.legend(loc="lower right", fontsize=14)
ax.legend()


if save_plot:
    filename = "rep_vs_norep_Kcrit_R_I"
    folder = os.path.abspath('../plots/local')
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename + ".png"))

plt.show()

