import numpy as np
from scipy.optimize import curve_fit
import os, sys, csv
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.family"] = "serif"
plt.rcParams['text.usetex'] = True
small = 18
big = 18

plt.rc('font', size=big)          # controls default text sizes
plt.rc('axes', labelsize=big)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
plt.rc('legend', fontsize=small)    # legend fontsize


# filename = 'G_N10000_noise0.20_phi1.0_Kavg1.0_Kstd0.0_xTy1.0'
filename = "G_N1000_noise0.20_phi0.4_Kavg1.0_Kstd0.0_xTy1.0_v1_dt5-3"
# filename = "G_N1000_noise0.20_phi0.4_Kavg1.0_Kstd0.0_xTy1.0_v1_dt5-4"
# filename = "G_N1000_noise0.20_phi0.4_Kavg1.0_Kstd0.0_xTy1.0_v0.5_dt5-3"
# filename = "G_N1000_noise0.20_phi0.4_Kavg1.0_Kstd0.0_xTy1.0_v0.1"

folder = os.path.abspath('../plots/nn_vs_RI')
file = os.path.join(folder, filename + ".txt")

with open(file) as f:
    reader = csv.reader(f, delimiter="\n")
    r = list(reader)

params = r[0][0].split('\t')[:-1]
nPart = int(params[0])
noise = float(params[1])
phi = float(params[2])

RI_list = [float(n) for n in r[1][0].split('\t')[:-1]]
mean_com = [float(p) for p in r[2][0].split('\t')[:len(RI_list)]]
sd_com = [float(p) for p in r[2][0].split('\t')[len(RI_list):-1]]


fig, ax = plt.subplots(figsize=(7,5))

# ax.plot(RI_list, mean_com, '-o')
ax.errorbar(RI_list, mean_com, yerr=sd_com, fmt='o-', capsize=3)
ax.set_xlabel(r"$R_I$")
ax.set_ylabel("Mean distance to nearest neighbor")
ax.set_title(r"$N=$" + str(nPart) + r", $\eta=$" + str(noise) + r", $\rho=$" + str(phi))
ax.set_ylim(0.2,0.8)

folder = os.path.abspath('../plots/nn_vs_RI')
if not os.path.exists(folder):
    os.makedirs(folder)
plt.savefig(os.path.join(folder, filename + ".png"), bbox_inches="tight")

# plt.show()