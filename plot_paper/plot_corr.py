import numpy as np
import os, sys, csv
import matplotlib.pyplot as plt
import matplotlib
from analysis.analysis_functions import * 


mode = 'G'
nPart_range = [10000]
phi = 1.0
noise = "0.20"
K_arr = [1.0, 1.5, 2.0]
# K_avg_range = np.concatenate((np.round(np.arange(-1.0,0.0,0.1),1), np.round(np.arange(0.0, 0.6, 0.1), 1),K_arr))
# K_avg_range = np.delete(K_avg_range, 9)
# K_avg_range = np.concatenate((np.round(np.arange(-1.0,0.0,0.2),1), np.round(np.arange(0.0, 0.6, 0.2), 1),K_arr))
K_avg_range = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
# K_avg_range = [-0.5,0.0]
K_std_range = [8.0]
Rp = 1.0
xTy = 1.0
seed_range = np.arange(1,21,1)
r_scale = "log"
y_scale = "log"
timestep_range = [0,1,2,3,4,5]

d_type = "dv_par"
x_scale = "log"
bin_ratio = 2

small = 18
big = 28

plt.rc('font', size=big)          # controls default text sizes
plt.rc('axes', labelsize=big)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
plt.rc('legend', fontsize=small)    # legend fontsize

matplotlib.rcParams["font.family"] = "serif"
plt.rcParams['text.usetex'] = True
colors = plt.cm.GnBu(np.linspace(0.2, 1, len(K_avg_range)))

fig, ax = plt.subplots(figsize=(10,7))

# labels = [r"$C_\parallel(r)$", r"$C_\perp(r)$"]
# d_type_list = ['dv_par', 'dv_perp']
# for d_type in d_type_list:
i = 0
for nPart in nPart_range:
    for K_std in K_std_range:
        exponents = []
        for K_avg in K_avg_range:
            K = str(K_avg) + "_" + str(K_std)
            r_plot, corr_bin_av = read_corr_vel(mode, nPart, phi, noise, K, Rp, xTy, seed_range, x_scale, d_type, bin_ratio)
            ax.plot(r_plot, np.abs(corr_bin_av), '-', label=r"$K_{AVG}=$" + str(K_avg), color=colors[i])
            i += 1
                
ax.set_xscale("log")
ax.set_xlim(left=1)
ax.set_yscale("log")
ax.set_ylim(bottom=10**-4, top=10**0)
ax.set_xlabel(r"$r$")
ax.set_ylabel(r"$C_\parallel(r)$")
ax.legend()

filename = d_type + '_' + mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_Rp' + str(Rp) + '_xTy' + str(xTy)
folder = os.path.abspath('../plots/for_figures/correlation_velocity')
if not os.path.exists(folder):
    os.makedirs(folder)
plt.savefig(os.path.join(folder, filename + ".pdf"), bbox_inches="tight")

# plt.show()