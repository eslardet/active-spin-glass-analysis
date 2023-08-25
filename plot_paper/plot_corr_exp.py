import numpy as np
import os, sys, csv
import matplotlib.pyplot as plt
import matplotlib
from analysis_functions_vicsek_new import * 


mode = 'G'
nPart_range = [10000]
phi = 1.0
noise = "0.20"
K_arr = [1.0, 1.5, 2.0]
# K_avg_range = np.concatenate((np.round(np.arange(-1.0,0.0,0.1),1), np.round(np.arange(0.0, 0.6, 0.1), 1),K_arr))
# K_avg_range = np.delete(K_avg_range, 9)
K_avg_range = np.concatenate((np.round(np.arange(-1.0,0.0,0.2),1), np.round(np.arange(0.0, 0.6, 0.2), 1),K_arr))
# K_avg_range = [-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4,0.5,1.0]
# K_avg_range = [-0.5,0.0]
K_std_range = [8.0]
Rp = 1.0
xTy = 1.0
seed_range = np.arange(1,21,1)
r_scale = "log"
y_scale = "log"
timestep_range = [0,1,2,3,4,5]

d_type = "dv_par"
min_r = 2
max_r = 10

small = 18
big = 28

plt.rc('font', size=big)          # controls default text sizes
plt.rc('axes', labelsize=big)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
plt.rc('legend', fontsize=big)    # legend fontsize

matplotlib.rcParams["font.family"] = "serif"
plt.rcParams['text.usetex'] = True


fig, ax = plt.subplots(figsize=(10,7))

labels = [r"$C_\parallel(r)$", r"$C_\perp(r)$"]
d_type_list = ['dv_par', 'dv_perp']
for d_type in d_type_list:
    for nPart in nPart_range:
        for K_std in K_std_range:
            exponents = []
            for K_avg in K_avg_range:
                K = str(K_avg) + "_" + str(K_std)
                exponents.append(get_exponent_corr_vel(mode, nPart, phi, noise, K, Rp, xTy, seed_range, d_type, min_r, max_r))
            ax.plot(K_avg_range, exponents, '-o', label=labels[d_type_list.index(d_type)])

ax.set_xlabel(r"$K_{AVG}$")
ax.set_ylabel(r"$\lambda$")
ax.legend(loc="lower right")

filename = d_type + '_' + mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_Rp' + str(Rp) + '_xTy' + str(xTy)
folder = os.path.abspath('../plots/for_figures/correlation_velocity_exp')
if not os.path.exists(folder):
    os.makedirs(folder)
plt.savefig(os.path.join(folder, filename + ".pdf"), bbox_inches="tight")

plt.show()