import numpy as np
import os, sys, csv
import matplotlib.pyplot as plt
import matplotlib
sys.path.insert(1, '/Users/el2021/Code/2D_ActiveSpinGlass_EL/Active_Spin_Glass/analysis')
from analysis_functions import * 


mode = 'G'
nPart_range = [10000]
phi = 1.0
noise = "0.20"
K_arr = [1.0, 1.5, 2.0]
# K_avg_range = np.concatenate((np.round(np.arange(-1.0,0.0,0.1),1), np.round(np.arange(0.0, 0.6, 0.1), 1),K_arr))
# K_avg_range = np.delete(K_avg_range, 9)
# K_avg_range = np.concatenate((np.round(np.arange(-1.0,0.0,0.2),1), np.round(np.arange(0.0, 0.6, 0.2), 1),K_arr))
# K_avg_range = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
K_avg_range = [0.0]
# K_std_range = np.arange(1.0, 8.1, 1.0)
K_std_range = [1.0, 8.0]
# K_std_range = [8.0]
Rp = 1.0
xTy = 1.0
seed_range = np.arange(1,21,1)
r_scale = "lin"
y_scale = "log"
timestep_range = [0,1,2,3,4,5]

bin_ratio = 1

min_grid_size = 0.25

small = 22
big = 28

plt.rc('font', size=small)          # controls default text sizes
plt.rc('axes', labelsize=small)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
plt.rc('legend', fontsize=small)    # legend fontsize

matplotlib.rcParams["font.family"] = "serif"
plt.rcParams['text.usetex'] = True
# plt.rcParams['axes.labelpad'] = 10

colors = plt.cm.GnBu(np.linspace(0.2, 1, len(K_avg_range)))
# colors = plt.cm.GnBu(np.linspace(0.2, 1, len(K_std_range)))
# colors = plt.cm.BuGn(np.linspace(0.2, 1, len(K_avg_range)))

fig, ax = plt.subplots(figsize=(8,4))
# fig, ax = plt.subplots(figsize=(10,7))

# labels = [r"$C_\parallel(r)$", r"$C_\perp(r)$"]
# d_type_list = ['dv_par', 'dv_perp']
# for d_type in d_type_list:
i = 0
for nPart in nPart_range:
    for K_std in K_std_range:
        # exponents = []
        for K_avg in K_avg_range:
            K = str(K_avg) + "_" + str(K_std)
            # r_plot, corr_bin_av = read_corr_density_points(mode, nPart, phi, noise, K, Rp, xTy, seed_range, r_scale, bin_ratio)
            dist, corr = read_corr_density(mode, nPart, phi, noise, K, Rp, xTy, seed_range, min_grid_size)
            # dist, corr = get_r_corr_all(mode, nPart, phi, noise, K, Rp, xTy, seed_range, pos_ex=True, timestep_range=[0], min_grid_size=1)
            r_plot, corr_bin_av = get_corr_binned(dist, corr, min_r=0, max_r=5)
            # r_plot = r_plot[6:]
            # corr_bin_av = corr_bin_av[6:]
            # r_plot.insert(0,0)
            # corr_bin_av.insert(0,1)
            # ax.plot(r_plot, np.abs(corr_bin_av), '-', label=r"$\overline{K}=" + str(K_avg) + r"$", color=colors[i])
            ax.plot(r_plot, corr_bin_av, '-', label=r"$\sigma_K=" + str(round(K_std)) + r"$")
            i += 1

# seed_range = np.arange(1,3,1)
# i = 0
# for nPart in nPart_range:
#     for K_std in K_std_range:
#         # exponents = []
#         for K_avg in K_avg_range:
#             K = str(K_avg) + "_" + str(K_std)
#             r_plot, corr_bin_av = read_corr_density(mode, nPart, phi, noise, K, Rp, xTy, seed_range, r_scale, bin_ratio)
#             r_plot = r_plot[6:]
#             corr_bin_av = corr_bin_av[6:]
#             # r_plot.insert(0,0)
#             # corr_bin_av.insert(0,1)
#             # ax.plot(r_plot, np.abs(corr_bin_av), '-', label=r"$\overline{K}=" + str(K_avg) + r"$", color=colors[i])
#             ax.plot(r_plot, np.abs(corr_bin_av), '-', label=r"$\sigma_K=" + str(round(K_std)) + r", \ \tilde{\rho}=2.0$")
#             i += 1

# ax.set_xscale("log")
# ax.set_xlim(right=5)
ax.set_xlim(left=0, right=5)
# ax.set_yscale("log")
# ax.set_ylim(top=1)
ax.set_ylim(bottom=0, top=1)
ax.set_xlabel(r"$r$")
ax.set_ylabel(r"$C_\rho(r)$", labelpad=10)
ax.legend(frameon=False)

# filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_Rp' + str(Rp) + '_xTy' + str(xTy)
filename = "corr_density_K1_8_linlin_d_rho1_g0.25"
folder = os.path.abspath('../plots/for_figures/correlation_density')
if not os.path.exists(folder):
    os.makedirs(folder)
# plt.savefig(os.path.join(folder, filename + ".png"), bbox_inches="tight")
# plt.savefig(os.path.join(folder, filename + ".svg"), bbox_inches="tight")
plt.show()