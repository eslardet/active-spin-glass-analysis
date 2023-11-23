import numpy as np
from scipy.optimize import curve_fit
import os, sys, csv
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.family"] = "serif"
plt.rcParams['text.usetex'] = True
small = 12
big = 18

plt.rc('font', size=big)          # controls default text sizes
plt.rc('axes', labelsize=big)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
plt.rc('legend', fontsize=small)    # legend fontsize

K_list = ["1.0_1.0", "1.9_2.0", "3.7_4.0", "5.5_6.0", "7.3_8.0"]

fig, ax = plt.subplots(figsize=(7,5))

for i in range(len(K_list)):
    filename = 'G_noise0.20_phi1.0_K' + K_list[i] + '_Rp1.0_xTy1.0'

    file = os.path.abspath('../plots/p_order_vs_N/' + filename + '.txt')

    with open(file) as f:
        reader = csv.reader(f, delimiter="\n")
        r = list(reader)

    params = r[0][0].split('\t')[:-1]
    rho = float(params[2])

    n_list = [float(n) for n in r[1][0].split('\t')[:-1]]
    psi_list = [float(p) for p in r[2][0].split('\t')[:-1]]
    psi_sd_list = [float(p) for p in r[3][0].split('\t')[:-1]]


    l_list = np.array([np.sqrt(n/rho) for n in n_list])

    psi_list_shift = [p+1-psi_list[0] for p in psi_list]

    ## Plot L vs Psi with fitted curve
    # ax.plot(l_list, psi_list, 'o-', label = K_list[i])
    ax.plot(l_list, psi_list_shift, 'o-', label = K_list[i])

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel(r"$L$")
ax.set_ylabel(r"$\Psi$")
ax.legend()

plt.savefig(os.path.abspath('../plots/p_order_vs_N/K1_1_superimpose_shifted.png'), bbox_inches="tight")
# plt.show()