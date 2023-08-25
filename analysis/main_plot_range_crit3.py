import numpy as np
import analysis_functions_vicsek_old as fun
import os
import matplotlib.pyplot as plt
import sys
import time

mode = 'G'
nPart = 10000
phi = 1.0
noise_range_1 = [format(i, '.2f') for i in np.arange(0.40,0.65,0.20)]
noise_range_2 = ["0.80"]
#noise_range = ["0.60"]
#K_avg_range = np.round(np.arange(0.0,1.1,0.1),1)
K_avg_range_1 = [format(i, '.2f') for i in np.arange(0.40,0.61,0.01)] 
K_avg_range_2 = np.round(np.concatenate((np.arange(-1.0,0.0,0.1), np.arange(0.0,1.1,0.1))),1)
K_avg_range_3 = np.round(np.concatenate((np.arange(-2.0,0.0,0.1), np.arange(0.0,0.1,0.1))),1)
#K_std_range = np.round(np.arange(1.0,8.1,1.0),1)
K_std_range_1 = [0.1]
K_std_range_2 = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,10.0,20.0]
K_std_range_3 = [100.0]
K_std_range = K_std_range_2+K_std_range_3
K_std_range_plot = [np.log(k) for k in K_std_range]
xTy = 5.0
seed_range = np.arange(1,21,1)


fig, ax = plt.subplots()

noise = "0.60"
K_crit_list = []
# for K_std in K_std_range_1:
#     K_crit = critical_value_kavg(mode, nPart, phi, noise, K_avg_range_1, K_std, xTy, seed_range, cutoff=0.3)
#     print(K_std, K_crit)
#     K_crit_list.append(K_crit)
for K_std in K_std_range_2:
    K_crit = critical_value_kavg(mode, nPart, phi, noise, K_avg_range_2, K_std, xTy, seed_range, cutoff=0.3)
    print(K_std, K_crit)
    K_crit_list.append(K_crit)
for K_std in K_std_range_3:
    K_crit = critical_value_kavg(mode, nPart, phi, noise, K_avg_range_3, K_std, xTy, seed_range, cutoff=0.3)
    print(K_std, K_crit)
    K_crit_list.append(K_crit)

print(K_crit_list)
slope, intercept = np.poly1d(np.polyfit(K_std_range, K_crit_list, 1)).c
K_crit_list_plot = [np.log((k-intercept)/slope) for k in K_crit_list]
# ax.plot(K_std_range_plot, K_crit_list_plot, '-o', label=r"$\eta = $" + str(noise))

# ax.plot(K_std_range_plot, K_std_range_plot, '--', color="grey", label="Slope = 1")
# ax.set_xlabel(r"$\log(K_{STD})$")
# ax.set_ylabel(r"$\log(\tilde{K}_{AVG}^C)$")
# ax.legend()
# # ax.set_xscale('log')
# # ax.set_yscale('log')

# folder = os.path.abspath('../plots/Kavg_crit_vs_Kstd/')
# filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_xTy' + str(xTy) + '_log.png'
# if not os.path.exists(folder):
#     os.makedirs(folder)
# plt.savefig(os.path.join(folder, filename))