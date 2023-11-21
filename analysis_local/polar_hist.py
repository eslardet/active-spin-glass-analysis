import sys
sys.path.insert(1, './analysis/analysis_functions')
from stats import *
from visuals import *

import numpy as np
import os
import matplotlib.pyplot as plt
import time
import csv
from scipy.stats import circmean, circvar

mode = "G"
nPart = 100000
phi = 1.0
noise = "0.20"
K = "0.8_8.0"
Rp = 1.0
xTy = 5.0
seed = 1

pos_ex_file = get_file_path(mode, nPart, phi, noise, K, Rp, xTy, seed, file_name='pos_exact')
x, y, theta, view_time = get_pos_ex_snapshot(pos_ex_file)

# inparFile, posFile = get_files(mode, nPart, phi, noise, K, Rp, xTy, seed)
# # theta_all = get_theta_arr(inparFile, posFile, min_T=1, max_T=2)
# x_all, y_all, theta_all = get_pos_arr(inparFile=inparFile, posFile=posFile)
# theta = [t for sublist in theta_all for t in sublist]

theta_wrap = [t%(2*np.pi) for t in theta]


print(np.rad2deg(circmean(theta_wrap)), np.rad2deg(circvar(theta_wrap)))

fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
ax.set_yticklabels([])
ax.hist(theta_wrap, bins=200, ec='k')

# del_pos(mode, nPart, phi, noise, K, Rp, xTy, seed)

folder = os.path.abspath('../plots/polar_hist')
filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed) + '.png'
if not os.path.exists(folder):
    os.makedirs(folder)
plt.savefig(os.path.join(folder, filename))

plt.show()

# for nPart in [1000, 2500, 6400, 10000, 22500, 40000]:
#     for K_avg in [-0.1]:
#         K = str(K_avg) + "_" + str(8.0)

#         pos_ex_file = get_file_path(mode, nPart, phi, noise, K, Rp, xTy, seed, file_name='pos_exact')
#         x, y, theta, view_time = get_pos_ex_snapshot(pos_ex_file)
#         theta_wrap = [t%(2*np.pi) for t in theta]
#         print(np.rad2deg(circmean(theta_wrap)), np.rad2deg(circvar(theta_wrap)))
        
        # plot_polar_hist(mode, nPart, phi, noise, K, Rp, xTy, seed, bins=200)