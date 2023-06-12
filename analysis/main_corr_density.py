import numpy as np
import scipy
import analysis_functions_vicsek_new as fun
import os
import matplotlib.pyplot as plt
import sys
import time

mode = 'G'
nPart = 10000
phi = 1.0
noise = "0.20"
K_avg = 0.0
K_std = 8.0
K = str(K_avg) + "_" + str(K_std)
Rp = 1.0
xTy = 1.0
seed_range = np.arange(1,21,1)
timestep_range = np.arange(0,6,1)

linlin=True
loglin=False
loglog=False

t0 = time.time()
# fun.plot_corr_density_pos_ex(mode, nPart, phi, noise, K, Rp, xTy, seed_range, linlin=linlin, loglin=loglin, loglog=loglog)
# fun.plot_corr_density(mode, nPart, phi, noise, K, Rp, xTy, seed_range, timestep_range=timestep_range, linlin=linlin, loglin=loglin, loglog=loglog)

# fun.write_corr_density(mode, nPart, phi, noise, K, Rp, xTy, seed_range, timestep_range)
# r_plot, corr = fun.read_corr_density(mode, nPart, phi, noise, K, Rp, xTy, seed_range)
# fun.plot_corr_density_file(mode, nPart, phi, noise, K, Rp, xTy, seed_range, log_y=True, bin_ratio=1)

r_lower = 0
r_upper = 3

exponents = []
exponents_2 = []
K_avg_range = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

for K_avg in K_avg_range:
    K = str(K_avg) + "_" + str(K_std)
    r_plot, corr = fun.read_corr_density(mode, nPart, phi, noise, K, Rp, xTy, seed_range)
    lower = np.where(np.array(r_plot) > r_lower)[0][0]
    upper = np.where(np.array(r_plot) < r_upper)[0][-1]
    exponents.append(np.polyfit(r_plot[lower:upper+1], np.log(corr[lower:upper+1]), 1)[0])
    exponents_2.append(scipy.optimize.curve_fit(lambda t,a,b: a*np.exp(b*t), r_plot[:upper], corr[:upper])[0][1])

K_avg_range = [2.5, 3.5]
seed_range = np.arange(1,11,1)

for K_avg in K_avg_range:
    K = str(K_avg) + "_" + str(K_std)
    r_plot, corr = fun.read_corr_density(mode, nPart, phi, noise, K, Rp, xTy, seed_range)
    lower = np.where(np.array(r_plot) > r_lower)[0][0]
    upper = np.where(np.array(r_plot) < r_upper)[0][-1]
    exponents.append(np.polyfit(r_plot[lower:upper+1], np.log(corr[lower:upper+1]), 1)[0])
    exponents_2.append(scipy.optimize.curve_fit(lambda t,a,b: a*np.exp(b*t), r_plot[:upper], corr[:upper])[0][1])

K_avg_range = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5]

plt.plot(K_avg_range, exponents, "o-", label="numpy polyfit")
plt.plot(K_avg_range, exponents_2, "o-", label="scipy curve_fit")
plt.legend()
plt.xlabel("K_avg")
plt.ylabel("Exponent")
plt.show()



print("Time taken: " + str(time.time() - t0))
