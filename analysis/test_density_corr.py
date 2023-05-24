import numpy as np
import analysis_functions_vicsek_new as fun
import matplotlib.pyplot as plt
import time
import scipy.stats as sps
import os
import freud
import analysis_functions_vicsek_new as fun
from matplotlib import cm, colors

mode = "G"
nPart = 10000
phi = 1.0
# noise_range = [format(i, '.3f') for i in np.arange(0.80,0.811,0.01)]
noise = "0.20"
# noise_range = np.arange(0.80,0.01,0.82)
Rp = 1.0
K = "0.0_8.0"
xTy = 1.0
seed = 1
rho_r_max = 1
samples = 10000
corr_r_max = 10
xscale = "lin"
yscale = "log"
r_bin_num = 20

# inparFile, posFile = fun.get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)

# fun.animate(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
# fun.plot_porder_time(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)

t0 = time.time()

# rij_all = []
# corr_all = []
# corr_r_max_sq = corr_r_max**2

# posFileExact = fun.get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, file_name='pos_exact')
# x, y, theta, view_time = fun.get_pos_ex_snapshot(file=posFileExact)

# L = np.sqrt(nPart / (phi*xTy))
# Ly = L
# Lx = L*xTy
# box = freud.Box.from_box([Lx, Ly])
# ld = freud.density.LocalDensity(r_max=rho_r_max, diameter=0)

# rng = np.random.default_rng(seed=1)
# if samples == None:
#     samples = nPart
# rand_points = np.zeros((samples, 3))
# rand_points[:,0] = rng.uniform(-Lx/2,Lx/2,samples)
# rand_points[:,1] = rng.uniform(-Ly/2,Ly/2,samples)

# points = np.zeros((nPart, 3))
# points[:,0] = x
# points[:,1] = y
# points = box.wrap(points)

# # get local densities
# rho_all = ld.compute(system=(box, points), query_points=rand_points).density

# rho_mean = np.mean(rho_all)
# d_fluc = [rho - rho_mean for rho in rho_all]
# # print(type(d_fluc))

# corr_dot = d_fluc

# # normalization
# c0 = 0
# for i in range(samples):
#     c0 += corr_dot[i] * corr_dot[i]
# c0 = c0/samples

# for i in range(samples):
#     for j in range(i+1, samples):
#         xij = rand_points[i,0] - rand_points[j,0]
#         xij = xij - Lx*round(xij/Lx)
#         if xij < corr_r_max:
#             yij = rand_points[i,1] - rand_points[j,1]
#             yij = yij - Ly*round(yij/Ly)
#             rij_sq = xij**2 + yij**2
#             if rij_sq < corr_r_max_sq:
#                 rij = np.sqrt(rij_sq)
#                 rij_all.append(rij)
#                 corr_all.append(corr_dot[i]*corr_dot[j]/c0)

# corr_all = np.array(corr_all)
# rij_all = np.array(rij_all)
# corr_bin_av = []
# bin_size = corr_r_max / r_bin_num

# if xscale == 'lin':
#     r_plot = np.linspace(0, corr_r_max, num=r_bin_num, endpoint=False) + bin_size/2
# elif xscale == 'log':
#     r_plot = np.logspace(-5, np.log10(corr_r_max), num=r_bin_num, endpoint=True)
# else:
#     raise Exception("xscale type not valid")

# for i in range(r_bin_num):
#     lower = r_plot[i]
#     try:
#         upper = r_plot[i+1]
#     except:
#         upper = corr_r_max+1
#     idx = np.where((rij_all>lower)&(rij_all<upper))
#     corr = np.mean(corr_all[idx])
#     corr_bin_av.append(corr)

# fig, ax = plt.subplots()
# ax.plot(r_plot, np.abs(corr_bin_av), '-')

# if xscale == 'log':
#     ax.set_xscale('log')
# if yscale == 'log':
#     ax.set_yscale('log')
# else:
#     ax.set_ylim(bottom=0)

# ax.set_xlabel(r"$r$")
# ax.set_ylabel(r"$C(r)$")

fun.plot_corr_density(mode, nPart, phi, noise, K, Rp, xTy, seed_range=range(1,2), yscale="log")

print(time.time()-t0)

# plt.show()