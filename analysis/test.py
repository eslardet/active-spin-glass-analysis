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
noise = "0.20"
Rp = 1.0
K = "0.0_8.0"
xTy = 1.0
seed = 1


# # inparFile, posFile = fun.get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
# posFileExact = fun.get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, file_name="pos_exact")
# x, y, theta, view_time = fun.get_pos_ex_snapshot(posFileExact)

# # fun.snapshot(mode, nPart, phi, noise, K, Rp, xTy, seed)

# L = np.sqrt(nPart / (phi*xTy))
# Ly = L
# Lx = L*xTy

# indices = []
# x_c = []
# y_c = []
# p_c = []

# for i in range(nPart):
#     xw = fun.pbc_wrap(x[i], L)
#     yw = fun.pbc_wrap(y[i], L)
#     if xw < 42 and xw>38:
#         if yw< 70 and yw>68:
#             indices.append(i)
#             x_c.append(xw)
#             y_c.append(yw)
#             p_c.append(theta[i])

# u = np.cos(p_c)
# v = np.sin(p_c)

# num_nei = fun.neighbour_counts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max=1)

# sim_dir = fun.get_sim_dir(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
# couplingFile = os.path.join(sim_dir, "coupling")


# with open(couplingFile) as f:
#     num = len(indices)
#     # K_matrix = np.zeros((num, num))
#     K_list = []
#     K_all = []
#     line = 0
#     i = 0
#     j = i+1
#     k = nPart-1
#     start_row = 0
#     K_list = []
#     for Kij in f:
#         if i in indices and j in indices:
#             # K_matrix[indices.index(i)][indices.index(j)] = float(Kij)
#             K_list.append(float(Kij))
#         K_all.append(Kij)
#         line += 1
#         if line == start_row + k:
#             i += 1
#             j = i+1
#             k -= 1
#             start_row = line
#         else:
#             j += 1

# # plt.hist(Kij, bins=100)
# plt.hist(K_all, bins=100)
# plt.show()
# norm = colors.Normalize(vmin=0.0, vmax=np.max(num_nei), clip=True)
# mapper = cm.ScalarMappable(norm=norm, cmap=cm.plasma)
# cols = mapper.to_rgba(num_nei)

# cols = [cols[i] for i in indices]

# fig, ax = plt.subplots(figsize=(10*xTy,10), dpi=72)
# # ax.scatter(x_c, y_c, s=0.1)
# ax.quiver(x_c, y_c, u, v, color=cols)
# ax.set_xlim(0,Lx)
# ax.set_ylim(0,Ly)
# plt.show()


arr = np.arange(0.040, 0.050, 0.0005)
print(format(arr[6], '.4f'))