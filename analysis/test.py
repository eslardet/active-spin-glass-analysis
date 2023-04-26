import numpy as np
import analysis_functions_vicsek_new as fun
import matplotlib.pyplot as plt
import time
import scipy.stats as sps
import os
import freud

mode = "G"
nPart = 10
phi = 1.0
noise = "0.20"
Rp = 1.0
K = "0.0_1.0"
xTy = 1.0
seed = 1
min_grid_size=2

# posFileExact = fun.get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed, file_name="pos_exact")


# L = np.sqrt(nPart / (phi*xTy))
# Ly = L
# Lx = L*xTy

# fig, ax = plt.subplots()

# x, y, theta, viewtime = fun.get_pos_ex_snapshot(posFileExact)


# r_max = 2
# r_max_sq = r_max**2

# t0 = time.time()
# points = np.zeros((nPart, 3))
# points[:,0] = x
# points[:,1] = y
# box = freud.Box.from_box([Lx, Ly])
# points = box.wrap(points)
# ld = freud.density.LocalDensity(r_max=r_max, diameter=0)
# n_nei = ld.compute(system=(box, points)).num_neighbors
# print(n_nei[:10])
# print("time=" + str(time.time()-t0))

# t0 = time.time()
# nei = np.zeros(nPart)
# for i in range(nPart):
#     for j in range(i+1, nPart):
#         xij = x[i]-x[j]
#         xij = fun.pbc_wrap(xij, Lx)
#         if np.abs(xij) < r_max:
#             yij = y[i]-y[j]
#             yij = fun.pbc_wrap(yij, Ly)
#             rij_sq = xij**2+yij**2
#             if rij_sq <= r_max_sq:
#                 nei[i] += 1
#                 nei[j] += 1
# print(nei[:10])
# print("time=" + str(time.time()-t0))


# fig, ax = plt.subplots()
# ax.hist(n_density, bins=100)

# n,x = np.hist(n_density, bins=100)
# bin_centers = 0.5*(x[1:]+x[:-1])
# ax.plot(bin_centers,n)


# ax.set_xlabel("Number density")
# ax.set_ylabel("Probability density")

# folder = os.path.abspath('../plots/density_distribution/')
# filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_xTy' + str(xTy) + '_s' + str(seed) + '.png'
# if not os.path.exists(folder):
#     os.makedirs(folder)
# plt.savefig(os.path.join(folder, filename))

# num = 50000
# rng = np.random.default_rng(seed=1)

# t0 = time.time()
# for i in range(num):
#     a = rng.uniform(0,5,1)

# print(time.time()-t0)

# t0 = time.time()
# a = rng.uniform(0,5,num)
# for i in range(num):
#     b = a[i]

# print(time.time()-t0)



posFileExact = fun.get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, file_name='pos_exact')
x, y = fun.get_pos_ex_snapshot(file=posFileExact)[:2]

x_all = [x]

# inparFile, posFile = fun.get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)

# x_all = []
# for i in range(10):
#     x, y, theta = fun.get_pos_snapshot(posFile, nPart, timestep=i)
#     x_all.append(x)

print(np.arange(-5,0,1))