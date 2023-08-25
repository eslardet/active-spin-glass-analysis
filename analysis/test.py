import numpy as np
from analysis_functions import *
import matplotlib.pyplot as plt
import time
import scipy.stats as sps
import os
import freud

mode = "G"
nPart = 1000
phi = 1.0
noise = "0.20"
K = "1.0_0.0"
xTy = 5.0
seed = 1
min_grid_size=2

posFileExact = get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed, file_name="pos_exact")


L = np.sqrt(nPart / (phi*xTy))
Ly = L
Lx = L*xTy

fig, ax = plt.subplots()

x, y, theta, viewtime = get_pos_ex_snapshot(posFileExact)

# x = pbc_wrap(x,Lx)
# y = pbc_wrap(y,Ly)

# ngrid_x = int(Lx // min_grid_size)
# grid_size_x = Lx / ngrid_x
# ngrid_y = int(Ly // min_grid_size)
# grid_size_y = Ly / ngrid_y

# grid_area = grid_size_x*grid_size_y

# grid_counts = np.zeros((ngrid_x, ngrid_y))

# for i in range(nPart):
#     gridx = int(x[i]//grid_size_x)
#     gridy = int(y[i]//grid_size_y)
#     grid_counts[gridx,gridy] += 1
# n_density = grid_counts / grid_area

r_max = 2
r_max_sq = r_max**2

t0 = time.time()
points = np.zeros((nPart, 3))
points[:,0] = x
points[:,1] = y
box = freud.Box.from_box([Lx, Ly])
points = box.wrap(points)
ld = freud.density.LocalDensity(r_max=r_max, diameter=0)
n_nei = ld.compute(system=(box, points)).num_neighbors
print(n_nei[:10])
print("time=" + str(time.time()-t0))

t0 = time.time()
nei = np.zeros(nPart)
for i in range(nPart):
    for j in range(i+1, nPart):
        xij = x[i]-x[j]
        xij = pbc_wrap(xij, Lx)
        if np.abs(xij) < r_max:
            yij = y[i]-y[j]
            yij = pbc_wrap(yij, Ly)
            rij_sq = xij**2+yij**2
            if rij_sq <= r_max_sq:
                nei[i] += 1
                nei[j] += 1
print(nei[:10])
print("time=" + str(time.time()-t0))


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