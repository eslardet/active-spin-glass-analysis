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
samples = 100

# inparFile, posFile = fun.get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)

# fun.animate(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
# fun.plot_porder_time(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)

posFileExact = fun.get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, file_name='pos_exact')
x, y, theta, view_time = fun.get_pos_ex_snapshot(file=posFileExact)



L = np.sqrt(nPart / (phi*xTy))
Ly = L
Lx = L*xTy
box = freud.Box.from_box([Lx, Ly])
ld = freud.density.LocalDensity(r_max=rho_r_max, diameter=0)

rng = np.random.default_rng(seed=1)
if samples == None:
    samples = nPart
rand_points = np.zeros((samples, 3))
rand_points[:,0] = rng.uniform(-Lx/2,Lx/2,samples)
rand_points[:,1] = rng.uniform(-Ly/2,Ly/2,samples)

points = np.zeros((nPart, 3))
points[:,0] = x
points[:,1] = y
points = box.wrap(points)

# get local density
# rho = 1 # placeholder
rho_all = ld.compute(system=(box, points), query_points=rand_points).density
d_fluc = [rho - phi for rho in rho_all]
# print(type(d_fluc))

corr_dot = rho_all
print(rho_all)

c0 = 0
for i in range(samples):
    c0 += np.dot(corr_dot[i], corr_dot[i])
c0 = c0/nPart

print(c0)