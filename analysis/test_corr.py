import numpy as np
import analysis_functions_vicsek_new as fun
import matplotlib.pyplot as plt
import time
import scipy.stats as sps
import os
import freud
import analysis_functions_vicsek_new as fun
from matplotlib import cm, colors

t0 = time.time()

mode = "G"
nPart = 10000
phi = 1.0
noise = "0.20"
Rp = 1.0
K = "0.0_1.0"
xTy = 1.0
seed = 1
r_max = 20
d_type = 'dv_par'

posFileExact = fun.get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, file_name='pos_exact')
x, y, theta, view_time = fun.get_pos_ex_snapshot(file=posFileExact)

L = np.sqrt(nPart / (phi*xTy))
Ly = L
Lx = L*xTy

velocity = [np.array([np.cos(p), np.sin(p)]) for p in theta]
av_vel = np.mean(velocity, axis=0)

dv = [v - av_vel for v in velocity]

av_unit = av_vel / np.linalg.norm(av_vel)
av_norm = np.array([-av_vel[1], av_vel[0]])

if d_type == 'v':
    corr_dot = velocity
elif d_type == 'dv':
    corr_dot = dv
elif d_type == 'dv_par':
    corr_dot = [np.dot(f, av_unit) * av_unit for f in dv]
elif d_type == 'dv_perp':
    corr_dot = [np.dot(f, av_norm) * av_norm for f in dv]
else:
    raise Exception("Type not valid. Must be 'v', 'dv', 'dv_par', or 'dv_perp'")

rij_all = []
corr_all = []

c0 = 0
for i in range(nPart):
    c0 += np.dot(corr_dot[i], corr_dot[i])
c0 = c0/nPart

for i in range(nPart):
    for j in range(i+1, nPart):
        xij = x[i] - x[j]
        xij = xij - Lx*round(xij/Lx)
        yij = y[i] - y[j]
        yij = yij - Ly*round(yij/Ly)
        rij = np.sqrt(xij**2 + yij**2)
        if rij < r_max:
            rij_all.append(rij)
            corr_all.append(np.dot(corr_dot[i],corr_dot[j])/c0)
            # corr_all.append(np.dot(corr_dot[i],corr_dot[j]))

r_bin_num = 20

corr_all = np.array(corr_all)
rij_all = np.array(rij_all)
corr_bin_av = []
bin_size = r_max / r_bin_num


# r_plot = np.linspace(0, r_max, num=r_bin_num, endpoint=False) + bin_size/2
r_plot = np.logspace(-5, np.log10(r_max), num=r_bin_num, endpoint=True)

for i in range(r_bin_num):
    lower = r_plot[i]
    try:
        upper = r_plot[i+1]
    except:
        upper = r_max+1
    idx = np.where((rij_all>lower)&(rij_all<upper))
    corr = np.mean(corr_all[idx])
    corr_bin_av.append(corr)

fig, ax = plt.subplots()
ax.plot(r_plot, np.abs(corr_bin_av), '-')

ax.set_ylim(bottom=0)

ax.set_xlabel(r"$r$")
ax.set_ylabel(r"$C(r)$ for " + d_type)
ax.set_xscale('log')
ax.set_yscale('log')

print("Time taken: " + str(time.time()-t0))

plt.show()
