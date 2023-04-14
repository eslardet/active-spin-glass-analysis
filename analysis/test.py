import numpy as np
import analysis_functions_vicsek as fun
import matplotlib.pyplot as plt
import time


# mode = "G"
# nPart = 10000
# phi = 1.0
# noise = "0.20"
# K = "0.0_8.0"
# xTy=5.0
# seed=1

# t0 = time.time()
# print(fun.neighbour_stats(mode, nPart, phi, noise, K, xTy, seed, pos_ex=True))

# print(time.time()-t0)

nPart = 2

Lx = 5
Ly = 5

x = [0, 0]
y = [0, 0]
theta = [0, np.pi/2]

# L = np.sqrt(nPart / (phi*xTy))
# Ly = L
# Lx = L*xTy

velocity = [np.array([np.cos(p), np.sin(p)]) for p in theta]
av_vel = np.mean(velocity, axis=0)

dv = [v - av_vel for v in velocity]

av_unit = av_vel / np.linalg.norm(av_vel)
av_norm = np.array([-av_vel[1], av_vel[0]])

# fluc_par = [np.dot(f, av_unit) * av_unit for f in fluc_vel]
dv_par = [np.dot(f, av_unit) * av_unit for f in dv] 
dv_perp = [np.dot(f, av_norm) * av_norm for f in dv]


fig, ax = plt.subplots()

rij_all = []
corr_all = []
r_max = 20

for i in range(nPart):
    for j in range(0, nPart):
        xij = x[i] - x[j]
        xij = xij - Lx*round(xij/Lx)
        yij = y[i] - y[j]
        yij = yij - Ly*round(yij/Ly)
        rij = np.sqrt(xij**2 + yij**2)
        if rij < r_max:
            rij_all.append(rij)
            corr_all.append(np.dot(dv_perp[i],dv_perp[j]))
        #     print(i,j,dv_perp[i],dv_perp[j],np.dot(dv_perp[i],dv_perp[j]))
        #     print(i,j,velocity[i],velocity[j],np.dot(velocity[i],velocity[j]))
        # ax.plot(rij, corr, '+', color='tab:blue', alpha=0.2)

# corr_all = np.array(corr_all)
# rij_all = np.array(rij_all)
# r_bin_num = int(r_max)
# print(r_bin_num)
# corr_bin_av = []
# bin_size = r_max / r_bin_num
# for i in range(r_bin_num):
#     lower = bin_size*i
#     upper = bin_size*(i+1)
#     idx = np.where((rij_all>lower)&(rij_all<upper))
#     corr = np.mean(corr_all[idx])
#     corr_bin_av.append(corr)

# r_plot = np.linspace(0, r_max, num=r_bin_num, endpoint=False) + bin_size/2

# # if scatter == True:
# ax.plot(rij_all, corr_all, '+', alpha=0.2)
# ax.plot(r_plot, corr_bin_av, '-')


# ax.set_xlabel(r"$r$")
# ax.set_ylabel(r"$C_{\perp}(r)$")

# plt.show()

print(np.logspace(-5, np.log10(10), num=7, endpoint=True))