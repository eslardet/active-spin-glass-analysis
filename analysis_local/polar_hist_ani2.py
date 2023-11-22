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
nPart = 10000
phi = 1.0
noise = "0.20"
K = "0.0_8.0"
Rp = 1.0
xTy = 1.0
seed = 1

# pos_ex_file = get_file_path(mode, nPart, phi, noise, K, Rp, xTy, seed, file_name='pos_exact')
# x, y, theta, view_time = get_pos_ex_snapshot(pos_ex_file)

# inparFile, posFile = get_files(mode, nPart, phi, noise, K, Rp, xTy, seed)
# theta_all = get_theta_arr(inparFile, posFile, min_T=0, max_T=300)
# theta = [t for sublist in theta_all for t in sublist]

# theta_wrap = [t%(2*np.pi) for t in theta]


# print(np.rad2deg(circmean(theta_wrap)), np.rad2deg(circvar(theta_wrap)))

inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)

x_all, y_all, theta_all = get_pos_arr(inparFile=inparFile, posFile=posFile)

inpar_dict = get_params(inparFile)

nPart = inpar_dict["nPart"]
phi = inpar_dict["phi"]
noise = inpar_dict["noise"]
mode = inpar_dict["mode"]
DT = inpar_dict["DT"]
seed = inpar_dict["seed"]
xTy = inpar_dict["xTy"]

min_T=0

with open(posFile) as f:
    reader = csv.reader(f, delimiter="\t")
    startT = float(list(reader)[6][0])

plt.rcParams["animation.html"] = "jshtml"
plt.ioff()
plt.rcParams['animation.embed_limit'] = 2**128

L = np.sqrt(nPart / (phi*xTy))
Ly = L
Lx = L*xTy

fig, ax = plt.subplots(figsize=(10*xTy,10))

norm = colors.Normalize(vmin=0.0, vmax=2*np.pi, clip=True)
plt.set_cmap('hsv')

mapper = cm.ScalarMappable(norm=norm, cmap=cm.hsv)
# plt.colorbar(mappable=mapper, ax=ax)

x = pbc_wrap(x_all[0],Lx)
y = pbc_wrap(y_all[0],Ly)
theta = theta_all[0]
cols = np.mod(theta, 2*np.pi)
arrows = ax.quiver(x, y, np.cos(theta), np.sin(theta), norm(cols))

def init():
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    return arrows,

def update(n):
    x = pbc_wrap(x_all[n],Lx)
    y = pbc_wrap(y_all[n],Ly)
    theta = theta_all[n]
    arrows.set_offsets(np.c_[x, y])
    cols = np.mod(theta, 2*np.pi)
    arrows.set_UVC(np.cos(theta), np.sin(theta), norm(cols))
    ax.set_title("t = " + str(round(n*DT+startT, 1)), fontsize=10, loc='left')
    
    return arrows,

ani = FuncAnimation(fig, update, init_func=init, frames=len(theta_all), interval=50, blit=True)

folder = os.path.abspath('../animations/polar_hist')
filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed) + '_both.mp4'
if not os.path.exists(folder):
    os.makedirs(folder)
ani.save(os.path.join(folder, filename))



plt.rcParams["animation.html"] = "jshtml"
plt.ioff()
plt.rcParams['animation.embed_limit'] = 2**128

L = np.sqrt(nPart / (phi*xTy))
Ly = L
Lx = L*xTy

inparFile, posFile = get_files(mode, nPart, phi, noise, K, Rp, xTy, seed)
theta_all = get_theta_arr(inparFile, posFile, min_T=0, max_T=300)

fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
ax.set_yticklabels([])
ax.set_ylim(0, 250)
theta = theta_all[0]
theta_wrap = [t%(2*np.pi) for t in theta]
ax.hist(theta_wrap, bins=200, ec='k')

def update(n):
    theta = theta_all[n]
    theta_wrap = [t%(2*np.pi) for t in theta]
    ax.clear()
    ax.set_yticklabels([])
    ax.set_ylim(0, 250)
    ax.hist(theta_wrap, bins=200, ec='k')
    ax.set_title("t = " + str(n+3000), fontsize=10, loc='left')

ani = FuncAnimation(fig, update, frames=300, interval=50)

folder = os.path.abspath('../animations/polar_hist')
filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed) + '.mp4'
if not os.path.exists(folder):
    os.makedirs(folder)
ani.save(os.path.join(folder, filename))