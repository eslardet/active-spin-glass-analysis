import numpy as np
import sys
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import csv
# sys.path.insert(0, os.path.abspath('../analysis/'))
from analysis.analysis_functions import *



mode = "G"
nPart = 10000
phi = 1.0
noise = "0.20"
K = "0.0_8.0"
Rp = 1.0
xTy = 1.0
seed = 1

snapshot(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, pos_ex=True)
inparFile, posFile = get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
posFileExact = get_file_path(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, file_name="pos_exact")
inpar_dict = get_params(inparFile)

nPart = inpar_dict["nPart"]
phi = inpar_dict["phi"]
noise = inpar_dict["noise"]
mode = inpar_dict["mode"]
DT = inpar_dict["DT"]
eqT = inpar_dict["eqT"]
xTy = inpar_dict["xTy"]
simulT = inpar_dict["simulT"]

L = np.sqrt(nPart / (phi*xTy))
Ly = L
Lx = L*xTy

x, y, theta, view_time = get_pos_ex_snapshot(file=posFileExact)

x = pbc_wrap(x,Lx)
y = pbc_wrap(y,Ly)
u = np.cos(theta)
v = np.sin(theta)


matplotlib.rcParams["font.family"] = "serif"
fig, ax = plt.subplots(figsize=(1,1), dpi=1000)

norm = colors.Normalize(vmin=0.0, vmax=2*np.pi, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.hsv)
cols = mapper.to_rgba(np.mod(theta, 2*np.pi))
ax.quiver(x, y, u, v, color=cols, scale=40, minlength=0, width=50)
# plt.colorbar(mappable=mapper, ax=ax)
ax.set_xlim(0,Lx/5)
ax.set_ylim(Ly*4/5,Ly)
ax.set_aspect('equal')

# plt.axis('off')
plt.tick_params(top=False, bottom=False, left=False, right=False,
                labelleft=False, labelbottom=False)
plt.tight_layout()

folder = os.path.abspath('../plots/for_figures/snaps')
filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_n' + str(noise) + '_K' + str(K) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '_s' + str(seed) + '.pdf'
if not os.path.exists(folder):
    os.makedirs(folder)
plt.savefig(os.path.join(folder, filename),bbox_inches='tight', transparent=True,pad_inches=0)