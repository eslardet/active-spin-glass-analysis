import numpy as np
import analysis_functions as fun
import os
import matplotlib.pyplot as plt
import sys


# Particles on a hexagonal lattice

sigma=1 # diameter

Nx = 10
Ny = 10

Lx = Nx*sigma
Ly = np.sqrt(3)/2*Ny*sigma
xTy = Lx/Ly

Ntot = Nx*Ny

y = np.zeros(Ntot)
x = np.zeros(Ntot)

for i in range(Ny):
    for j in range(Nx):
        y[i*Nx+j] = i*np.sqrt(3)/2*sigma + sigma/2
        x[i*Nx+j] = j*sigma
        if i % 2 == 1:
            x[i*Nx+j] += sigma/2


fig, ax = plt.subplots(figsize=(5*xTy,5), dpi=72)

diameter = (ax.get_window_extent().height * 72/fig.dpi) /Ly *sigma
ax.set_xlim(0,Lx)
ax.set_ylim(0,Ly)
ax.plot(x, y, 'o', ms=diameter)
ax.plot(x, y, 'k.')
plt.show()
            