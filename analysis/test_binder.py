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
nPart = 90000
phi = 1.0
# noise_range = [format(i, '.3f') for i in np.arange(0.80,0.811,0.01)]
noise = "0.81"
# noise_range = np.arange(0.80,0.01,0.82)
Rp = 1.0
K = "1.0_0.0"
xTy = 1.0
seed = 101

min_T = 2200
max_T = None

inparFile, posFile = fun.get_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed)
inpar_dict = fun.get_params(inparFile)
DT = inpar_dict["DT"]
simulT = inpar_dict["simulT"]
if min_T == None:
    min_T = 0
if max_T == None:
    max_T = simulT

p_order = []

with open(posFile) as f:
    line_count = 1
    timestep = int(min_T//DT)
    for line in f:
        if 8 + timestep*(nPart + 1) <= line_count <= 7 + timestep*(nPart + 1) + nPart:
            if line_count == 8 + timestep*(nPart+1):
                cos_sum = 0
                sin_sum = 0
                cos_sq_sum = 0
                cos_sin_sum = 0
            theta = float(line.split('\t')[2])
            cos_sum += np.cos(theta)
            sin_sum += np.sin(theta)
            cos_sq_sum += np.cos(theta)**2
            cos_sin_sum += np.sin(theta)*np.cos(theta)
            if line_count == 7 + timestep*(nPart + 1) + nPart:
                p_order.append(np.sqrt(cos_sum**2+sin_sum**2)/nPart)
                timestep += 1
        line_count += 1
        if timestep*DT > max_T:
            break
fig, ax = plt.subplots()
p_mean = np.mean(p_order)
p_second = np.mean([p**2 for p in p_order])
p_third = np.mean([p**3 for p in p_order])
p_fourth = np.mean([p**4 for p in p_order])
print(1 - p_fourth/(3*(p_second**2)))

# binder = [1 - p**4/(3*((p**2)**2)) for p in p_order]

# t_plot = np.arange(0, max_T+DT/4, DT)
# # ax.plot(t_plot, p_order, label="1")
# ax.plot(t_plot, [p**2 for p in p_order], label="2")
# ax.plot(t_plot, [p**4 for p in p_order], label="4")
# # ax.plot(t_plot, binder)

# ax.set_xlabel("time")
# ax.set_ylabel(r"Polar order parameter, $\Psi$")
# ax.legend()

# plt.show()