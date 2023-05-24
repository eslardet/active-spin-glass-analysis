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
nPart = 1000
phi = 1.0
# noise_range = [format(i, '.3f') for i in np.arange(0.80,0.811,0.01)]
noise = "0.20"
# noise_range = np.arange(0.80,0.01,0.82)
Rp = 1.0
K = "0.0_8.0"
xTy = 1.0
seed = 1
r_max = 1
tape_time = 150

# fun.write_contacts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, tape_time)

# ix, jx, contact_duration, r_max, tape_time = fun.read_contacts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max)
# contact_t = [i for i,v in enumerate(contact_duration) if v==tape_time]
# ixs = [int(ix[k]) for k in contact_t]
# jxs = [int(jx[k]) for k in contact_t]
# full_contact = np.unique(ixs+jxs)
# fun.animate_highlight(mode, nPart, phi, noise, K, Rp, xTy, seed, h=full_contact)

fun.plot_contacts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, log=True)

# print(np.max(contact_duration))
# print(np.unique(contact_duration, return_counts=True)[:10])
# t0 = time.time()
# fun.plot_K_vs_contact_time(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, log=True)
# print(time.time()-t0)