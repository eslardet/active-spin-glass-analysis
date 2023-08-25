import numpy as np
from analysis.analysis_functions import *
import os
import matplotlib.pyplot as plt
import sys
import time


mode = str(sys.argv[1])
nPart = int(sys.argv[2])
phi = float(sys.argv[3])
noise = sys.argv[4]
K = str(sys.argv[5]) + "_" + str(sys.argv[6])
Rp = sys.argv[7]
xTy = float(sys.argv[8])
seed = int(sys.argv[9])
r_max = float(sys.argv[10])
tape_time = int(sys.argv[11])

#write_stats(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, remove_pos=True, moments=False)

#write_contacts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, tape_time)

#plot_contacts(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, log=True)

plot_K_vs_contact_time(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, log=False)
plot_K_vs_contact_time(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, log=True)

del_files(mode, nPart, phi, noise, K, Rp, xTy, seed, files=["coupling"])
