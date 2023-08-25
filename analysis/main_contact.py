import numpy as np
from analysis.analysis_functions import *
import matplotlib.pyplot as plt
import time


mode = "G"
nPart = 10000
phi = 1.0
noise = "0.20"
K = "-1.0_8.0"
Rp = 1.0
xTy = 1.0
seed = 1
r_max = 1
tape_time = 500

t0 = time.time()
write_contacts(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, r_max=r_max, tape_time=tape_time)

plot_contacts(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, r_max=r_max, log=True)

plot_K_vs_contact_time(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, log=False)
plot_K_vs_contact_time(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, log=True)

del_files(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, files=["pos", "coupling"])

print(time.time()-t0)
