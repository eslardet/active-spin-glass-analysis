import numpy as np
sys.path.insert(1, '/Users/el2021/Code/2D_ActiveSpinGlass_EL/Active_Spin_Glass/analysis')
from analysis_functions import *
import matplotlib.pyplot as plt



mode = "G"
nPart = 10000
phi = 1.0
noise = "0.20"
K = "0.0_8.0"
Rp = 1.0
xTy = 1.0
seed = 1
r_max = 1
tape_time = 1

# write_contacts(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, r_max=r_max, tape_time=tape_time)

plot_contacts(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, Rp=Rp, xTy=xTy, seed=seed, r_max=r_max, log=True)

# plot_K_vs_contact_time(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max, log_x=True, log_y=True)