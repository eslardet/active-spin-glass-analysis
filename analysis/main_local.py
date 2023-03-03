import numpy as np
import numpy as np
import analysis_functions_vicsek as fun
import os
import matplotlib.pyplot as plt
import sys

mode = 'C'
nPart = 10000
phi = 1.0
noise = "0.80"
K = 1.0
xTy = 5.0
seed = 12


# fun.plot_density_profile(mode=mode, nPart=nPart, phi=phi, noise=noise, K=K, xTy=xTy, seed=seed)
fun.animate(mode, nPart, phi, noise, K, xTy, seed)
# fun.write_stats(mode, nPart, phi, noise, K, xTy, seed, density_var=False)
