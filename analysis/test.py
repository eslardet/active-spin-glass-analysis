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
noise_range = [format(i, '.3f') for i in np.arange(0.80,0.811,0.01)]
# noise_range = np.arange(0.80,0.01,0.82)
Rp = 1.0
K = "1.0_0.0"
xTy = 1.0
seed = 101

print(noise_range)

