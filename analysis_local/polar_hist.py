import sys
sys.path.insert(1, './analysis/analysis_functions')
from stats import *

import numpy as np
import os
import matplotlib.pyplot as plt
import time
import csv


mode = "G"
nPart = 10000
phi = 1.0
noise = "0.20"
K = "0.0_1.0"
Rp = 1.0
xTy = 1.0
seed = 1

pos_ex_file = get_file_path(mode, nPart, phi, noise, K, Rp, xTy, seed, file_name='pos_exact')
x, y, theta, view_time = get_pos_ex_snapshot(pos_ex_file)

fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
ax.set_yticklabels([])

theta_wrap = [t%(2*np.pi) for t in theta]

ax.hist(theta_wrap, bins=100, ec='k')

plt.show()