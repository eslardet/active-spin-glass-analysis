import sys
sys.path.insert(1, './analysis_functions')
from local_order_grid import *

import numpy as np
import time


mode = "G"
nPart = 10000
phi = 1.0
noise = "0.20"
K = "0.0_8.0"
Rp = 1.0
xTy = 1.0
seed = 1

seed_range = np.arange(1,21,1)
r_max_range = np.arange(1,21,1)
# r_max_range = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0]

K_avg_range = [0.0]
K_std_range = np.arange(0.0, 8.1, 1.0)
t0 = time.time()
plot_local_order_vs_l(mode, nPart, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed_range, r_max_range)
print(time.time()-t0)

K_avg_range = [-1.0, -0.5, 0.0, 0.5, 1.0]
K_std_range = [8.0]

t0 = time.time()
plot_local_order_vs_l(mode, nPart, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed_range, r_max_range)
print(time.time()-t0)

# t0 = time.time()
# for K_avg in K_avg_range:
#     for K_std in K_std_range:
#         plot_local_order_hist(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed_range, r_max_range)

# print(time.time()-t0)
