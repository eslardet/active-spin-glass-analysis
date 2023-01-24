import numpy as np
import analysis_functions as fun
import os
import matplotlib.pyplot as plt
import sys

mode = 'C'
nPart = 6000
phi = 0.1
Pe = 50.0
K = '10.0'
seed = 1

# fun.dist_coupling(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, seed=seed, avg_over=avg_over)

# couplings = fun.read_couplings(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, seed=seed)

# print(len(couplings))

# rij = fun.rij_avg(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, seed=seed, avg_over=avg_over)

# print(len(rij))
# print(rij)

# fun.plot_dist_coupling(mode, nPart, phi, Pe, KAVG, KSTD, seed, avg_over)

# fun.snapshot_pos_ex(mode=mode,nPart=nPart,phi=phi,Pe=Pe,K=K,seed=seed)
fun.snapshot(mode,nPart,phi,Pe,K,seed,view_time=6)
# fun.animate(mode,nPart,phi,Pe,K,seed,max_T=100)