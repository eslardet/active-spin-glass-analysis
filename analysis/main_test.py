import numpy as np
import analysis_functions_lattice as fun
import os
import matplotlib.pyplot as plt
import sys
import time

mode = "C"
nPart = 1024
Rp = 2.0
rotD = 0.0
K = -10.0
seed = 1

# fun.dist_coupling(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, seed=seed, avg_over=avg_over)

# couplings = fun.read_couplings(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, seed=seed)

# print(len(couplings))

# rij = fun.rij_avg(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, seed=seed, avg_over=avg_over)

# print(len(rij))
# print(rij)

# fun.plot_dist_coupling(mode, nPart, phi, Pe, KAVG, KSTD, seed, avg_over)

# fun.snapshot_pos_ex(mode=mode,nPart=nPart,phi=phi,Pe=Pe,K=K,seed=seed)
# inparFile, posFile = fun.get_files(mode=mode,nPart=nPart,phi=phi,Pe=Pe,K=K,seed=seed)
# fun.get_pos_arr(inparFile, posFile, min_T=None, max_T=None)

# fun.snapshot(mode,nPart,phi,Pe,K,seed,view_time=6)
# fun.animate(mode,nPart,phi,Pe,K,seed,max_T=100)

t0 = time.time()
p1 = fun.plot_porder_time(mode, nPart, K, Rp, rotD, seed, min_T=0, max_T=10)
print(time.time() - t0)


t1 = time.time()
p2 = fun.plot_porder_time_large(mode, nPart, K, Rp, rotD, seed, min_T=0, max_T=10)
print(time.time() - t1)

print(p1 == p2)