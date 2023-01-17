import numpy as np
import analysis_functions as fun
import os
import matplotlib.pyplot as plt
import sys

mode = 'G'
nPart = 100
phi = 0.2
Pe = 20.0
K = '1.0_1.0'
seed = 1
avg_over=1000

# fun.dist_coupling(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, seed=seed, avg_over=avg_over)

couplings = fun.read_couplings(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, seed=seed)

print(len(couplings))

rij = fun.rij_avg(mode=mode, nPart=nPart, phi=phi, Pe=Pe, K=K, seed=seed, avg_over=avg_over)

print(len(rij))
print(rij)