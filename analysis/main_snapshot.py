import numpy as np
from analysis_functions import *
import matplotlib.pyplot as plt



mode = "G"
nPart = 10000
phi = 1.0
phi_range = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 10.0]
noise = "0.20"
noise_range = [format(i, '.2f') for i in np.arange(0.1,0.8,0.02)]
K_avg_range = ["-0.5","-0.2","0.0", "0.5"]
# K_avg_range = np.round(np.arange(4.5,5.1,0.5),1)
K_avg = 0.0
K="1.0_0.0"
K_std = 8.0
#K_std_range = np.arange(1.0, 8.1, 1.0)
#K_std_range = [0.0,1.0]
#K = "0.05_0.0"
# Rp_range = np.round(np.arange(1.0,11.0,1.0),1)
# Rp_range = np.concatenate((np.round(np.arange(0.1,1.1,0.1),1), np.arange(2.0,10.1,1.0)))
# Rp_range = np.concatenate((np.round(np.arange(0.2,2.1,0.2),1), np.arange(3.0,4.1,1.0)))
Rp = 1.0
xTy = 1.0
seed = 1
#r_max = 2

#neighbour_hist(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max=2)

#stats_list = []

for noise in noise_range:
    #mean, med, std, nmax = neighbour_stats(mode, nPart, phi, noise, K, Rp, xTy, seed, pos_ex=True))
    snapshot(mode, nPart, phi, noise, K, Rp, xTy, seed, pos_ex=True)
    #neighbour_hist(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max=r_max)
    #neighbour_hist(mode, nPart, phi, noise, K, xTy, seed, r_max=r_max)
#print("Mean")
#for stat in stats_list:
#    print(stat[0])

#print("\nSD")
#for stat in stats_list:
#    print(stat[1])

#print("\nMax")
#for stat in stats_list:
#    print(stat[2])

