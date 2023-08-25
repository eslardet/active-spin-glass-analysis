import numpy as np
from analysis_functions import *
import matplotlib.pyplot as plt



mode = "G"
nPart = 1000
phi = 0.4
noise = "0.20"
#K_avg_range = ["-0.5","0.1","1.0"]
K_avg_range = np.round(np.arange(4.5,5.1,0.5),1)
K="1.0_0.0"
K_std = 8.0
#K_std_range = np.arange(1.0, 8.1, 1.0)
#K_std_range = [0.0,1.0]
#K = "0.05_0.0"
Rp_range = np.round(np.arange(1.0,11.0,1.0),1)
xTy = 1.0
seed = 2
#r_max = 2

#neighbour_hist(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max=2)

#stats_list = []

for Rp in Rp_range:
    #K = str(K_avg) + "_" + str(K_std)
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

