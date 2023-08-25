import numpy as np
import analysis_functions_vicsek as fun
import matplotlib.pyplot as plt



mode = "G"
nPart = 10000
phi = 1.0
noise = "0.20"
K_avg = 0.0
#K_std_range = np.arange(2.0, 8.1, 1.0)
K_std_range = [1.0, 4.0, 8.0]
#K = "0.0_8.0"
Rp = 1.0
xTy = 5.0
seed = 1
r_max = 1

#neighbour_hist(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max=2)

#stats_list = []

neighbour_hist_overlay(mode, nPart, phi, noise, K_avg, K_std_range, xTy, seed, r_max)

#for K_std in K_std_range:
    #K = str(K_avg) + "_" + str(K_std)
    #mean, med, std, nmax = neighbour_stats(mode, nPart, phi, noise, K, Rp, xTy, seed, pos_ex=True))
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

