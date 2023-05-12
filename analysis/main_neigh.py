import numpy as np
import analysis_functions_vicsek_new as fun
import matplotlib.pyplot as plt



mode = "G"
nPart = 10000
phi = 1.0
noise = "0.20"
K_avg = 0.0
K_std_range = [1.0, 2.0, 8.0]
# K = "0.0_8.0"
Rp = 1.0
xTy = 1.0
seed = 1
r_max = 5
n_max = None
c_max = None

stats_list = []

# K = "0.0_1.0"
# fun.neighbour_hist(mode, nPart, phi, noise, K, xTy, seed)
# fun.snapshot(mode, nPart, phi, noise, K, xTy, seed, pos_ex=True)

# for K_std in K_std_range:
K_std = 8.0
K = str(K_avg) + "_" + str(K_std)
# fun.neighbour_hist(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max=r_max, n_max=n_max, c_max=c_max)
fun.snapshot(mode, nPart, phi, noise, K, Rp, xTy, seed, pos_ex=True, neigh_col=True, r_max=r_max)
# print(fun.neighbour_stats(mode, nPart, phi, noise, K, xTy, seed, pos_ex=True))

# for K_std in K_std_range:
#     K = str(K_avg) + "_" + str(K_std)
#     stats_list.append(fun.neighbour_stats(mode, nPart, phi, noise, K, xTy, seed, pos_ex=True))

# print("Mean")
# for stat in stats_list:
#     print(stat[0])

# print("\nSD")
# for stat in stats_list:
#     print(stat[1])

# print("\nMax")
# for stat in stats_list:
#     print(stat[2])

