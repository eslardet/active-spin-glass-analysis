import numpy as np
import analysis_functions_vicsek_rep as fun
import matplotlib.pyplot as plt



mode = "G"
nPart = 1000
phi = 0.1
noise = "0.20"
K_avg = 0.0
K_std_range = np.arange(1.0, 8.1, 1.0)
# K = "0.0_8.0"
Rp = 2.0
xTy=5.0
seed=1

stats_list = []

K = "0.0_1.0"
# fun.neighbour_hist(mode, nPart, phi, noise, K, xTy, seed)
# fun.snapshot(mode, nPart, phi, noise, K, xTy, seed, pos_ex=True)

K = "2.0_1.0"
fun.neighbour_hist(mode, nPart, phi, noise, K, xTy, seed)
fun.snapshot(mode, nPart, phi, noise, K, xTy, seed, pos_ex=True)
print(fun.neighbour_stats(mode, nPart, phi, noise, K, xTy, seed, pos_ex=True))

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

