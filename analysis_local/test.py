import sys
# sys.path.insert(1, '././analysis_functions')
sys.path.insert(1, './analysis/analysis_functions')
# from analysis.analysis_functions.import_files import *
from pt import *

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import time
from scipy.signal import fftconvolve, correlate2d, convolve2d
from matplotlib import colors

# N = 10000

# K1 = 1.0
# K2 = -1.0

# Kstd = 8*np.sqrt(2)

# vec1 = []
# for i in range(N):
#     vec1.append(np.random.normal(1,Kstd))

# vec2 = []
# for i in range(N):
#     vec2.append(np.random.normal(-1,Kstd))

# arr = np.zeros((N,N))
# for i in range(N):
#     for j in range(1):
#         arr[i,j] = (vec1[i]+vec2[j])/2

# print(np.mean(arr), np.std(arr))

# arr2 = np.zeros((N,N))
# for i in range(N):
#     for j in range(1):
#         arr2[i,j] = np.random.normal(0,8)

# print(np.mean(arr2), np.std(arr2))

# plt.hist(arr[:,0].flatten(), bins=100, alpha=0.5, label="2 vecs")
# plt.hist(arr2[:,0].flatten(), bins=100, alpha=0.5, label="1 arr")
# plt.legend()
# plt.show()

N = 6
K = np.zeros((N,N))

for i in range(N):
    K[i,i] = 0
    for j in range(i+1, N):
        if i < N/3:
            if j < N/3:
                K[i,j] = 1
                K[j,i] = 1
            elif j < 2*N/3:
                K[i,j] = 1
                K[j,i] = -1
            else:
                K[i,j] = -3
                K[j,i] = 3
        elif i < 2*N/3:
            if N/3 <= j < 2*N/3:
                K[i,j] = 1
                K[j,i] = 1
            elif j>= 2*N/3:
                K[i,j] = 2
                K[j,i] = -2
        else:
            if j >= 2*N/3:
                K[i,j] = 1
                K[j,i] = 1

print(K)