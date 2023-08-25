import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


x = [0,0,0,0]
y = [1,3,4,5]

nPart = len(x)
total_dist = 0

def pbc_wrap(x, L):
    """
    Wrap points into periodic box with length L
    """
    return x - L*np.round(x/L)

# print(pbc_wrap(4,5))

L = 5

for i in range(nPart):
    min_dist = np.infty # initialise min_dist benchmark
    for j in range(nPart):
        if i != j:
            xij = pbc_wrap(x[i]-x[j],L)
            # xij = x[i] - x[j]
            if np.abs(xij) < min_dist:
                yij = pbc_wrap(y[i]-y[j],L)
                # yij = y[i] - y[j]
                if np.abs(yij) < min_dist:
                    rij = np.sqrt(xij**2+yij**2)
                    if rij < min_dist:
                        min_dist = rij
    total_dist += min_dist

mean_dist_nn = total_dist/nPart

print(mean_dist_nn)