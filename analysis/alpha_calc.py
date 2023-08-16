import sys
import numpy as np
from scipy.stats import norm
from scipy.special import erf

# K_avg = float(sys.argv[1])
# K_std = float(sys.argv[2])
K_avg = 0
K_std = 1

alpha = 1 - norm(K_avg, K_std).cdf(0)
print(alpha)
# exit(str(alpha))

alpha = 1/2*(1-erf(-K_avg/(np.sqrt(2)*K_std)))
print(alpha)

alpha = 1/2 - 1/2 * erf(-K_avg/(np.sqrt(2)*K_std))
K0 = K_avg + K_std*np.sqrt((1-alpha)/alpha)
K1 = K_avg - K_std*np.sqrt(alpha/(1-alpha))

print(K0,K1)

print(alpha*K0+K1*(1-alpha))
print(np.abs(K0-K1)*np.sqrt(alpha*(1-alpha)))

samples = []
for i in range(10000):
    if np.random.rand() < alpha:
        K = K0
    else:
        K = K1
    samples.append(K)

print(np.mean(samples))
print(np.std(samples))