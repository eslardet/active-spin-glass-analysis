import sys
import numpy as np
from scipy.stats import norm
from scipy.special import erf

# K_avg = float(sys.argv[1])
# K_std = float(sys.argv[2])
K_avg = -0.3
K_std = 6

# alpha = 1 - norm(K_avg, K_std).cdf(0)
# print(alpha)
# exit(str(alpha))

for K_avg in np.round(np.arange(-1.0,1.1,0.1),1):

    alpha = 1/2*(1-erf(-K_avg/(np.sqrt(2)*K_std)))


    alpha = 1/2 - 1/2 * erf(-K_avg/(np.sqrt(2)*K_std))
    K0 = K_avg + K_std*np.sqrt((1-alpha)/alpha)
    K1 = K_avg - K_std*np.sqrt(alpha/(1-alpha))

    print(K_avg, K0,K1,alpha)

# print(alpha*K0+K1*(1-alpha))
# print(np.abs(K0-K1)*np.sqrt(alpha*(1-alpha)))

# samples = []
# for i in range(10000):
#     if np.random.rand() < alpha:
#         K = K0
#     else:
#         K = K1
#     samples.append(K)

# print(np.mean(samples))
# print(np.std(samples))



# alpha = 0.485
# K0 = -8.06
# K1 = 7.94

# K_avg = alpha*K1 + (1-alpha)*K0
# K_std = np.abs(K1-K0)*np.sqrt(alpha*(1-alpha))

# print(K_avg, K_std)