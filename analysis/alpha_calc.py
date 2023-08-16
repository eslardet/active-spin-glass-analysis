import sys
import numpy as np
from scipy.stats import norm
from scipy.special import erf

# K_avg = float(sys.argv[1])
# K_std = float(sys.argv[2])
K_avg = -0.5
K_std = 1

alpha = 1 - norm(K_avg, K_std).cdf(0)
print(alpha)
# exit(str(alpha))

alpha = 1/2*(1-erf(-K_avg/(np.sqrt(2)*K_std)))
print(alpha)