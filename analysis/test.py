import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


for K in np.arange(-1.0,1.0,0.1):
    print(np.round(K,1), np.round(1-norm.cdf(x=0, loc=K, scale=8.0),2))