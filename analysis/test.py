import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt



a = np.random.normal(0, 2, 10000)
print(np.average(a, weights=a>0))
print(np.average(a, weights=a<0))