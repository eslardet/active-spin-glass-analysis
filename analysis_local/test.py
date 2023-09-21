import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import time
from scipy.signal import fftconvolve, correlate

# n = 5
# x = np.arange(0, n, 1)
# y = np.arange(n, 2*n, 1)

# print(x)
# print(y)
# avs = np.arange(n,0,-1)
# corr_x = correlate(x,x, mode='full')[n-1:] / avs
# corr_y = correlate(y,y, mode='full')[n-1:] / avs

# corr = corr_x + corr_y
# corr = corr / corr[0]

# print(corr)



# t0 = time.time()
# tile_x = np.tile(x, (n,1))
# tile_y = np.tile(y, (n,1)).T

# dist = np.sqrt(tile_x**2 + tile_y**2)
# print(dist)
# print(time.time()-t0)


# corr = np.round(fftconvolve(x,y[::-1,::-1], mode='full')[n-1:,n-1:],0)
# avs = np.outer(np.arange(h,0,-1), np.arange(w,0,-1))
# if take_av == True:
#     corr = corr/avs
# corr = corr/corr[0,0]

print([format(i, '.2f') for i in np.arange(-0.65,-0.50,0.01)])