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

print(get_sim_dir(mode="G", nPart=1000, phi=1.0, noise="0.20", K="1.0_1.0", Rp=1.0, xTy=1.0, seed=1))

# n = 50
# x = np.arange(1, n+1, 1)
# A = np.outer(x, x)

# # t0 = time.time()
# # corr2d = correlate2d(A,A, mode='full', boundary='wrap')[n-1:,n-1:]
# # print("correlate2d (control) ", time.time()-t0, "\n")

# # t0 = time.time()
# # sci_fft = np.round(fftconvolve(A,A[::-1,::-1], mode='same'),0)
# # print("fftconvolve ", time.time()-t0, np.all(sci_fft==corr2d), "\n")

# # t0 = time.time()
# # conv2d = convolve2d(A,A[::-1,::-1], mode='full', boundary='wrap')[n-1:,n-1:]
# # print("convolve2d ", time.time()-t0, np.all(conv2d==corr2d), "\n")

# ## Winner in time and accuracy!
# t0 = time.time()
# a=np.fft.fft2(A)
# b=np.fft.fft2(A[::-1,::-1])
# c=np.real(np.fft.ifft2(a*b))[::-1,::-1]
# np_fft = np.round(c,0)
# # print("np.fft ", time.time()-t0, np.all(np_fft==corr2d), "\n")

# avs = np.outer(np.arange(n,0,-1), np.arange(n,0,-1))
# # if take_av == True:
# #     corr = corr/avs
# # corr = corr/corr[0,0]

# # correlate2d 

a = []

b = np.array(([1,2,3,4],[1,2,3,4])).flatten()
c = np.array(([2,3,4,0],[1,2,3,0])).flatten()

print(b[np.where(c!=0)]/c[np.where(c!=0)])


# print(np.divide(b,c, where=c!=0))


# print(np.mean(a, axis=0))