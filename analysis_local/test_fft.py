import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import time
from scipy.signal import fftconvolve, correlate2d, convolve2d

n = 5
m = 5
x = np.arange(1, n+1, 1)
y = np.arange(1, m+1, 1)
A = np.outer(x, y)

# t0 = time.time()
# corr2d = correlate2d(A,A, mode='full', boundary='wrap')[n-1:,m-1:]
# print("correlate2d (control) ", time.time()-t0, "\n")

# t0 = time.time()
# sci_fft = np.round(fftconvolve(A,A[::-1,::-1], mode='same'),0)
# print("fftconvolve ", time.time()-t0, np.all(sci_fft==corr2d), "\n")

t0 = time.time()
# conv2d = convolve2d(A,A[::-1,::-1], mode='full', boundary='wrap')[n-1:,m-1:]
# conv2d = convolve2d(A,A[::-1,::-1], mode='same', boundary='wrap')
# print("convolve2d ", time.time()-t0, np.all(conv2d==corr2d), "\n")
# print(conv2d)

## Winner in time and accuracy!
t0 = time.time()
a=np.fft.fft2(A)
b=np.fft.fft2(A[::-1,::-1])
c=np.real(np.fft.ifft2(a*b))[::-1,::-1]
np_fft = np.round(c,0)
# print("np.fft ", time.time()-t0, np.all(np_fft==corr2d), "\n")
# print(np_fft)

# corr = corr/corr[0,0]

def pbc_wrap_calc(x, L):
    """
    Wrap points into periodic box with length L (from 0 to L) for display
    """
    return x - L*np.round(x/L)

n=3
m=4

# t0 = time.time()
x = pbc_wrap_calc(np.tile(np.arange(0,m), (n,1)),m)
# y = x.transpose()
y = pbc_wrap_calc(np.tile(np.arange(0,n), (m,1)),n).T
# x = pbc_wrap_calc(x, n)
# y = pbc_wrap_calc(y, n)
dist_0 = np.sqrt(x**2+y**2)

# ngridx = 3
# ngridy = 4
# x = pbc_wrap_calc(np.tile(np.arange(0,ngridy), (ngridx,1)),ngridy)
# y = pbc_wrap_calc(np.tile(np.arange(0,ngridx), (ngridy,1)),ngridx).T
# dist_0 = np.sqrt(x**2+y**2)
# print("pbc_wrap_calc ", time.time()-t0, "\n")

t0 = time.time()
dist_1 = np.zeros((n,m))
for i in range(n):
    for j in range(m):
        dist_1[i,j] = np.sqrt(pbc_wrap_calc((i),n)**2+pbc_wrap_calc((j),m)**2)

print("for loop ", time.time()-t0, "\n")
print(np.all(dist_1==dist_0))
