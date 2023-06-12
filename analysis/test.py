import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


std = np.arange(1, 9, 1)
avg = np.arange(0, -0.8, -0.1)

val = []
for i in range(8):
    val.append(1-norm.cdf(0, loc=avg[i], scale=std[i]))

plt.plot(std, val, "o-")
plt.show()

# for l in np.arange(-5, 6, 1):
#     val = []
#     for s in std:
#         val.append(1-norm.cdf(0, loc=l, scale=s))

#     plt.plot(std, val, "-o", label=l)
# # plt.yscale("log")
# # plt.xscale("log")
# plt.legend()
# plt.show()