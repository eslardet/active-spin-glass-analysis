import numpy as np
from analysis.analysis_functions import *
import matplotlib.pyplot as plt
from matplotlib import cm, colors


# 0.60 
K_crit_list = [0.52, 0.51747787627541, 0.48204703428778123, 0.44709777838820564, 0.38430423876538755,
                0.33215852575542254, 0.2740370748833158, 0.22053303218349699, 0.167170987252005, 0.06301664791932983, -0.378394607286131, -0.763178508380671]

K_std_range = [0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 20.0, 100.0]
K_std_range_plot = [np.log(k) for k in K_std_range]


slope, intercept = np.poly1d(np.polyfit(K_std_range, K_crit_list, 1)).c
print(slope)
fig, ax = plt.subplots()

slope, intercept = np.poly1d(np.polyfit(K_std_range[1:-1], K_crit_list[1:-1], 1)).c
K_crit_list_plot = [np.log((k-intercept)/slope) for k in K_crit_list]
# ax.plot(K_std_range_plot, K_crit_list_plot, '-o', label=r"$\eta = $" + str(0.60))
ax.plot(K_std_range_plot, [np.log(k) for k in K_crit_list], '-o', label=r"$\eta = $" + str(0.60))

# ax.plot(K_std_range_plot, K_std_range_plot, '--', color="grey", label="Slope = 1")
ax.set_xlabel(r"$\log(K_{STD})$")
ax.set_ylabel(r"$\log(\tilde{K}_{AVG}^C)$")
ax.legend()
# ax.set_xscale('log')
# ax.set_yscale('log')

plt.show()

# # for i, noise in enumerate(noise_range):
# #     ax.plot(K_std_range, K_crit[i], '-o', label=r"$\eta = $" + str(noise))

# ax.plot(K_std_range, K_crit[2], '-o', label=r"$\eta=0.60")

# ax.plot(np.unique(K_std_range), np.poly1d(np.polyfit(K_std_range, K_crit[2], 1))(np.unique(K_std_range)), label="Line of best fit")
# # ax.plot(K_std_range, [-i/14 + 0.1 for i in K_std_range])
# # ax.plot(K_std_range, [-i/14 + 0.3 for i in K_std_range])
# # ax.plot(K_std_range, [-i/18 + 0.6 for i in K_std_range])
# # ax.plot(K_std_range, [-i/20 + 1.1 for i in K_std_range])

# ax.set_xlabel(r"$K_{STD}$")
# ax.set_ylabel(r"$K_{AVG}^C$")
# ax.legend()

# plt.show()