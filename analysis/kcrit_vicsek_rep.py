import numpy as np
import analysis_functions_vicsek as fun
import matplotlib.pyplot as plt
from matplotlib import cm, colors

K_crit = []

# 0.20
K_crit.append([0.15441575061332655,0.25138932302165573,0.5809097989546506,0.6525134826375487,
               0.9091268188747478,1.1838375802760692,1.4224331238594516,1.58267673488568])


# 0.60 
K_crit.append([0.6413101301156421,0.6955772329297512,0.845226149852214,1.1079428916063725,
               1.3128064292371968,1.5574155061297836,1.7505368586773973,2.0174884897241])


K_std_range = np.arange(1,9,1)
noise_range = np.round(np.arange(0.2, 0.65, 0.4),2)


fig, ax = plt.subplots()

for i, noise in enumerate(noise_range):
    ax.plot(K_std_range, K_crit[i], '-o', label=r"$\eta = $" + str(noise))

ax.plot(K_std_range, [i/5 + 0.3 for i in K_std_range])
# ax.plot(K_std_range, [-i/14 + 0.3 for i in K_std_range])
# ax.plot(K_std_range, [-i/18 + 0.6 for i in K_std_range])
# ax.plot(K_std_range, [-i/20 + 1.1 for i in K_std_range])

ax.set_xlabel(r"$K_{STD}$")
ax.set_ylabel(r"$K_{AVG}^C$")
ax.legend()

plt.show()