import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm, colors
from decimal import Decimal
import freud
import scipy.stats as sps
import bisect

import csv, os

num_Kstd = 4

file = os.path.abspath("plot/rep_Rp_phi0.1.txt")

with open(file) as f:
        reader = csv.reader(f, delimiter="\n")
        r = list(reader)

fig, ax = plt.subplots()
for k in range(num_Kstd):
    params = r[3*k][0].split('\t')
    K_std = params[3]
    Rp = params[1]

    K_avg = r[3*k+1][0].split('\t')[:-1]
    K_avg_plot = [float(i) for i in K_avg]
    
    p_ss = r[3*k+2][0].split('\t')[:-1]
    p_ss_plot = [float(i) for i in p_ss]

    ax.plot(K_avg_plot, p_ss_plot, "-o", label=r"$R_p=$" + str(Rp))
    # ax.plot(K_avg_plot, p_ss_plot, "-o")

params = r[3*k][0].split('\t')
nPart = params[0]
Rp = params[1]
noise = params[2]
Kstd = params[3]
# rho = 1.0
phi = 0.1

ax.set_xlabel(r"$K_{AVG}$")
ax.set_ylabel(r"Polar order parameter, $\Psi$")
ax.set_ylim([0,1])
ax.set_title(r"$N=$" + str(nPart) + r"; $\phi=$" + str(phi) + r"; $\eta=$" + str(noise) + r"; $K_{STD}=$" + str(Kstd))
ax.legend(loc="lower right")
plt.show()


