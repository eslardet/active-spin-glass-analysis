import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm, colors
from decimal import Decimal
import freud
import scipy.stats as sps
import bisect

import csv, os

num_Kstd = 8

file = os.path.abspath("plot/no_rep_full.txt")

with open(file) as f:
        reader = csv.reader(f, delimiter="\n")
        r = list(reader)

fig, ax = plt.subplots()
for k in range(num_Kstd):
    params = r[3*k][0].split('\t')
    K_std = params[3]

    K_avg = r[3*k+1][0].split('\t')[:-1]
    K_avg_plot = [float(i) for i in K_avg]
    
    p_ss = r[3*k+2][0].split('\t')[:-1]
    p_ss_plot = [float(i) for i in p_ss]

    ax.plot(K_avg_plot, p_ss_plot, "-o", label=r"$K_{STD}=$" + str(K_std))

params = r[3*k][0].split('\t')
nPart = params[0]
Rp = params[1]
noise = params[2]
rho = 1.0

ax.set_xlabel(r"$K_{AVG}$")
ax.set_ylabel(r"Polar order parameter, $\Psi$")
ax.set_ylim([0,1])
ax.set_title(r"$N=$" + str(nPart) + r"; $\rho=$" + str(rho) + r"; $\eta=$" + str(noise) + r"; $R_p=$" + str(Rp))
ax.legend()
plt.show()


