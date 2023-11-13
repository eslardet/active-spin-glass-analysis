import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm, colors
from decimal import Decimal
import freud
import scipy.stats as sps
import bisect

import csv, os

num_Kstd = 9
cutoff = 0.2
KAVG_crit = []
Rp_plot = []
save_plot = False
filename = "phase_transition"

file = os.path.abspath("plot/data/" + filename + ".txt")

with open(file) as f:
        reader = csv.reader(f, delimiter="\n")
        r = list(reader)

# fig, ax = plt.subplots()
for k in range(num_Kstd):
    params = r[3*k][0].split('\t')
    # print(params)
    K_std = params[4]
    nPart = params[0]
    Rp = params[1]

    K_avg = r[3*k+1][0].split('\t')[:-1]
    K_avg_plot = [float(i) for i in K_avg]
    
    p_ss = r[3*k+2][0].split('\t')[:-1]
    p_ss_plot = [float(i) for i in p_ss]

    for i in range(len(p_ss)):
        if p_ss_plot[i] > cutoff: # For a strictly increasing function
            break

    # Midpoint method
    # KAVG_crit = (KAVG_range[i] + KAVG_range[i-1])/2

    # Equation of line method (more accurate)
    grad = (p_ss_plot[i]-p_ss_plot[i-1])/(K_avg_plot[i]-K_avg_plot[i-1])
    intercept = p_ss_plot[i] - grad*K_avg_plot[i]

    cr = (cutoff-intercept)/grad
    KAVG_crit.append(cr)

    print(K_std, cr)


# plt.show()


