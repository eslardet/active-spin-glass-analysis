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

small = 22
big = 28

plt.rc('font', size=big)          # controls default text sizes
plt.rc('axes', labelsize=big)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)    # legend fontsize

matplotlib.rcParams["font.family"] = "serif"
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.labelpad']=10

fig, axs = plt.subplots(2,2, figsize=(24,16))
plt.subplots_adjust(wspace=0.2)

#### (a) ####
ax = axs[0][0]

filename = "phase_transition_RI_rho0.13"
file = os.path.abspath("plot_paper/data/" + filename + ".txt")
with open(file) as f:
    reader = csv.reader(f, delimiter="\n")
    r = list(reader)

num_RI = int((len(r))/3)

colors = plt.cm.BuPu(np.linspace(0.2, 1, num_RI))

ax.vlines(0, 0, 1, linestyle="dashed", color="black")

for k in range(num_RI):
    params = r[3*k][0].split('\t')
    # print(params)
    K_std = float(params[4])
    nPart = params[0]
    Rp = params[1]

    K_avg = r[3*k+1][0].split('\t')[:-1]
    K_avg_plot = [float(i) for i in K_avg]
    
    p_ss = r[3*k+2][0].split('\t')[:-1]
    p_ss_plot = [float(i) for i in p_ss]
    # ax.plot(K_avg_plot, p_ss_plot, "-o", color=colors[k], label=r"$\sigma_I=\ $" + str(Rp))
    if str(Rp) == "I":
        ax.plot(K_avg_plot, p_ss_plot, linestyle="dashdot", marker='v', label=r"$\sigma_I=\infty$", color="tab:red")
    else:
        ax.plot(K_avg_plot, p_ss_plot, "-o", color=colors[k], label=r"$\sigma_I=" + str(round(float(Rp))) + r"$")


ax.set_xlabel(r"$\overline{K}$")
ax.set_ylabel(r"$\Psi$")

ax.set_ylim([0,1])
ax.set_xlim([-1.0,1.0])

ax.legend(ncol=1, frameon=False, fontsize=18)
ax.text(-0.08, 1.0, "(a)", transform=ax.transAxes, va='top', ha='right')


#### (b) ####
ax = axs[0][1]

filename = "phase_transition_RI_rho0.13"
file = os.path.abspath("plot_paper/data/" + filename + ".txt")
with open(file) as f:
    reader = csv.reader(f, delimiter="\n")
    r = list(reader)

num_RI = int((len(r))/3)

file = os.path.abspath("plot_paper/data/" + filename + ".txt")

with open(file) as f:
        reader = csv.reader(f, delimiter="\n")
        r = list(reader)

RI_range = []
K_avg_crit = []

for k in range(num_RI):
    params = r[3*k][0].split('\t')

    if k < num_RI-1:
        RI = float(params[1])
        RI_range.append(RI)
    K_avg_range = r[3*k+1][0].split('\t')[:-1]
    K_avg_range = [float(i) for i in K_avg_range]
        
    p_ss = r[3*k+2][0].split('\t')[:-1]
    p_ss = [float(i) for i in p_ss]

    # ax.plot(K_avg_plot, p_ss_plot, "-o", color=colors[k], label=r"$F, K_{STD}=\ $" + str(K_std))

    cutoff = 0.2
    for i in range(len(p_ss)):
        if p_ss[i] > cutoff: # For a strictly increasing function
            break

    # Equation of line method (more accurate)
    grad = (p_ss[i]-p_ss[i-1])/(K_avg_range[i]-K_avg_range[i-1])
    intercept = p_ss[i] - grad*K_avg_range[i]

    KAVG_crit = (cutoff-intercept)/grad
    K_avg_crit.append(KAVG_crit)
    
L = np.sqrt(1000/0.13)

ax.plot(RI_range/L, K_avg_crit[:-1], '-o')
ax.hlines(K_avg_crit[-1], 0, 50, linestyle="dashdot", color="tab:red")
ax.text(0.05, 0.18, r"$\sigma_K=\infty$", color="tab:red")
ax.text(-0.08, 1.0, "(b)", transform=ax.transAxes, va='top', ha='right')

ax.set_xlim([0,0.6])
ax.set_ylabel(r"$\overline{K}_c$")
ax.set_xlabel(r"$\sigma_I/L$", labelpad=5)


#### (c) ####
ax = axs[1][0]

filename = "phase_transition_rho"
file = os.path.abspath("plot_paper/data/" + filename + ".txt")
with open(file) as f:
    reader = csv.reader(f, delimiter="\n")
    r = list(reader)

num_rho = int((len(r))/3)

colors = plt.cm.BuPu(np.linspace(0.2, 1, num_rho))

ax.vlines(0, 0, 1, linestyle="dashed", color="black")
for k in range(num_rho):
    params = r[3*k][0].split('\t')
    # print(params)
    K_std = float(params[4])
    nPart = params[0]
    Rp = params[1]
    rho = float(params[2])

    K_avg = r[3*k+1][0].split('\t')[:-1]
    K_avg_plot = [float(i) for i in K_avg]
    
    p_ss = r[3*k+2][0].split('\t')[:-1]
    p_ss_plot = [float(i) for i in p_ss]
    ax.plot(K_avg_plot, p_ss_plot, "-o", color=colors[k], label=r"$\rho=" + str(round(float(rho),2)) + r"$")


ax.set_xlabel(r"$\overline{K}$")
ax.set_ylabel(r"$\Psi$")
ax.set_ylim([0,1])
ax.set_xlim([-1.0,1.0])
ax.legend(frameon=False)
ax.text(-0.08, 1.0, "(c)", transform=ax.transAxes, va='top', ha='right')

#### (d) ####
ax = axs[1][1]
filename = "phase_transition_rho"
file = os.path.abspath("plot_paper/data/" + filename + ".txt")
with open(file) as f:
    reader = csv.reader(f, delimiter="\n")
    r = list(reader)

num_rho = int((len(r))/3)

file = os.path.abspath("plot_paper/data/" + filename + ".txt")

with open(file) as f:
        reader = csv.reader(f, delimiter="\n")
        r = list(reader)

rho_range = []
K_avg_crit = []

for k in range(num_rho):
    params = r[3*k][0].split('\t')

    rho = float(params[2])
    rho_range.append(rho)
    K_avg_range = r[3*k+1][0].split('\t')[:-1]
    K_avg_range = [float(i) for i in K_avg_range]
        
    p_ss = r[3*k+2][0].split('\t')[:-1]
    p_ss = [float(i) for i in p_ss]

    # ax.plot(K_avg_plot, p_ss_plot, "-o", color=colors[k], label=r"$F, K_{STD}=\ $" + str(K_std))

    cutoff = 0.2
    for i in range(len(p_ss)):
        if p_ss[i] > cutoff: # For a strictly increasing function
            break

    # Equation of line method (more accurate)
    grad = (p_ss[i]-p_ss[i-1])/(K_avg_range[i]-K_avg_range[i-1])
    intercept = p_ss[i] - grad*K_avg_range[i]

    KAVG_crit = (cutoff-intercept)/grad
    K_avg_crit.append(KAVG_crit)

nPart = 90000
ax.plot([np.sqrt(rho/nPart) for rho in rho_range], K_avg_crit, '-o')

ax.set_xlim(left=0)
ax.set_ylim(top=0)
ax.set_ylabel(r"$\overline{K}_c$")
ax.set_xlabel(r"$\sigma_I/L$")


ax.text(-0.08, 1.0, "(d)", transform=ax.transAxes, va='top', ha='right')

folder = os.path.abspath('../plots/for_figures/pt')
if not os.path.exists(folder):
    os.makedirs(folder)
filename = "phase_transition_RI_subplot"
plt.savefig(os.path.join(folder, filename + ".pdf"), bbox_inches='tight')

# plt.show()


