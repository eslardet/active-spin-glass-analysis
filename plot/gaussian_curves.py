import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.stats import norm
import os

small = 18
medium = 22
big = 28

plt.rc('font', size=big)          # controls default text sizes
plt.rc('axes', labelsize=big)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=medium)    # fontsize of the tick labels
plt.rc('ytick', labelsize=medium)    # fontsize of the tick labels
plt.rc('legend', fontsize=medium)    # legend fontsize

matplotlib.rcParams["font.family"] = "serif"
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.labelpad']=10

fig, ax = plt.subplots(figsize=(12,6))

x_plot = np.arange(-5, 5, 0.001)
plt.vlines(0,0.0, 0.50, colors="black", linestyles="dashed")
plt.plot(x_plot, norm.pdf(x_plot, 0, 1), label=r"$\overline{K}=0; \sigma_K=1$")
plt.plot(x_plot, norm.pdf(x_plot, 0, 2), label=r"$\overline{K}=0; \sigma_K=2$")
plt.plot(x_plot, norm.pdf(x_plot, 1, 1), label=r"$\overline{K}=1;  \sigma_K=1$")

plt.ylabel("Probability Density")
plt.xlabel(r"$K_{ij}$")

plt.ylim(bottom=0)
plt.xlim(left=-5, right=5)
plt.legend(loc="upper left")

filename = "gaussian_curves"
folder = os.path.abspath('../plots/for_figures/poster')
if not os.path.exists(folder):
    os.makedirs(folder)
plt.savefig(os.path.join(folder, filename + ".png"), bbox_inches="tight")
plt.show()