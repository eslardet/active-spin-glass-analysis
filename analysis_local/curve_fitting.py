import numpy as np
from scipy.optimize import curve_fit
import os, sys, csv
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.family"] = "serif"
plt.rcParams['text.usetex'] = True
small = 12
big = 18

plt.rc('font', size=big)          # controls default text sizes
plt.rc('axes', labelsize=big)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
plt.rc('legend', fontsize=small)    # legend fontsize

# filename = 'G_noise0.20_phi1.0_K1.0_1.0_Rp1.0_xTy1.0'
# filename = 'G_noise0.20_phi1.0_K-0.1_8.0_Rp1.0_xTy1.0'
filename = 'G_noise0.20_phi1.0_K0.1_8.0_Rp1.0_xTy1.0'
# filename = 'G_noise0.20_phi1.0_K1.0_8.0_Rp1.0_xTy1.0'
# filename = 'C_noise0.20_phi1.0_K1.0_Rp1.0_xTy1.0'
# filename = 'G_noise0.20_phi1.0_K1.9_2.0_Rp1.0_xTy1.0'

force_alpha = False

file = os.path.abspath('../plots/p_order_vs_N/' + filename + '.txt')

with open(file) as f:
    reader = csv.reader(f, delimiter="\n")
    r = list(reader)

params = r[0][0].split('\t')[:-1]
rho = float(params[2])

n_list = [float(n) for n in r[1][0].split('\t')[:-1]][4:]
psi_list = [float(p) for p in r[2][0].split('\t')[:-1]][4:]
psi_sd_list = [float(p) for p in r[3][0].split('\t')[:-1]][4:]

# n_list = n_list[:4] + [n_list[-1]]
# psi_list = psi_list[:4] + [psi_list[-1]]
# psi_sd_list = psi_sd_list[:4] + [psi_sd_list[-1]]

l_list = np.array([np.sqrt(n/rho) for n in n_list])
# print(n_list)
# print(psi_list)

def func(L, alpha, coeff, p_inf):
    return L**(-alpha)*np.exp(coeff) + p_inf

alpha2 = 2/3
def func_2(L, coeff, p_inf):
    return L**(-alpha2)*np.exp(coeff) + p_inf

fig, ax = plt.subplots(figsize=(7,5))

# # alpha, coeff, p_inf = curve_fit(func, l_list, psi_list, p0=[0.8654663306632585, -2.0327707413968774, 0.9597420767562417], maxfev=5000)[0]
# alpha, coeff, p_inf = curve_fit(func, l_list, psi_list, sigma=psi_sd_list)[0]

if force_alpha == False:
    alpha, coeff, p_inf = curve_fit(func, l_list, psi_list)[0]
    print(alpha, coeff, p_inf)
else:
    coeff2, p_inf2 = curve_fit(func_2, l_list, psi_list, sigma=psi_sd_list)[0]
    print(coeff2, p_inf2)

if force_alpha == False:
    pcov = curve_fit(func, l_list, psi_list)[1]
    print(np.sqrt(np.diag(pcov))[0])
    print(np.sqrt(np.diag(pcov))[2])
else:   
    pcov = curve_fit(func_2, l_list, psi_list)[1]
    print(np.sqrt(np.diag(pcov))[1])
## Values for K_std=0.0
# alpha = 0.6811452537098752 
# coeff = -2.5243854611060774 
# p_inf = 0.9586729401024823

## Plot L vs Psi with fitted curve
ax.plot(l_list, psi_list, 'o-')
# ax.errorbar(l_list, psi_list, yerr=psi_sd_list, fmt='o-')
x_plot = np.linspace(l_list[0],l_list[-1],100)
if force_alpha == False:
    ax.plot(x_plot, func(x_plot, alpha, coeff, p_inf), label=r"$\alpha=$" + str(round(alpha,2)))
else:
    ax.plot(x_plot, func_2(x_plot, coeff2, p_inf2), label=r"$\alpha=2/3$")

# ax.legend()
ax.set_xscale('log')
ax.set_xlabel(r"$L$")
ax.set_ylabel(r"$\Psi$")
# plt.show()

## Plot L vs Psi - Psi_inf with fitted line
if force_alpha == False:
    psi_2_list = np.array([p-p_inf for p in psi_list])
else:
    psi_2_list = np.array([p-p_inf2 for p in psi_list])
ax_in = fig.add_axes([0.5, 0.5, 0.38, 0.35])
ax_in.loglog(l_list, psi_2_list, 'o')
if force_alpha == False:
    ax_in.loglog(l_list, np.exp(-alpha*np.log(l_list)+coeff), '--', label=r"Slope $=" + str(round(-alpha,2)) + r"$")
else:
    ax_in.loglog(l_list, np.exp(-alpha2*np.log(l_list)+coeff2), '--', label=r"Slope $=" + str(round(-alpha2,2)) + r"$")
ax_in.set_xlabel(r"$L$", fontsize=small)
ax_in.set_ylabel(r"$\Psi-\Psi_{\infty}$", fontsize=small)
ax_in.legend(frameon=False)
# ax.set_xbound(upper=10**3)
ax_in.set_ybound([5*10**-4, 10**-2])



# fig, ax = plt.subplots(figsize=(7,5))
# ax.plot(l_list, psi_list, 'o')
# x_plot = np.linspace(l_list[0],l_list[-1],100)
# ax.plot(x_plot, (x_plot**alpha) * np.exp(coeff) + p_inf)
# ax.set_xscale('log')
# ax.set_xlabel(r"$L$")
# ax.set_ylabel(r"$\Psi$")
# plt.show()


# plt.savefig(os.path.abspath('../plots/p_order_vs_N/psi_inf_fit.png'), bbox_inches="tight")
# plt.savefig(os.path.abspath('../plots/p_order_vs_N/psi_fit.png'), bbox_inches="tight")
plt.savefig(os.path.abspath('../plots/p_order_vs_N/' + filename + '.png'), bbox_inches="tight")
plt.show()