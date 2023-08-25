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

filename = 'G_noise0.20_phi1.0_K1.0_1.0_Rp1.0_xTy1.0'
# filename = 'C_noise0.20_phi1.0_K1.0_Rp1.0_xTy1.0'

file = os.path.abspath('../plots/p_order_vs_N/' + filename + '.txt')

with open(file) as f:
    reader = csv.reader(f, delimiter="\n")
    r = list(reader)

params = r[0][0].split('\t')[:-1]
rho = float(params[2])

n_list = [float(n) for n in r[1][0].split('\t')[:-3]]
psi_list = [float(p) for p in r[2][0].split('\t')[:-3]]
psi_sd_list = [float(p) for p in r[3][0].split('\t')[:-3]]
l_list = np.array([np.sqrt(n/rho) for n in n_list])

print(n_list)
print(psi_list)

def func(L, alpha, coeff, p_inf):
    return L**(-alpha)*np.exp(coeff) + p_inf

fig, ax = plt.subplots(figsize=(7,5))

# alpha, coeff, p_inf = curve_fit(func, l_list, psi_list, p0=[0.8654663306632585, -2.0327707413968774, 0.9597420767562417], maxfev=5000)[0]
alpha, coeff, p_inf = curve_fit(func, l_list, psi_list, sigma=psi_sd_list)[0]
print(alpha, coeff, p_inf)

# pcov = curve_fit(func, l_list, psi_list)[1]
# print(np.sqrt(np.diag(pcov))[0])


## Plot L vs Psi with fitted curve
# ax.plot(l_list, psi_list, 'o')
# # ax.errorbar(l_list, psi_list, yerr=psi_sd_list, fmt='o')
# x_plot = np.linspace(l_list[0],l_list[-1],100)
# ax.plot(x_plot, func(x_plot, alpha, coeff, p_inf))
# ax.set_xscale('log')
# ax.set_xlabel(r"$L$")
# ax.set_ylabel(r"$\Psi$")
# plt.show()

## Plot L vs Psi - Psi_inf with fitted line
# psi_2_list = np.array([p-p_inf for p in psi_list])
# # slope, coeff = np.polyfit(np.log(l_list), np.log(psi_2_list), 1)
# # print(slope, coeff)
# ax.loglog(l_list, psi_2_list, 'o')
# ax.loglog(l_list, np.exp(-alpha*np.log(l_list)+coeff), '--', label="Slope=" + str(round(-alpha,2)))
# ax.set_xlabel(r"$L$")
# ax.set_ylabel(r"$\Psi-\Psi_{\infty}$")
# ax.legend()
# plt.show()



# fig, ax = plt.subplots(figsize=(7,5))
# ax.plot(l_list, psi_list, 'o')
# x_plot = np.linspace(l_list[0],l_list[-1],100)
# ax.plot(x_plot, (x_plot**alpha) * np.exp(coeff) + p_inf)
# ax.set_xscale('log')
# ax.set_xlabel(r"$L$")
# ax.set_ylabel(r"$\Psi$")
# plt.show()


# plt.savefig(os.path.abspath('../plots/p_order_vs_N/psi_inf_fit.png'), bbox_inches="tight")
# plt.show()