import numpy as np
import sys
sys.path.insert(1, '/Users/el2021/Code/2D_ActiveSpinGlass_EL/Active_Spin_Glass/analysis')
from analysis_functions import *
import time
import numba


@numba.jit()
def pbc_wrap_calc(x, L):
    """
    Wrap points into periodic box with length L (from -L/2 to L/2) for calculations
    """
    return x - L*np.round(x/L)

# Calculate distances between all particles
@numba.jit()
def get_particle_distances(nPart, phi, xTy, x, y):
    nPart = len(x)
    L = np.sqrt(nPart / (phi*xTy))
    Ly = L
    Lx = L*xTy
    D = np.zeros((nPart, nPart)) # Could do as vector instead
    for i in range(nPart):
        for j in range(i+1, nPart):
            xij = pbc_wrap_calc(x[i]-x[j], Lx)
            yij = pbc_wrap_calc(y[i]-y[j], Ly)
            rij = np.sqrt(xij**2 + yij**2)
            D[i,j] = rij
            D[j,i] = rij
    return D

@numba.jit()
def local_order_param(D, theta, r_max):
    nPart = len(theta)
    order_param = np.zeros(nPart)
    for i in range(nPart):
        idx = np.where(D[i,:] <= r_max)[0]
        theta_i = theta[idx]
        order_param[i] = np.sqrt(np.sum(np.cos(theta_i))**2 + np.sum(np.sin(theta_i))**2)/len(theta_i)

    return np.mean(order_param)

def global_order_param(theta):
    return np.sqrt(np.sum(np.cos(theta))**2 + np.sum(np.sin(theta))**2)/len(theta)

def local_order_param_all(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max_range):
    posExFile = get_file_path(mode, nPart, phi, noise, K, Rp, xTy, seed, file_name='pos_exact')
    x, y, theta, viewtime = get_pos_ex_snapshot(posExFile)
    x = np.array(x)
    y = np.array(y)
    theta = np.array(theta)

    D = get_particle_distances(nPart, phi, xTy, x, y)

    o_plot = []
    for r_max in r_max_range:
        o_plot.append(local_order_param(D, theta, r_max))

    g = global_order_param(theta)

    return o_plot, g

def plot_local_order_vs_l(mode, nPart, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed_range, r_max_range, show_g=False):
    fig, ax = plt.subplots()

    for K_avg in K_avg_range:
        for K_std in K_std_range:
            K = str(K_avg) + "_" + str(K_std)
            o_plot_all = []
            g_all = []
            for seed in seed_range:
                o_plot, g = local_order_param_all(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max_range)
                o_plot_all.append(o_plot)
                g_all.append(g)
            o_plot_mean = np.mean(o_plot_all, axis=0)
            g_mean = np.mean(g_all)
            ax.plot(r_max_range, o_plot_mean, '-o', label = r"$\overline{K}=$"+ str(K_avg) + r", $\sigma_K=$" + str(K_std))
            if show_g == True:
                ax.hlines(g_mean, 0, r_max_range[-1], linestyle='dashed', color='gray')

    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\Psi(\ell)$')
    ax.legend()

    folder = os.path.abspath('../plots/p_order_local_vs_l/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '.png'
    # filename = "N10000_K0.0_8.0_loglog.png"
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))
    plt.show()


mode = "G"
nPart = 10000
phi = 1.0
noise = "0.20"
K = "0.0_8.0"
Rp = 1.0
xTy = 1.0
seed = 1

K_avg_range = [0.0]
K_std_range = [7.0]
seed_range = np.arange(1,21,1)
r_max_range = np.arange(0,21,1)

plot_local_order_vs_l(mode, nPart, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed_range, r_max_range, show_g=True)

# t0 = time.time()
# posExFile = get_file_path(mode, nPart, phi, noise, K, Rp, xTy, seed, file_name='pos_exact')
# x, y, theta, viewtime = get_pos_ex_snapshot(posExFile)
# x = np.array(x)
# y = np.array(y)
# theta = np.array(theta)
# D = get_particle_distances(nPart, phi, xTy, x, y)

# print(time.time() - t0)


# fig, ax = plt.subplots()

# # nPart = 1000
# # for K_avg in [-1.0, 0.0, 1.0]:
# #     K = str(K_avg) + "_8.0"
# #     r_max_range = np.arange(0,31,1)
# #     o_plot, g = local_order_param_all(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max_range)
# #     ax.plot(r_max_range, o_plot, '-o', label = "K = " + K + ", ordered")
# #     ax.hlines(g, 0, 30, linestyle='dashed', color='gray')

# colors = plt.cm.BuPu(np.linspace(0.2, 1, 9))
# nPart = 10000
# for Kstd in np.arange(8.0, 8.1, 1.0):
#     K = "0.0_" + str(Kstd)
#     # r_max_range = np.concatenate((np.arange(0,31,1), np.arange(31,101,10)))
#     r_max_range = np.arange(0,21,1)
#     o_plot, g = local_order_param_all(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max_range)
#     # o_plot = [o-g for o in o_plot]
#     ax.plot(r_max_range, o_plot, '-o', label = "K = " + K, color = colors[int(Kstd)])
#     ax.hlines(g, 0, r_max_range[-1], linestyle='dashed', color='gray')

# # ax.set_yscale('log')
# # ax.set_xscale('log')
# ax.set_xlabel(r'$\ell$')
# ax.set_ylabel(r'$\Psi(\ell)$')
# ax.legend()


# folder = os.path.abspath('../plots/p_order_local_vs_l/')
# # filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_Kavg' + str(K_avg)+ '_Kstd' + str(K_std) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '.png'
# filename = "N10000_K0.0_8.0_loglog.png"
# if not os.path.exists(folder):
#     os.makedirs(folder)
# plt.savefig(os.path.join(folder, filename))

# plt.show()
