import sys
sys.path.insert(1, './analysis/analysis_functions')
from local_order_grid import *

import numpy as np
import time
import numba
import seaborn as sns

@numba.jit(nopython=True)
def pbc_wrap(x, L):
    """
    Wrap points into periodic box with length L (from 0 to L) for display
    """
    return x - L*np.round(x/L) + L/2

@numba.jit(nopython=True)
def pbc_wrap_calc(x, L):
    """
    Wrap points into periodic box with length L (from -L/2 to L/2) for calculations
    """
    return x - L*np.round(x/L)

@numba.jit(nopython=True)
def get_distance_matrix(ngridx, ngridy, min_grid_size):
    """
    Output matrix is distance shift matrix in terms of x, y distance wrapped by number of grid points
    """
    x = pbc_wrap_calc(np.tile(np.arange(0,ngridy), (ngridx,1)),ngridy)*min_grid_size
    y = pbc_wrap_calc(np.tile(np.arange(0,ngridx), (ngridy,1)),ngridx).T*min_grid_size
    dist = np.sqrt(x**2+y**2)
    return dist

def local_order_param_grid(x, y, theta, nPart, phi, xTy, min_grid_size=1):

    L = np.sqrt(nPart / (phi*xTy))
    Ly = L
    Lx = L*xTy

    ngridx = int(Lx // min_grid_size)
    ngridy = int(Ly // min_grid_size)

    grid_size_x = Lx / ngridx
    grid_size_y = Ly / ngridy

    count_arr = np.zeros((ngridx, ngridy))
    sin_sum = np.zeros((ngridx, ngridy))
    cos_sum = np.zeros((ngridx, ngridy))
    for i in range(nPart):
        ix = int(pbc_wrap(x[i],Lx) // grid_size_x)
        iy = int(pbc_wrap(y[i],Ly) // grid_size_y)
        count_arr[ix, iy] += 1
        sin_sum[ix, iy] += np.sin(theta[i])
        cos_sum[ix, iy] += np.cos(theta[i])
    sum = np.sqrt(sin_sum**2 + cos_sum**2).flatten()
    counts = count_arr.flatten()
    order_param = sum[counts!=0]/counts[counts!=0]
    # order_param = np.divide(np.sqrt(sin_sum**2 + cos_sum**2),count_arr, where=count_arr!=0)
    return order_param

@numba.jit(nopython=True)
def global_order_param(theta):
    return np.sqrt(np.sum(np.cos(theta))**2 + np.sum(np.sin(theta))**2)/len(theta)

def local_order_param_mean(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max_range):
    posExFile = get_file_path(mode, nPart, phi, noise, K, Rp, xTy, seed, file_name='pos_exact')
    x, y, theta, viewtime = get_pos_ex_snapshot(posExFile)
    x = np.array(x)
    y = np.array(y)
    theta = np.array(theta)

    o_mean = []
    for r_max in r_max_range:
        o_mean.append(np.mean(local_order_param_grid(x, y, theta, nPart, phi, xTy, min_grid_size=r_max)))

    return o_mean

def local_order_param_all(mode, nPart, phi, noise, K, Rp, xTy, seed_range, r_max):
    o_all = []
    for seed in seed_range:
        posExFile = get_file_path(mode, nPart, phi, noise, K, Rp, xTy, seed, file_name='pos_exact')
        x, y, theta, viewtime = get_pos_ex_snapshot(posExFile)
        x = np.array(x)
        y = np.array(y)
        theta = np.array(theta)
        o_all += local_order_param_grid(x, y, theta, nPart, phi, xTy, min_grid_size=r_max).flatten().tolist()
    return o_all

def plot_local_order_vs_l(mode, nPart, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed_range, r_max_range):
    fig, ax = plt.subplots()
    colors = plt.cm.BuPu(np.linspace(0.2, 1, len(K_std_range)*len(K_avg_range)))
    i = 0
    for K_avg in K_avg_range:
        for K_std in K_std_range:
            K = str(K_avg) + "_" + str(K_std)
            o_plot_all = []
            for seed in seed_range:
                o_plot = local_order_param_mean(mode, nPart, phi, noise, K, Rp, xTy, seed, r_max_range)
                o_plot_all.append(o_plot)
            o_plot_mean = np.mean(o_plot_all, axis=0)
            ax.plot(r_max_range, o_plot_mean, '-o', label = r"$\overline{K}=$"+ str(K_avg) + r", $\sigma_K=$" + str(K_std), color=colors[i])
            i += 1
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\Psi(\ell)$')
    ax.legend(loc="lower right")

    folder = os.path.abspath('../plots/p_order_local_grid_vs_l/')
    filename = mode + '_N' + str(nPart) + '_phi' + str(phi) + '_Rp' + str(Rp) + '_xTy' + str(xTy) + '.png'
    # filename = "N10000_K0.0_8.0_loglog.png"
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))
    plt.show()

def plot_local_order_hist(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed_range, r_max_range):
    fig, ax = plt.subplots()
    for r_max in r_max_range:
        K = str(K_avg) + "_" + str(K_std)
        o_list = local_order_param_all(mode, nPart, phi, noise, K, Rp, xTy, seed_range, r_max)
        # ax.hist(o_list, bins=100, range=(0,1), density=True, label = r"$\overline{K}=$" + str(K_avg) + r", $\sigma_K=$" + str(K_std) + r", $\ell=$" + str(r_max), alpha=0.5)
        sns.histplot(o_list, bins=100, binrange=(0,1), stat='probability', kde=True, label = r"$\overline{K}=$" + str(K_avg) + r", $\sigma_K=$" + str(K_std) + r", $\ell=$" + str(r_max), alpha=0.5)
    ax.legend()
    ax.set_xlabel(r'$\Psi(\ell)$')
    ax.set_ylabel(r'$P(\Psi(\ell))$')
    ax.set_ylim(0, 0.2)

    folder = os.path.abspath('../plots/p_order_local_grid_hist/')
    filename = "N" + str(nPart) + "_K" + str(K) + ".png"
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

K_avg = 0.0
K_std = 8.0
seed_range = np.arange(1,2,1)
r_max_range = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
# plot_local_order_hist(mode, nPart, phi, noise, K_avg, K_std, Rp, xTy, seed_range, r_max_range)


# nPart = 1000
K_avg_range = [0.0]
K_std_range = np.arange(0.0, 8.1, 1.0)
# K_avg_range = [-1.0, 0.0, 1.0]
# K_std_range = [8.0]
r_max_range = np.arange(1,21,1)
plot_local_order_vs_l(mode, nPart, phi, noise, K_avg_range, K_std_range, Rp, xTy, seed_range, r_max_range)